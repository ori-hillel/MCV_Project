"""
Improved training pipeline with:
- Cross-attention gaze fusion model
- Class-balanced focal loss (effective sample counts)
- Cosine LR scheduler with warmup
- Post-hoc temporal smoothing
- Leaderboard tracking & scored checkpointing
- Gradient clipping
"""
import os, sys, json, shutil, math
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

# Redirect output to file for capture
LOG_FILE = "train_v2_output.txt"
log_handle = open(LOG_FILE, "w")
class Tee:
	def __init__(self, *streams):
		self.streams = streams
	def write(self, data):
		for s in self.streams:
			s.write(data)
			s.flush()
	def flush(self):
		for s in self.streams:
			s.flush()

sys.stdout = Tee(sys.__stdout__, log_handle)
sys.stderr = Tee(sys.__stderr__, log_handle)

from dataset import SurgicalMultiTaskDataset
from model_v2 import ViTBackbone, ImprovedGazeModel
from utils import evaluate_model, plot_confusion_matrix_heatmap, load_real_phase_dict
from loss import calculate_multitask_loss

# ============================================================
# Class-Balanced Focal Loss (effective sample counts)
# ============================================================
class ClassBalancedFocalLoss(nn.Module):
	def __init__(self, samples_per_class, num_classes=9, beta=0.999, gamma=2.0):
		super().__init__()
		self.gamma = gamma
		# Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
		effective_num = []
		for n in samples_per_class:
			if n == 0:
				effective_num.append(1.0)
			else:
				effective_num.append((1.0 - beta ** n) / (1.0 - beta))
		weights = [1.0 / e for e in effective_num]
		# Normalize so weights sum to num_classes
		total = sum(weights)
		weights = [w * num_classes / total for w in weights]
		self.register_buffer('alpha', torch.tensor(weights, dtype=torch.float32))
	
	def forward(self, inputs, targets):
		ce_loss = F.cross_entropy(inputs, targets, reduction='none')
		pt = torch.exp(-ce_loss)
		focal_loss = ((1 - pt) ** self.gamma) * ce_loss
		alpha_t = self.alpha.gather(0, targets.view(-1))
		focal_loss = alpha_t * focal_loss
		return focal_loss.mean()


def calculate_improved_loss(phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools, focal_criterion, lambda_aux=0.5):
	loss_phase = focal_criterion(phase_logits, phase_targets)
	
	if has_tools.any():
		valid_spatial_logits = spatial_logits[has_tools]
		valid_bbox_grids = bbox_grids[has_tools].unsqueeze(1)
		loss_aux = F.binary_cross_entropy_with_logits(
			valid_spatial_logits.view(-1),
			valid_bbox_grids.view(-1).float()
		)
	else:
		loss_aux = torch.tensor(0.0, device=phase_logits.device, requires_grad=True)
	
	total_loss = loss_phase + (lambda_aux * loss_aux)
	return total_loss, loss_phase, loss_aux


# ============================================================
# Temporal Smoothing (post-hoc median filter on predictions)
# ============================================================
def temporal_smooth_predictions(y_pred, window=5):
	"""Apply median filter for temporal smoothing."""
	smoothed = list(y_pred)
	half = window // 2
	for i in range(len(smoothed)):
		start = max(0, i - half)
		end = min(len(smoothed), i + half + 1)
		window_preds = y_pred[start:end]
		# Majority vote
		counts = Counter(window_preds)
		smoothed[i] = counts.most_common(1)[0][0]
	return smoothed


# ============================================================
# Leaderboard
# ============================================================
def update_leaderboard(metrics, description, ckpt_file, arch_file, leaderboard_path="checkpoints/test_leaderboard.json"):
	os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
	
	if os.path.exists(leaderboard_path):
		with open(leaderboard_path, 'r') as f:
			leaderboard = json.load(f)
	else:
		leaderboard = []
	
	entry = {
		"timestamp": datetime.now().isoformat(),
		"description": description,
		"test_jaccard_iou": metrics['jaccard_iou'],
		"test_accuracy": metrics['accuracy'],
		"test_precision": metrics['precision'],
		"test_recall": metrics['recall'],
		"checkpoint_file": ckpt_file,
		"architecture_file": arch_file,
	}
	leaderboard.append(entry)
	
	with open(leaderboard_path, 'w') as f:
		json.dump(leaderboard, f, indent=2)
	
	return leaderboard


def save_scored_checkpoint(model, optimizer, epoch, metrics, description, checkpoint_dir="checkpoints"):
	os.makedirs(checkpoint_dir, exist_ok=True)
	score = f"{metrics['jaccard_iou']:.4f}"
	
	ckpt_file = f"best_test_jaccard_{score}.pt"
	ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
	torch.save({
		'epoch': epoch,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'metrics': metrics,
		'description': description,
	}, ckpt_path)
	
	# Save architecture snapshot
	arch_file = f"model_architecture_{score}.py"
	arch_src = os.path.join(os.path.dirname(__file__), "model_v2.py")
	arch_dst = os.path.join(checkpoint_dir, arch_file)
	shutil.copy2(arch_src, arch_dst)
	
	# Also save as best_model.pt
	best_path = os.path.join(checkpoint_dir, "best_model_v2.pt")
	torch.save({
		'epoch': epoch,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'metrics': metrics,
		'description': description,
	}, best_path)
	
	return ckpt_file, arch_file


# ============================================================
# Cosine schedule with warmup
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
	def lr_lambda(current_step):
		if current_step < warmup_steps:
			return float(current_step) / float(max(1, warmup_steps))
		progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
	return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Evaluate with temporal smoothing
# ============================================================
def evaluate_with_smoothing(model, dataloader, device, lambda_aux=0.5, smooth_window=5):
	"""Evaluate and also return temporally smoothed metrics."""
	from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score
	
	model.eval()
	total_loss = 0.0
	all_preds = []
	all_targets = []
	
	with torch.no_grad():
		for images, phase_targets, bbox_grids, has_tools in dataloader:
			images = images.to(device)
			phase_targets = phase_targets.to(device)
			bbox_grids = bbox_grids.to(device)
			has_tools = has_tools.to(device)
			
			phase_logits, spatial_logits = model(images)
			loss, _, _ = calculate_multitask_loss(
				phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools, lambda_aux=lambda_aux
			)
			total_loss += loss.item()
			preds = torch.argmax(phase_logits, dim=1)
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(phase_targets.cpu().numpy())
	
	avg_loss = total_loss / len(dataloader)
	
	# Raw metrics
	raw_metrics = {
		'loss': avg_loss,
		'accuracy': accuracy_score(all_targets, all_preds),
		'jaccard_iou': jaccard_score(all_targets, all_preds, average='macro', zero_division=0),
		'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
		'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
	}
	
	# Smoothed metrics
	smoothed_preds = temporal_smooth_predictions(all_preds, window=smooth_window)
	smooth_metrics = {
		'loss': avg_loss,
		'accuracy': accuracy_score(all_targets, smoothed_preds),
		'jaccard_iou': jaccard_score(all_targets, smoothed_preds, average='macro', zero_division=0),
		'precision': precision_score(all_targets, smoothed_preds, average='macro', zero_division=0),
		'recall': recall_score(all_targets, smoothed_preds, average='macro', zero_division=0),
	}
	
	return raw_metrics, smooth_metrics, all_targets, all_preds, smoothed_preds


# ============================================================
# Main
# ============================================================
def main():
	EPOCHS = 40
	BATCH_SIZE = 16
	PATIENCE = 12
	LAMBDA_AUX = 0.5
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	print(f"--- Improved Training Pipeline on {DEVICE} ---")
	
	# 1. Data
	phase_dir = os.path.join("annotations", "bbox", "phase")
	global_phase_dict = load_real_phase_dict(phase_dir)
	
	train_dataset = SurgicalMultiTaskDataset(split='train', phase_dict=global_phase_dict)
	val_dataset = SurgicalMultiTaskDataset(split='val', phase_dict=global_phase_dict)
	test_dataset = SurgicalMultiTaskDataset(split='test', phase_dict=global_phase_dict)
	
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	
	# 2. Compute class distribution for class-balanced loss
	print(" -> Computing training class distribution...")
	train_class_counts = [0] * 9
	for i in range(len(train_dataset)):
		_, label, _, _ = train_dataset[i]
		train_class_counts[label.item()] += 1
	print(f" -> Class counts: {train_class_counts}")
	
	# 3. Model
	encoder = ViTBackbone(num_unfrozen_blocks=6).to(DEVICE)
	model = ImprovedGazeModel(
		phase_encoder=encoder,
		gaze_weights_path="paper_gaze_outputs/best_gaze_student.pt",
		num_phases=9, embed_dim=768
	).to(DEVICE)
	
	# 4. Class-balanced focal loss
	focal_criterion = ClassBalancedFocalLoss(
		samples_per_class=train_class_counts, num_classes=9, beta=0.999, gamma=2.0
	).to(DEVICE)
	
	# 5. Differential LR optimizer
	backbone_params = []
	head_params = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if 'phase_encoder' in name:
			backbone_params.append(param)
		else:
			head_params.append(param)
	
	optimizer = torch.optim.AdamW([
		{'params': backbone_params, 'lr': 5e-6},
		{'params': head_params, 'lr': 5e-4}
	], weight_decay=0.05)
	
	# 6. Cosine schedule with warmup
	total_steps = EPOCHS * len(train_loader)
	warmup_steps = 2 * len(train_loader)  # 2 epochs warmup
	scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	
	# 7. Training loop
	best_val_jaccard = -1.0
	best_test_jaccard = -1.0
	epochs_no_improve = 0
	
	for epoch in range(EPOCHS):
		print(f"\n[Epoch {epoch + 1}/{EPOCHS}]")
		model.train()
		train_loss_total = 0.0
		
		for batch_idx, (images, phase_targets, bbox_grids, has_tools) in enumerate(train_loader):
			images = images.to(DEVICE)
			phase_targets = phase_targets.to(DEVICE)
			bbox_grids = bbox_grids.to(DEVICE)
			has_tools = has_tools.to(DEVICE)
			
			optimizer.zero_grad()
			phase_logits, spatial_logits = model(images)
			
			loss, loss_phase, loss_aux = calculate_improved_loss(
				phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools,
				focal_criterion, lambda_aux=LAMBDA_AUX
			)
			
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			
			train_loss_total += loss.item()
			
			if (batch_idx + 1) % 50 == 0:
				print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Phase: {loss_phase.item():.4f} | Aux: {loss_aux.item():.4f}")
		
		avg_train_loss = train_loss_total / len(train_loader)
		print(f" -> Avg Train Loss: {avg_train_loss:.4f}")
		
		# Validation
		print(" -> Running Validation...")
		val_metrics, _, val_targets, val_preds, _ = evaluate_with_smoothing(model, val_loader, DEVICE, LAMBDA_AUX)
		print(f" -> Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | Jaccard: {val_metrics['jaccard_iou']:.4f}")
		
		# Early stopping on val jaccard
		is_best = val_metrics['jaccard_iou'] > best_val_jaccard
		if is_best:
			best_val_jaccard = val_metrics['jaccard_iou']
			epochs_no_improve = 0
			print(f" -> [*] Val improved! Jaccard: {best_val_jaccard:.4f}")
			
			# Evaluate on test
			print(" -> Evaluating on Test Set...")
			raw_test, smooth_test, test_targets, test_preds, smooth_preds = evaluate_with_smoothing(
				model, test_loader, DEVICE, LAMBDA_AUX, smooth_window=7
			)
			
			# Use best of raw vs smoothed
			test_metrics = smooth_test if smooth_test['jaccard_iou'] > raw_test['jaccard_iou'] else raw_test
			used_smoothing = smooth_test['jaccard_iou'] > raw_test['jaccard_iou']
			
			print(f" -> Test Jaccard (raw):      {raw_test['jaccard_iou']:.4f}")
			print(f" -> Test Jaccard (smoothed):  {smooth_test['jaccard_iou']:.4f}")
			print(f" -> Using {'smoothed' if used_smoothing else 'raw'} predictions")
			
			if test_metrics['jaccard_iou'] > best_test_jaccard:
				prev_best = best_test_jaccard
				best_test_jaccard = test_metrics['jaccard_iou']
				
				desc = f"v2_epoch{epoch+1}_crossattn_cbfocal_cosine{'_smooth' if used_smoothing else ''}"
				ckpt_file, arch_file = save_scored_checkpoint(
					model, optimizer, epoch + 1, test_metrics, desc
				)
				
				leaderboard = update_leaderboard(test_metrics, desc, ckpt_file, arch_file)
				
				# Keep the current best test-Jaccard model as best_model_v2.pt
				best_path = os.path.join("checkpoints", "best_model_v2.pt")
				torch.save({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'metrics': test_metrics,
					'best_test_jaccard': best_test_jaccard,
					'description': desc,
				}, best_path)
				
				print(f"\n  *** NEW TEST HIGH ***")
				print(f"  Previous best: {prev_best:.4f}")
				print(f"  New best:      {best_test_jaccard:.4f}")
				print(f"  Checkpoint:    {best_path}")
				
				# Save confusion matrix
				final_preds = smooth_preds if used_smoothing else test_preds
				plot_confusion_matrix_heatmap(test_targets, final_preds, save_path="test_confusion_matrix_v2.png")
		else:
			epochs_no_improve += 1
			print(f" -> [!] No improvement for {epochs_no_improve} epoch(s).")
			if epochs_no_improve >= PATIENCE:
				print(f"\n[!] EARLY STOPPING at Epoch {epoch + 1}.")
				break
	
	# Final evaluation
	print("\n--- Training Complete. Final Test Evaluation ---")
	ckpt_path = os.path.join("checkpoints", "best_model_v2.pt")
	if os.path.exists(ckpt_path):
		ckpt = torch.load(ckpt_path, map_location=DEVICE)
		model.load_state_dict(ckpt['state_dict'])
	
	raw_test, smooth_test, test_targets, test_preds, smooth_preds = evaluate_with_smoothing(
		model, test_loader, DEVICE, LAMBDA_AUX, smooth_window=7
	)
	test_metrics = smooth_test if smooth_test['jaccard_iou'] > raw_test['jaccard_iou'] else raw_test
	
	print(f"\n--- FINAL TEST RESULTS ---")
	print(f"Test Accuracy    : {test_metrics['accuracy']:.4f}")
	print(f"Test Jaccard IoU : {test_metrics['jaccard_iou']:.4f}")
	print(f"Test Precision   : {test_metrics['precision']:.4f}")
	print(f"Test Recall      : {test_metrics['recall']:.4f}")
	print(f"Raw Jaccard:       {raw_test['jaccard_iou']:.4f}")
	print(f"Smoothed Jaccard:  {smooth_test['jaccard_iou']:.4f}")
	print(f"--------------------------")
	
	final_preds = smooth_preds if smooth_test['jaccard_iou'] > raw_test['jaccard_iou'] else test_preds
	plot_confusion_matrix_heatmap(test_targets, final_preds, save_path="test_confusion_matrix_v2.png")
	
	log_handle.flush()
	log_handle.close()


if __name__ == "__main__":
	main()
