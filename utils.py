import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# FIX: Added accuracy, precision, and recall scores for the evaluator
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score, accuracy_score, precision_score, \
	recall_score
import glob
import csv

# FIX: Import the loss calculator so the evaluator can compute validation loss
from loss import calculate_multitask_loss


def evaluate_model(model, dataloader, device, lambda_aux=0.5):
	"""
	Evaluates the multi-task model and calculates strict macro metrics.
	"""
	model.eval()
	total_loss = 0.0
	all_preds = []
	all_targets = []
 	
	# Disable gradient tracking for evaluation to save VRAM and compute
	with torch.no_grad():
		for batch_idx, (images, phase_targets, bbox_grids, has_tools) in enumerate(dataloader):
			images = images.to(device)
			phase_targets = phase_targets.to(device)
			bbox_grids = bbox_grids.to(device)
			has_tools = has_tools.to(device)
 			
			# Forward pass
			phase_logits, spatial_logits = model(images)
 			
			# Calculate loss using the exact same function as training
			loss, _, _ = calculate_multitask_loss(
				phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools, lambda_aux=lambda_aux
			)
			total_loss += loss.item()
 			
			# Extract the predicted phase (highest logit)
			preds = torch.argmax(phase_logits, dim=1)
 			
			# Store targets and predictions for sklearn
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(phase_targets.cpu().numpy())
			print(f"Batch {batch_idx+1}/{len(dataloader)} done; current accuracy proxy={accuracy_score(all_targets, all_preds):.4f}")
 	
	avg_loss = total_loss / len(dataloader)
 	
	# Calculate mathematically strict metrics
	metrics = {
		'loss': avg_loss,
		'accuracy': accuracy_score(all_targets, all_preds),
		'jaccard_iou': jaccard_score(all_targets, all_preds, average='macro', zero_division=0),
		'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
		'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0)
	}
 	
	return metrics, all_targets, all_preds


def compute_metrics(y_true, y_pred, num_classes=9):
	"""
	Computes Frame Accuracy, Jaccard Index (IoU), Precision, and Recall across phases.
	"""
	acc = (np.array(y_true) == np.array(y_pred)).mean()
	jaccard = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
	
	report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
	macro_precision = report['macro avg']['precision']
	macro_recall = report['macro avg']['recall']
	
	return {
		'accuracy': acc,
		'jaccard_iou': jaccard,
		'precision': macro_precision,
		'recall': macro_recall
	}


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
	torch.save(state, checkpoint_path)
	if is_best:
		best_path = os.path.join(checkpoint_dir, "best_model.pt")
		torch.save(state, best_path)
		print(f" Saved new best model checkpoint to {best_path}")


def plot_training_curves(train_losses, val_losses, metrics_history, save_path="training_curves.png"):
	epochs = range(1, len(train_losses) + 1)
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	# Loss Plot
	ax1.plot(epochs, [x['total'] for x in train_losses], label='Train Total Loss', color='blue')
	ax1.plot(epochs, [x['total'] for x in val_losses], label='Val Total Loss', color='orange', linestyle='--')
	ax1.set_title("Training and Validation Loss")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend()
	ax1.grid(True)
	
	# Metrics Plot
	ax2.plot(epochs, [m['jaccard_iou'] for m in metrics_history], label='Val Jaccard IoU', color='green')
	ax2.plot(epochs, [m['accuracy'] for m in metrics_history], label='Val Accuracy', color='purple')
	ax2.set_title("Validation Evaluation Metrics")
	ax2.set_xlabel("Epochs")
	ax2.set_ylabel("Score")
	ax2.legend()
	ax2.grid(True)
	
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()


def plot_confusion_matrix_heatmap(y_true, y_pred, phase_names=None, save_path="confusion_matrix.png"):
	labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Force the 9x9 grid
	cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
	
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
	            xticklabels=phase_names if phase_names else labels,
	            yticklabels=phase_names if phase_names else labels)
	plt.title("Normalized Surgical Phase Confusion Matrix")
	plt.xlabel("Predicted Phase")
	plt.ylabel("Ground Truth Phase")
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()


def visualize_spatial_prediction(image_tensor, target_grid, pred_grid, save_path="prediction_overlay.png"):
	"""
	Overlays ground truth vs predicted spatial token grid on top of input frame.
	"""
	img = image_tensor.permute(1, 2, 0).cpu().numpy()
	img = (img - img.min()) / (img.max() - img.min())  # Unnormalize for plot
	# add eps to denominator
	
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(img)
	axes[0].set_title("Input Frame")
	axes[0].axis('off')
	
	axes[1].imshow(img)
	axes[1].imshow(target_grid.cpu().numpy(), cmap='jet', alpha=0.5)
	axes[1].set_title("Ground Truth Grid")
	axes[1].axis('off')
	
	axes[2].imshow(img)
	axes[2].imshow(torch.sigmoid(pred_grid).detach().cpu().numpy()[0], cmap='jet', alpha=0.5)
	axes[2].set_title("Predicted Spatial Grid")
	axes[2].axis('off')
	
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()


def load_real_phase_dict(phase_dir):
	"""
	Scans the annotations/phase directory, maps string phase names to integer IDs,
	and builds a dictionary mapping the relative image path to the ground-truth integer phase.
	"""
	phase_dict = {}
	phase_name_to_id = {}  # This will store our string -> int mapping
	
	csv_files = glob.glob(os.path.join(phase_dir, '*.csv'))
	
	if not csv_files:
		raise FileNotFoundError(f"CRITICAL: No .csv files found in {phase_dir}")
	
	for csv_path in csv_files:
		filename = os.path.basename(csv_path)
		vid_folder = filename.split('_')[0]
		clip_prefix = filename.replace('.csv', '')
		
		with open(csv_path, 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			
			for line_idx, row in enumerate(reader):
				if not row or len(row) < 2:
					continue
				
				raw_frame_id = row[0].strip()
				# Lowercase the string to prevent "Closure" and "closure" from becoming two different classes
				raw_phase_str = row[1].strip().lower()
				
				# Skip headers safely
				if raw_frame_id.lower() in ['frame', 'frame_id', 'id', 'name']:
					continue
				
				# 1. Map the text string to a PyTorch-compatible integer
				if raw_phase_str not in phase_name_to_id:
					phase_name_to_id[raw_phase_str] = len(phase_name_to_id)
				
				phase_label = phase_name_to_id[raw_phase_str]
				
				# 2. Construct the dataloader key
				if raw_frame_id.isdigit():
					frame_file = f"{clip_prefix}_{int(raw_frame_id):04d}.jpg"
				else:
					frame_file = raw_frame_id if raw_frame_id.endswith('.jpg') else f"{raw_frame_id}.jpg"
				
				rel_path = f"{vid_folder}/{frame_file}"
				phase_dict[rel_path] = phase_label
	
	# Print the mapping legend so you know exactly which integer corresponds to which phase
	print("\n=== Surgical Phase Mapping Legend ===")
	for phase_name, phase_id in phase_name_to_id.items():
		print(f" Class {phase_id}: {phase_name.capitalize()}")
	print("=====================================\n")
	
	return phase_dict