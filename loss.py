import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaWeightedFocalLoss(nn.Module):
	def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
		super(AlphaWeightedFocalLoss, self).__init__()
		self.gamma = gamma
		self.reduction = reduction
		self.alpha = alpha
	
	def forward(self, inputs, targets):
		ce_loss = F.cross_entropy(inputs, targets, reduction='none')
		pt = torch.exp(-ce_loss)
		focal_loss = ((1 - pt) ** self.gamma) * ce_loss
		
		if self.alpha is not None:
			alpha_t = self.alpha.gather(0, targets.data.view(-1))
			focal_loss = alpha_t * focal_loss
		
		if self.reduction == 'mean':
			return focal_loss.mean()
		return focal_loss.sum()


# --- Custom Alpha Weights for EgoSurgery-Phase ---
# Punish the network for missing rare classes (weight=5.0)
# and give almost no reward for just guessing dominant classes like Dissection (weight=0.5)
weights = torch.tensor([5.0, 3.0, 4.0, 2.0, 0.5, 2.0, 0.5, 5.0, 5.0], dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = weights.to(device)

focal_criterion = AlphaWeightedFocalLoss(alpha=weights, gamma=2.0)


def calculate_multitask_loss(phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools, lambda_aux=0.5):
	loss_phase = focal_criterion(phase_logits, phase_targets)
	
	if has_tools.any():
		valid_spatial_logits = spatial_logits[has_tools]
		valid_bbox_grids = bbox_grids[has_tools].unsqueeze(1)
		# loss_aux = F.binary_cross_entropy_with_logits(valid_spatial_logits, valid_bbox_grids.float())
		
		# Flatten both tensors to 1D arrays to guarantee shape alignment
		loss_aux = F.binary_cross_entropy_with_logits(
			valid_spatial_logits.view(-1),
			valid_bbox_grids.view(-1).float()
		)
	else:
		loss_aux = torch.tensor(0.0, device=phase_logits.device, requires_grad=True)
	
	total_loss = loss_phase + (lambda_aux * loss_aux)
	return total_loss, loss_phase, loss_aux