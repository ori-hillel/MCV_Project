import torch
import torch.nn.functional as F
import numpy as np


class UnsupervisedSurpriseExtractor:
	"""
	Computes spatial-temporal surprise heatmaps using local configuration energy
	based on appearance cosine distance and motion dynamics.
	"""
	
	def __init__(self, grid_size=14, scale_weight=1.0, alpha=2.0):
		self.grid_size = grid_size
		self.scale_weight = scale_weight
		self.alpha = alpha
	
	def compute_surprise_map(self, current_frame, prev_frame=None):
		"""
		current_frame: PyTorch Tensor [3, H, W]
		prev_frame: PyTorch Tensor [3, H, W] or None
		returns: Normalized 2D spatial grid [grid_size, grid_size]
		"""
		# Pool frame into local token grid
		grid_features = F.adaptive_avg_pool2d(current_frame, (self.grid_size, self.grid_size))
		grid_features = grid_features.view(3, -1).permute(1, 0)  # [H*W, Channels]
		
		# Spatial local bond energy (Appearance Variance across neighbors)
		norm_feats = F.normalize(grid_features, p=2, dim=-1)
		similarity_matrix = torch.mm(norm_feats, norm_feats.t())
		spatial_surprise = 1.0 - similarity_matrix.mean(dim=-1).view(self.grid_size, self.grid_size)
		
		# Temporal local bond energy (if previous frame exists)
		if prev_frame is not None:
			prev_grid_features = F.adaptive_avg_pool2d(prev_frame, (self.grid_size, self.grid_size))
			prev_grid_features = prev_grid_features.view(3, -1).permute(1, 0)
			prev_norm_feats = F.normalize(prev_grid_features, p=2, dim=-1)
			
			temporal_diff = 1.0 - (norm_feats * prev_norm_feats).sum(dim=-1).view(self.grid_size, self.grid_size)
			total_surprise = torch.clamp(spatial_surprise + temporal_diff, max=self.alpha)
		else:
			total_surprise = spatial_surprise
		
		# Normalize energy map to [0, 1] range
		min_val, max_val = total_surprise.min(), total_surprise.max()
		if max_val > min_val:
			total_surprise = (total_surprise - min_val) / (max_val - min_val)
		else:
			total_surprise = torch.zeros_like(total_surprise)
		
		return total_surprise