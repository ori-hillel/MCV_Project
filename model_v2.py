import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from gaze_student import ViTGazeStudent, gaze_distribution


class ViTBackbone(nn.Module):
	def __init__(self, num_unfrozen_blocks=4):
		super().__init__()
		self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
		
		for param in self.vit.parameters():
			param.requires_grad = False
		
		total_blocks = len(self.vit.encoder.layers)
		start_unfreeze = total_blocks - num_unfrozen_blocks
		
		for i in range(start_unfreeze, total_blocks):
			for param in self.vit.encoder.layers[i].parameters():
				param.requires_grad = True
		
		for param in self.vit.encoder.ln.parameters():
			param.requires_grad = True
	
	def forward(self, x):
		x = self.vit._process_input(x)
		n = x.shape[0]
		batch_class_token = self.vit.class_token.expand(n, -1, -1)
		x = torch.cat([batch_class_token, x], dim=1)
		x = x + self.vit.encoder.pos_embedding
		x = self.vit.encoder.dropout(x)
		x = self.vit.encoder.layers(x)
		x = self.vit.encoder.ln(x)
		return x


class GazeCrossAttentionFusion(nn.Module):
	"""Cross-attention fusion: gaze distribution queries patch tokens."""
	def __init__(self, embed_dim=768, num_heads=4, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.scale = self.head_dim ** -0.5
		
		# Gaze-conditioned query projection
		self.gaze_proj = nn.Sequential(
			nn.Linear(1, embed_dim),
			nn.GELU(),
		)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.dropout = nn.Dropout(dropout)
		
		# Learned gating parameter
		self.gate = nn.Parameter(torch.tensor(0.1))
	
	def forward(self, patch_tokens, gaze_probs):
		"""
		patch_tokens: [B, 196, 768]
		gaze_probs: [B, 196] spatial gaze distribution
		"""
		B, N, D = patch_tokens.shape
		
		# Create gaze-conditioned queries by adding gaze embedding to patch tokens
		gaze_embed = self.gaze_proj(gaze_probs.unsqueeze(-1))  # [B, 196, 768]
		query_input = self.norm1(patch_tokens + gaze_embed)
		
		Q = self.q_proj(query_input)  # [B, 196, 768]
		K = self.k_proj(self.norm2(patch_tokens))  # [B, 196, 768]
		V = self.v_proj(patch_tokens)  # [B, 196, 768]
		
		# Multi-head attention
		Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		
		attn = (Q @ K.transpose(-2, -1)) * self.scale
		
		# Bias attention with gaze distribution
		gaze_bias = gaze_probs.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 196]
		attn = attn + gaze_bias * 5.0  # Scale gaze bias
		
		attn = F.softmax(attn, dim=-1)
		attn = self.dropout(attn)
		
		out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
		out = self.out_proj(out)
		
		# Gated residual
		fused = patch_tokens + torch.sigmoid(self.gate) * out
		return fused


class DualStreamPooling(nn.Module):
	"""Global + focal (gaze-weighted) pooling for classification."""
	def __init__(self, embed_dim=768):
		super().__init__()
		self.global_proj = nn.Linear(embed_dim, embed_dim)
		self.focal_proj = nn.Linear(embed_dim, embed_dim)
		self.merge = nn.Sequential(
			nn.Linear(embed_dim * 2, embed_dim),
			nn.GELU(),
			nn.Dropout(0.2),
		)
	
	def forward(self, patch_tokens, gaze_probs, cls_token):
		"""
		patch_tokens: [B, 196, 768]
		gaze_probs: [B, 196]
		cls_token: [B, 768]
		"""
		# Global: mean pool
		global_feat = patch_tokens.mean(dim=1)  # [B, 768]
		global_feat = self.global_proj(global_feat)
		
		# Focal: gaze-weighted pool
		weights = gaze_probs.unsqueeze(-1)  # [B, 196, 1]
		focal_feat = (patch_tokens * weights).sum(dim=1)  # [B, 768]
		focal_feat = self.focal_proj(focal_feat)
		
		# Merge
		merged = self.merge(torch.cat([global_feat, focal_feat], dim=-1))
		return cls_token + merged


class ImprovedGazeModel(nn.Module):
	def __init__(self, phase_encoder, gaze_weights_path, num_phases=9, embed_dim=768, grid_size=14):
		super().__init__()
		self.phase_encoder = phase_encoder
		
		# Frozen gaze encoder
		self.gaze_encoder = ViTGazeStudent(dropout=0.0)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		checkpoint = torch.load(gaze_weights_path, map_location=device)
		if 'state_dict' in checkpoint:
			self.gaze_encoder.load_state_dict(checkpoint['state_dict'])
		elif 'model_state_dict' in checkpoint:
			self.gaze_encoder.load_state_dict(checkpoint['model_state_dict'])
		else:
			self.gaze_encoder.load_state_dict(checkpoint)
		for param in self.gaze_encoder.parameters():
			param.requires_grad = False
		
		self.grid_h = grid_size
		self.grid_w = grid_size
		
		# Cross-attention gaze fusion
		self.gaze_fusion = GazeCrossAttentionFusion(embed_dim=embed_dim, num_heads=4, dropout=0.1)
		
		# Dual-stream pooling
		self.dual_pool = DualStreamPooling(embed_dim=embed_dim)
		
		# Spatial head
		self.spatial_head = nn.Sequential(
			nn.Linear(embed_dim, 256),
			nn.GELU(),
			nn.Linear(256, 1)
		)
		
		# Phase head with stronger capacity
		self.phase_head = nn.Sequential(
			nn.LayerNorm(embed_dim),
			nn.Dropout(p=0.3),
			nn.Linear(embed_dim, 512),
			nn.GELU(),
			nn.Dropout(p=0.2),
			nn.Linear(512, num_phases)
		)
	
	def forward(self, x):
		# A. Generate gaze prior
		with torch.no_grad():
			raw_gaze_logits = self.gaze_encoder(x)
			gaze_probs = gaze_distribution(raw_gaze_logits).squeeze(1)  # [B, 14, 14]
		
		# B. Extract features
		features = self.phase_encoder(x)  # [B, 197, 768]
		cls_token = features[:, 0, :]  # [B, 768]
		patch_tokens = features[:, 1:, :]  # [B, 196, 768]
		
		gaze_flat = gaze_probs.view(x.shape[0], 196)  # [B, 196]
		
		# C. Cross-attention gaze fusion
		fused_patches = self.gaze_fusion(patch_tokens, gaze_flat)
		
		# D. Spatial output
		spatial_logits = self.spatial_head(fused_patches).squeeze(-1)
		spatial_logits = spatial_logits.view(-1, self.grid_h, self.grid_w)
		
		# E. Phase output with dual-stream pooling
		pooled = self.dual_pool(fused_patches, gaze_flat, cls_token)
		phase_logits = self.phase_head(pooled)
		
		return phase_logits, spatial_logits
