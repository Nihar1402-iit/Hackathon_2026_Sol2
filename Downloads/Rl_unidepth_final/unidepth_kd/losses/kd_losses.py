"""
Knowledge distillation loss functions.

Implements various KD loss variants:
- Feature distillation (MSE on normalized features)
- Attention distillation (KL divergence)
- Depth distillation (scale-aware)
- Relational distillation
- Positional encoding distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FeatureDistillationLoss(nn.Module):
    """
    Feature distillation loss with normalization.
    
    L_feat = Σ || F_s' / ||F_s'|| − F_t' / ||F_t'|| ||²
    
    Normalizes features for robust comparison across scales.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-6):
        """
        Args:
            reduction: 'mean' or 'sum'
            eps: Small value for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, features_s: torch.Tensor, features_t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features_s: Student features (B, C, H, W) or (B, N, D)
            features_t: Teacher features (same shape or different channel dim)
            mask: Optional mask for valid regions
            
        Returns:
            Feature distillation loss
        """
        # Align spatial and channel dimensions if needed
        if features_s.shape != features_t.shape:
            if features_s.dim() == 4:  # Spatial features (B, C, H, W)
                # Align spatial resolution
                h, w = features_t.shape[2:]
                features_s = F.interpolate(features_s, size=(h, w), mode='bilinear', 
                                          align_corners=False)
                # Align channels to common minimum
                min_channels = min(features_s.shape[1], features_t.shape[1])
                features_s = features_s[:, :min_channels, :, :]
                features_t = features_t[:, :min_channels, :, :]
            else:  # Token features (B, N, D)
                # Align sequence length
                min_n = min(features_s.shape[1], features_t.shape[1])
                features_s = features_s[:, :min_n, :]
                features_t = features_t[:, :min_n, :]
                # Align embedding dimension
                min_d = min(features_s.shape[2], features_t.shape[2])
                features_s = features_s[:, :, :min_d]
                features_t = features_t[:, :, :min_d]
        
        # Normalize features
        if features_s.dim() == 4:  # Spatial
            norm_s = torch.norm(features_s, dim=1, keepdim=True).clamp(min=self.eps)
            norm_t = torch.norm(features_t, dim=1, keepdim=True).clamp(min=self.eps)
        else:  # Tokens
            norm_s = torch.norm(features_s, dim=-1, keepdim=True).clamp(min=self.eps)
            norm_t = torch.norm(features_t, dim=-1, keepdim=True).clamp(min=self.eps)
        
        features_s_norm = features_s / norm_s
        features_t_norm = features_t / norm_t
        
        # MSE loss
        loss = (features_s_norm - features_t_norm) ** 2
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + self.eps)
            else:
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class AttentionDistillationLoss(nn.Module):
    """
    Attention distillation loss using KL divergence.
    
    L_attn = Σ KL(A_t || A_s)
    
    KL(P||Q) = Σ P log(P / Q)
    
    Average across all attention heads and layers.
    """
    
    def __init__(self, reduction: str = 'mean', temperature: float = 1.0,
                 eps: float = 1e-6):
        """
        Args:
            reduction: 'mean' or 'sum'
            temperature: Temperature for softening distributions
            eps: Small value for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, attn_s_list: List[torch.Tensor], 
                attn_t_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attn_s_list: List of student attention maps (B, H, N, N)
            attn_t_list: List of teacher attention maps (B, H, N, N)
            
        Returns:
            Attention distillation loss
        """
        total_loss = 0.0
        count = 0
        
        for attn_s, attn_t in zip(attn_s_list, attn_t_list):
            # Handle different sequence lengths
            n_s = attn_s.shape[-1]
            n_t = attn_t.shape[-1]
            
            if n_s != n_t:
                # Interpolate to match
                if n_s < n_t:
                    # Upsample student
                    attn_s = F.interpolate(
                        attn_s.reshape(-1, n_s, n_s).unsqueeze(1),
                        size=(n_t, n_t),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1).reshape(attn_s.shape[0], attn_s.shape[1], n_t, n_t)
                else:
                    # Downsample student
                    attn_s = F.adaptive_avg_pool2d(
                        attn_s.reshape(-1, 1, n_s, n_s),
                        output_size=(n_t, n_t)
                    ).reshape(attn_s.shape[0], attn_s.shape[1], n_t, n_t)
            
            # Temperature scaling and softmax
            attn_s = (attn_s / self.temperature).softmax(dim=-1)
            attn_t = (attn_t / self.temperature).softmax(dim=-1)
            
            # KL divergence
            log_attn_s = torch.log(attn_s + self.eps)
            kl = (attn_t * (torch.log(attn_t + self.eps) - log_attn_s)).sum(dim=-1).mean()
            
            total_loss += kl
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=attn_s.device, requires_grad=True)
        
        if self.reduction == 'mean':
            return total_loss / count
        else:
            return total_loss


class DepthDistillationLoss(nn.Module):
    """
    Depth map distillation loss.
    
    L_KD_depth = (1/n) Σ (log D_s − log D_t)²
    
    Scale-aware loss that handles scale ambiguity in monocular depth.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-6):
        """
        Args:
            reduction: 'mean' or 'sum'
            eps: Small value for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, depth_s: torch.Tensor, depth_t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            depth_s: Student depth (B, 1, H, W)
            depth_t: Teacher depth (B, 1, H, W)
            mask: Optional valid mask
            
        Returns:
            Depth distillation loss
        """
        # Align spatial dimensions
        if depth_s.shape != depth_t.shape:
            h, w = depth_t.shape[2:]
            depth_s = F.interpolate(depth_s, size=(h, w), mode='bilinear', align_corners=False)
        
        # Log loss (scale-aware)
        log_diff = torch.log(torch.clamp(depth_s, min=self.eps)) - \
                   torch.log(torch.clamp(depth_t, min=self.eps))
        
        loss = log_diff ** 2
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + self.eps)
            else:
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class RelationalDistillationLoss(nn.Module):
    """
    Relational distillation loss.
    
    Matches relations between feature pairs:
    R_ij = ||F_i − F_j||
    
    L_rel = ||R_s − R_t||²
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, features_s: torch.Tensor, features_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_s: Student features (B, C, H, W) or (B, N, D)
            features_t: Teacher features (same shape)
            
        Returns:
            Relational distillation loss
        """
        # Align spatial dimensions if needed
        if features_s.shape != features_t.shape:
            if features_s.dim() == 4:
                h, w = features_t.shape[2:]
                features_s = F.interpolate(features_s, size=(h, w), mode='bilinear',
                                          align_corners=False)
            else:
                features_s = features_s[:, :features_t.shape[1], :]
        
        # Compute pairwise distances
        if features_s.dim() == 4:  # Spatial
            B, C, H, W = features_s.shape
            # Reshape to (B, HW, C)
            feat_s = features_s.reshape(B, C, -1).transpose(1, 2)
            feat_t = features_t.reshape(B, C, -1).transpose(1, 2)
        else:  # Tokens
            feat_s = features_s
            feat_t = features_t
        
        # Pairwise distances
        dist_s = torch.cdist(feat_s, feat_s, p=2)  # (B, N, N)
        dist_t = torch.cdist(feat_t, feat_t, p=2)  # (B, N, N)
        
        # MSE on relations
        loss = ((dist_s - dist_t) ** 2).mean()
        
        return loss


class PositionalEncodingDistillationLoss(nn.Module):
    """
    Positional encoding distillation loss.
    
    L_pos = ||P_s − P_t||²
    
    Distills positional information from teacher to student.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pos_s: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_s: Student positional embeddings (1, N, D)
            pos_t: Teacher positional embeddings (1, N, D)
            
        Returns:
            Positional encoding loss
        """
        # Align sequence length
        n_s = pos_s.shape[1]
        n_t = pos_t.shape[1]
        
        if n_s != n_t:
            if n_s < n_t:
                # Interpolate student positions
                pos_s = F.interpolate(pos_s.permute(0, 2, 1).unsqueeze(-1),
                                     size=(n_t, 1),
                                     mode='bilinear',
                                     align_corners=False).squeeze(-1).permute(0, 2, 1)
            else:
                # Interpolate teacher positions
                pos_t = F.interpolate(pos_t.permute(0, 2, 1).unsqueeze(-1),
                                     size=(n_s, 1),
                                     mode='bilinear',
                                     align_corners=False).squeeze(-1).permute(0, 2, 1)
        
        # MSE loss
        loss = ((pos_s - pos_t) ** 2).mean()
        
        return loss


def create_kd_loss(loss_type: str = 'feature', **kwargs) -> nn.Module:
    """
    Create KD loss function.
    
    Args:
        loss_type: 'feature', 'attention', 'depth', 'relational', 'positional'
        **kwargs: Additional arguments
    
    Returns:
        Loss module
    """
    losses = {
        'feature': FeatureDistillationLoss,
        'attention': AttentionDistillationLoss,
        'depth': DepthDistillationLoss,
        'relational': RelationalDistillationLoss,
        'positional': PositionalEncodingDistillationLoss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown KD loss type: {loss_type}")
    
    return losses[loss_type](**kwargs)
