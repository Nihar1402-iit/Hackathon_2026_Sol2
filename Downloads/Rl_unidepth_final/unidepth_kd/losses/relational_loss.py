"""
Relational loss for multi-scale feature distillation.

Implements losses that match higher-order relationships between features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ChannelRelationLoss(nn.Module):
    """
    Loss matching channel-wise correlations between features.
    
    Encourages student to learn similar channel relationships as teacher.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, features_s: torch.Tensor, features_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_s: Student features (B, C, H, W)
            features_t: Teacher features (B, C, H, W)
            
        Returns:
            Channel relation loss
        """
        # Reshape to (B, C, -1)
        B, C, H, W = features_s.shape
        feat_s = features_s.reshape(B, C, -1)
        feat_t = features_t.reshape(B, C, -1)
        
        # Compute channel correlations
        # Normalize across spatial dimension
        feat_s = F.normalize(feat_s, p=2, dim=2)
        feat_t = F.normalize(feat_t, p=2, dim=2)
        
        # Channel-wise covariance
        corr_s = torch.bmm(feat_s, feat_s.transpose(1, 2))  # (B, C, C)
        corr_t = torch.bmm(feat_t, feat_t.transpose(1, 2))  # (B, C, C)
        
        # MSE on correlations
        loss = ((corr_s - corr_t) ** 2).mean()
        
        return loss


class MultiScaleRelationalLoss(nn.Module):
    """
    Loss matching relationships across multiple scales.
    
    Matches relations between features at different scales.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, features_s_list: List[torch.Tensor],
                features_t_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_s_list: List of student features at different scales
            features_t_list: List of teacher features at different scales
            
        Returns:
            Multi-scale relational loss
        """
        total_loss = 0.0
        count = 0
        
        for i in range(len(features_s_list)):
            for j in range(i + 1, len(features_s_list)):
                feat_s_i = features_s_list[i]
                feat_s_j = features_s_list[j]
                feat_t_i = features_t_list[i]
                feat_t_j = features_t_list[j]
                
                # Align spatial dimensions
                h_min = min(feat_s_i.shape[2], feat_t_i.shape[2])
                w_min = min(feat_s_i.shape[3], feat_t_i.shape[3])
                
                feat_s_i = F.adaptive_avg_pool2d(feat_s_i, (h_min, w_min))
                feat_s_j = F.adaptive_avg_pool2d(feat_s_j, (h_min, w_min))
                feat_t_i = F.adaptive_avg_pool2d(feat_t_i, (h_min, w_min))
                feat_t_j = F.adaptive_avg_pool2d(feat_t_j, (h_min, w_min))
                
                # Compute cross-scale relations
                B, C_i, H, W = feat_s_i.shape
                C_j = feat_s_j.shape[1]
                
                # Reshape and normalize
                f_s_i = feat_s_i.reshape(B, C_i, -1)
                f_s_j = feat_s_j.reshape(B, C_j, -1)
                f_t_i = feat_t_i.reshape(B, C_i, -1)
                f_t_j = feat_t_j.reshape(B, C_j, -1)
                
                # Compute relations
                rel_s = torch.bmm(F.normalize(f_s_i, p=2, dim=1),
                                 F.normalize(f_s_j, p=2, dim=1).transpose(1, 2))
                rel_t = torch.bmm(F.normalize(f_t_i, p=2, dim=1),
                                 F.normalize(f_t_j, p=2, dim=1).transpose(1, 2))
                
                # Loss
                loss = ((rel_s - rel_t) ** 2).mean()
                total_loss += loss
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=features_s_list[0].device, requires_grad=True)
        
        return total_loss / count


class DecoderFeatureDistillationLoss(nn.Module):
    """
    Distill decoder features.
    
    Matches features at intermediate decoder layers.
    """
    
    def __init__(self, reduction: str = 'mean', normalize: bool = True):
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize
    
    def forward(self, dec_feat_s: torch.Tensor, dec_feat_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dec_feat_s: Student decoder features (B, C, H, W)
            dec_feat_t: Teacher decoder features (B, C, H, W)
            
        Returns:
            Decoder feature distillation loss
        """
        # Align spatial dimensions
        if dec_feat_s.shape != dec_feat_t.shape:
            h, w = dec_feat_t.shape[2:]
            dec_feat_s = F.interpolate(dec_feat_s, size=(h, w), mode='bilinear',
                                      align_corners=False)
        
        if self.normalize:
            # Normalize
            norm_s = torch.norm(dec_feat_s, dim=1, keepdim=True).clamp(min=1e-6)
            norm_t = torch.norm(dec_feat_t, dim=1, keepdim=True).clamp(min=1e-6)
            dec_feat_s = dec_feat_s / norm_s
            dec_feat_t = dec_feat_t / norm_t
        
        loss = ((dec_feat_s - dec_feat_t) ** 2).mean()
        
        return loss
