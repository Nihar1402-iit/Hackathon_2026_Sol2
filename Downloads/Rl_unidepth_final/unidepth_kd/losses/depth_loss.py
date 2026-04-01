"""
Depth estimation loss functions.

Implements multiple depth loss variants:
- L1/L2 loss
- Scale-invariant log loss
- SSIM loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1DepthLoss(nn.Module):
    """Simple L1 depth loss."""
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            depth_pred: Predicted depth (B, 1, H, W)
            depth_gt: Ground truth depth (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W), optional
            
        Returns:
            L1 loss
        """
        loss = torch.abs(depth_pred - depth_gt)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class L2DepthLoss(nn.Module):
    """L2 (MSE) depth loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            depth_pred: Predicted depth (B, 1, H, W)
            depth_gt: Ground truth depth (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W), optional
            
        Returns:
            L2 loss
        """
        loss = (depth_pred - depth_gt) ** 2
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-invariant log loss.
    
    Invariant to unknown depth scaling factor, useful for monocular depth.
    
    L_silog = (1/n) Σ (log d - log d_hat)²
            - (1/n²)(Σ (log d - log d_hat))²
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            depth_pred: Predicted depth (B, 1, H, W)
            depth_gt: Ground truth depth (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W), optional
            
        Returns:
            Scale-invariant log loss
        """
        # Clamp to prevent log of zero
        eps = 1e-6
        depth_pred = torch.clamp(depth_pred, min=eps)
        depth_gt = torch.clamp(depth_gt, min=eps)
        
        # Log differences
        log_diff = torch.log(depth_pred) - torch.log(depth_gt)
        
        if mask is not None:
            log_diff = log_diff * mask
            n = mask.sum() + 1e-8
        else:
            n = depth_pred.numel()
        
        # Scale-invariant log loss formula
        term1 = (log_diff ** 2).sum() / n
        term2 = (log_diff.sum() ** 2) / (n ** 2)
        
        loss = term1 - term2
        
        return loss


class GradientLoss(nn.Module):
    """
    Gradient loss for depth smoothness.
    
    Encourages smooth predictions while preserving sharp edges.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            depth_pred: Predicted depth (B, 1, H, W)
            depth_gt: Ground truth depth (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W), optional
            
        Returns:
            Gradient matching loss
        """
        # Compute gradients
        grad_pred_x = torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:])
        grad_pred_y = torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :])
        
        grad_gt_x = torch.abs(depth_gt[:, :, :, :-1] - depth_gt[:, :, :, 1:])
        grad_gt_y = torch.abs(depth_gt[:, :, :-1, :] - depth_gt[:, :, 1:, :])
        
        # Loss
        loss_x = torch.abs(grad_pred_x - grad_gt_x)
        loss_y = torch.abs(grad_pred_y - grad_gt_y)
        
        loss = loss_x.mean() + loss_y.mean()
        
        return loss


class SSIMDepthLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) based loss for depth.
    
    More perceptually aligned than pixel-wise losses.
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_kernel()
        self.register_buffer('kernel', kernel)
    
    def _create_kernel(self) -> torch.Tensor:
        """Create Gaussian kernel for SSIM computation."""
        x = torch.arange(self.window_size) - self.window_size // 2
        x = torch.exp(-x.pow(2) / (2 * self.sigma ** 2))
        kernel = x.unsqueeze(1) @ x.unsqueeze(0)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_pred: Predicted depth (B, 1, H, W)
            depth_gt: Ground truth depth (B, 1, H, W)
            
        Returns:
            SSIM loss (0 = perfect, 1 = completely different)
        """
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Apply Gaussian blur
        pad = self.window_size // 2
        kernel = self.kernel.to(depth_pred.device).expand(depth_pred.size(1), -1, -1, -1)
        
        mean_pred = F.conv2d(depth_pred, kernel, padding=pad, groups=1)
        mean_gt = F.conv2d(depth_gt, kernel, padding=pad, groups=1)
        
        mean_pred_sq = mean_pred ** 2
        mean_gt_sq = mean_gt ** 2
        mean_cross = mean_pred * mean_gt
        
        var_pred = F.conv2d(depth_pred ** 2, kernel, padding=pad, groups=1) - mean_pred_sq
        var_gt = F.conv2d(depth_gt ** 2, kernel, padding=pad, groups=1) - mean_gt_sq
        cov = F.conv2d(depth_pred * depth_gt, kernel, padding=pad, groups=1) - mean_cross
        
        # SSIM formula
        numerator1 = 2 * mean_cross + C1
        numerator2 = 2 * cov + C2
        denominator1 = mean_pred_sq + mean_gt_sq + C1
        denominator2 = var_pred + var_gt + C2
        
        ssim = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        return 1 - ssim.mean()


def create_depth_loss(loss_type: str = 'l1', **kwargs) -> nn.Module:
    """
    Create depth loss function.
    
    Args:
        loss_type: 'l1', 'l2', 'silog', 'gradient', 'ssim'
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss module
    """
    losses = {
        'l1': L1DepthLoss,
        'l2': L2DepthLoss,
        'silog': ScaleInvariantLogLoss,
        'gradient': GradientLoss,
        'ssim': SSIMDepthLoss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return losses[loss_type](**kwargs)
