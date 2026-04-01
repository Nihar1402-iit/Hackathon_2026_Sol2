"""
Evaluation metrics for depth estimation.

Standard metrics: RMSE, AbsRel, Delta, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def compute_metrics(depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute depth estimation metrics.
    
    Args:
        depth_pred: Predicted depth (B, 1, H, W)
        depth_gt: Ground truth depth (B, 1, H, W)
        mask: Valid pixel mask (B, 1, H, W), optional
        
    Returns:
        Dictionary of metrics
    """
    # Detach and move to CPU if needed
    if depth_pred.is_cuda:
        depth_pred = depth_pred.detach().cpu()
    if depth_gt.is_cuda:
        depth_gt = depth_gt.detach().cpu()
    if mask is not None and mask.is_cuda:
        mask = mask.detach().cpu()
    
    # Flatten
    depth_pred = depth_pred.numpy().flatten()
    depth_gt = depth_gt.numpy().flatten()
    
    if mask is not None:
        mask = mask.numpy().flatten() > 0.5
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]
    
    # Remove invalid values
    valid = (depth_gt > 0) & (depth_pred > 0) & np.isfinite(depth_pred) & np.isfinite(depth_gt)
    depth_pred = depth_pred[valid]
    depth_gt = depth_gt[valid]
    
    if len(depth_pred) == 0:
        return {'rmse': 0.0, 'abs_rel': 0.0, 'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0}
    
    # Metrics
    metrics = {}
    
    # RMSE
    rmse = np.sqrt(np.mean((depth_pred - depth_gt) ** 2))
    metrics['rmse'] = float(rmse)
    
    # Absolute relative error
    abs_rel = np.mean(np.abs(depth_pred - depth_gt) / depth_gt)
    metrics['abs_rel'] = float(abs_rel)
    
    # Scale-invariant log RMSE
    log_pred = np.log(depth_pred + 1e-6)
    log_gt = np.log(depth_gt + 1e-6)
    silog = np.sqrt(np.mean((log_pred - log_gt) ** 2) - 
                    (np.mean(log_pred - log_gt) ** 2))
    metrics['silog'] = float(silog)
    
    # Delta accuracies
    ratio = depth_pred / depth_gt
    delta1 = np.mean((ratio > 1.25) | (1.0 / ratio > 1.25))
    delta2 = np.mean((ratio > 1.25 ** 2) | (1.0 / ratio > 1.25 ** 2))
    delta3 = np.mean((ratio > 1.25 ** 3) | (1.0 / ratio > 1.25 ** 3))
    
    metrics['delta1'] = float(delta1)
    metrics['delta2'] = float(delta2)
    metrics['delta3'] = float(delta3)
    
    return metrics


def compute_depth_accuracy(depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                          threshold: float = 1.25) -> float:
    """
    Compute accuracy@threshold metric.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        threshold: Threshold ratio (default 1.25)
        
    Returns:
        Accuracy (fraction of pixels within threshold)
    """
    ratio = depth_pred / (depth_gt + 1e-8)
    accurate = ((ratio > 1.0 / threshold) & (ratio < threshold)).float()
    return accurate.mean().item()


def compute_mae(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    """
    Compute mean absolute error.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        
    Returns:
        MAE value
    """
    return torch.abs(depth_pred - depth_gt).mean().item()


def compute_mse(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    """
    Compute mean squared error.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        
    Returns:
        MSE value
    """
    return ((depth_pred - depth_gt) ** 2).mean().item()


def compute_rmse(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    """
    Compute root mean squared error.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        
    Returns:
        RMSE value
    """
    return torch.sqrt(((depth_pred - depth_gt) ** 2).mean()).item()


def compute_abs_rel(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    """
    Compute absolute relative error.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        
    Returns:
        AbsRel value
    """
    return (torch.abs(depth_pred - depth_gt) / (depth_gt + 1e-8)).mean().item()


def compute_sq_rel(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    """
    Compute squared relative error.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        
    Returns:
        SqRel value
    """
    return (((depth_pred - depth_gt) ** 2) / (depth_gt + 1e-8)).mean().item()
