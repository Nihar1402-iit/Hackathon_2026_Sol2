"""
Visualization utilities for depth and attention maps.

Includes depth map visualization, attention map visualization,
and other debugging utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple
from pathlib import Path


def visualize_depth(depth: torch.Tensor, cmap: str = 'turbo',
                   min_val: Optional[float] = None,
                   max_val: Optional[float] = None) -> np.ndarray:
    """
    Visualize depth map as RGB image.
    
    Args:
        depth: Depth map (1, H, W) or (H, W)
        cmap: Colormap name
        min_val: Minimum depth value for normalization
        max_val: Maximum depth value for normalization
        
    Returns:
        RGB image (H, W, 3) in [0, 1]
    """
    # Convert to numpy
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    
    # Squeeze channel dimension
    if depth.ndim == 3:
        depth = depth[0]
    
    # Remove invalid values
    valid = np.isfinite(depth) & (depth > 0)
    
    if valid.sum() == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3))
    
    # Normalize
    if min_val is None:
        min_val = np.percentile(depth[valid], 2)
    if max_val is None:
        max_val = np.percentile(depth[valid], 98)
    
    depth_normalized = np.clip((depth - min_val) / (max_val - min_val + 1e-6), 0, 1)
    
    # Apply colormap
    colormap = cm.get_cmap(cmap)
    rgb = colormap(depth_normalized)[:, :, :3]  # Drop alpha channel
    
    return rgb


def visualize_attention(attn_map: torch.Tensor, img_size: int = 384,
                       patch_size: int = 16) -> np.ndarray:
    """
    Visualize attention map.
    
    Args:
        attn_map: Attention weights (B, H, N, N) or (H, N, N)
        img_size: Original image size
        patch_size: Patch size
        
    Returns:
        Visualization (H, W, 3)
    """
    # Convert to numpy
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if attn_map.ndim == 4:
        attn_map = attn_map[0]  # Take first sample
    
    # Average across heads
    if attn_map.ndim == 3:
        attn_map = attn_map.mean(axis=0)  # (N, N)
    
    # Get CLS token attention (first row, excluding self-attention)
    cls_attn = attn_map[0, 1:]  # Attention to patches only
    
    # Reshape to spatial
    patch_h = patch_w = img_size // patch_size
    attn_spatial = cls_attn.reshape(patch_h, patch_w)
    
    # Upsample to image size
    attn_spatial = np.repeat(np.repeat(attn_spatial, patch_size, axis=0),
                            patch_size, axis=1)
    
    # Normalize
    attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-6)
    
    # Apply colormap
    colormap = cm.get_cmap('jet')
    rgb = colormap(attn_spatial)[:, :, :3]
    
    return rgb


def compare_depth_maps(depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                      depth_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create side-by-side comparison of predicted and GT depth.
    
    Args:
        depth_pred: Predicted depth (1, H, W)
        depth_gt: Ground truth depth (1, H, W)
        depth_range: (min, max) for normalization
        
    Returns:
        Comparison image (H, 2*W, 3)
    """
    pred_rgb = visualize_depth(depth_pred, min_val=depth_range[0] if depth_range else None,
                              max_val=depth_range[1] if depth_range else None)
    gt_rgb = visualize_depth(depth_gt, min_val=depth_range[0] if depth_range else None,
                            max_val=depth_range[1] if depth_range else None)
    
    # Concatenate horizontally
    comparison = np.concatenate([pred_rgb, gt_rgb], axis=1)
    
    return comparison


def visualize_error(depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                   error_type: str = 'abs') -> np.ndarray:
    """
    Visualize prediction error.
    
    Args:
        depth_pred: Predicted depth (1, H, W)
        depth_gt: Ground truth depth (1, H, W)
        error_type: 'abs' (absolute) or 'rel' (relative)
        
    Returns:
        Error map (H, W, 3)
    """
    # Convert to numpy
    if isinstance(depth_pred, torch.Tensor):
        depth_pred = depth_pred.detach().cpu().numpy()
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.detach().cpu().numpy()
    
    # Squeeze
    if depth_pred.ndim == 3:
        depth_pred = depth_pred[0]
    if depth_gt.ndim == 3:
        depth_gt = depth_gt[0]
    
    # Compute error
    if error_type == 'abs':
        error = np.abs(depth_pred - depth_gt)
    else:  # relative
        error = np.abs(depth_pred - depth_gt) / (np.abs(depth_gt) + 1e-6)
    
    # Normalize
    error = np.clip(error / (np.percentile(error, 95) + 1e-6), 0, 1)
    
    # Apply colormap
    colormap = cm.get_cmap('hot')
    rgb = colormap(error)[:, :, :3]
    
    return rgb


def save_depth_visualization(depth: torch.Tensor, save_path: str,
                            min_val: Optional[float] = None,
                            max_val: Optional[float] = None):
    """
    Save depth map visualization to file.
    
    Args:
        depth: Depth tensor
        save_path: Path to save
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
    """
    rgb = visualize_depth(depth, min_val=min_val, max_val=max_val)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def save_comparison(depth_pred: torch.Tensor, depth_gt: torch.Tensor,
                   save_path: str):
    """
    Save comparison visualization.
    
    Args:
        depth_pred: Predicted depth
        depth_gt: Ground truth depth
        save_path: Path to save
    """
    comparison = compare_depth_maps(depth_pred, depth_gt)
    
    plt.figure(figsize=(16, 6))
    plt.imshow(comparison)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def create_grid_visualization(images: list, titles: list = None,
                             save_path: Optional[str] = None) -> np.ndarray:
    """
    Create grid visualization of multiple images.
    
    Args:
        images: List of image arrays
        titles: List of titles for each image
        save_path: Optional path to save
        
    Returns:
        Grid image array
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused axes
    for ax in axes[n_images:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return image_array
