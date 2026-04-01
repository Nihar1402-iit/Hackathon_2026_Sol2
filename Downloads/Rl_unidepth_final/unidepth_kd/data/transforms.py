"""
Data transforms for depth estimation.

Includes normalization, augmentation, and resizing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ImageNormalization:
    """Normalize RGB images using ImageNet statistics."""
    
    def __init__(self, use_imagenet_stats: bool = True):
        """
        Args:
            use_imagenet_stats: If True, use ImageNet mean/std
        """
        if use_imagenet_stats:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        else:
            self.mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
            self.std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Image tensor (3, H, W) in [0, 1]
            
        Returns:
            Normalized image
        """
        return (image - self.mean) / (self.std + 1e-6)


class DepthScaling:
    """Scale and normalize depth maps."""
    
    def __init__(self, method: str = 'min_max', scale_factor: float = 1.0):
        """
        Args:
            method: 'min_max', 'percentile', 'log'
            scale_factor: Additional scaling factor
        """
        self.method = method
        self.scale_factor = scale_factor
    
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: Depth tensor (1, H, W)
            
        Returns:
            Scaled depth
        """
        if self.method == 'min_max':
            d_min = depth.min()
            d_max = depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
        
        elif self.method == 'percentile':
            p2, p98 = torch.quantile(depth, torch.tensor([0.02, 0.98]))
            depth = torch.clamp(depth, min=p2, max=p98)
            if p98 - p2 > 1e-6:
                depth = (depth - p2) / (p98 - p2)
        
        elif self.method == 'log':
            depth = torch.log(depth + 1.0)
            d_max = depth.max()
            if d_max > 1e-6:
                depth = depth / d_max
        
        return depth * self.scale_factor


class RandomCrop:
    """Random crop with preservation of aspect ratio."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target (H, W)
        """
        self.size = size
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (3, H, W)
            depth: (1, H, W)
            mask: (1, H, W)
            
        Returns:
            Cropped (image, depth, mask)
        """
        H, W = image.shape[1], image.shape[2]
        crop_h, crop_w = self.size
        
        if H < crop_h or W < crop_w:
            return image, depth, mask
        
        # Random crop position
        top = torch.randint(0, H - crop_h + 1, (1,)).item() if H > crop_h else 0
        left = torch.randint(0, W - crop_w + 1, (1,)).item() if W > crop_w else 0
        
        image = image[:, top:top+crop_h, left:left+crop_w]
        depth = depth[:, top:top+crop_h, left:left+crop_w]
        mask = mask[:, top:top+crop_h, left:left+crop_w]
        
        return image, depth, mask


class CenterCrop:
    """Center crop."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target (H, W)
        """
        self.size = size
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (3, H, W)
            depth: (1, H, W)
            mask: (1, H, W)
            
        Returns:
            Center-cropped (image, depth, mask)
        """
        H, W = image.shape[1], image.shape[2]
        crop_h, crop_w = self.size
        
        if H < crop_h or W < crop_w:
            return image, depth, mask
        
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        
        image = image[:, top:top+crop_h, left:left+crop_w]
        depth = depth[:, top:top+crop_h, left:left+crop_w]
        mask = mask[:, top:top+crop_h, left:left+crop_w]
        
        return image, depth, mask


class Resize:
    """Resize images and depth."""
    
    def __init__(self, size: Tuple[int, int], keep_aspect_ratio: bool = False):
        """
        Args:
            size: Target (H, W)
            keep_aspect_ratio: Keep aspect ratio by padding
        """
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (3, H, W)
            depth: (1, H, W)
            mask: (1, H, W)
            
        Returns:
            Resized (image, depth, mask)
        """
        h, w = self.size
        
        image = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear',
                            align_corners=False).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        
        return image, depth, mask


class RandomHFlip:
    """Random horizontal flip."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flip
        """
        self.p = p
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply flip."""
        if torch.rand(1).item() < self.p:
            image = torch.fliplr(image)
            depth = torch.fliplr(depth)
            mask = torch.fliplr(mask)
        
        return image, depth, mask


class RandomVFlip:
    """Random vertical flip."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flip
        """
        self.p = p
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply flip."""
        if torch.rand(1).item() < self.p:
            image = torch.flipud(image)
            depth = torch.flipud(depth)
            mask = torch.flipud(mask)
        
        return image, depth, mask


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        """
        Args:
            transforms: List of transforms
        """
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply all transforms."""
        for transform in self.transforms:
            image, depth, mask = transform(image, depth, mask)
        return image, depth, mask


def get_train_transform(img_size: int = 384) -> Compose:
    """Get training augmentation pipeline."""
    return Compose([
        Resize((img_size, img_size)),
        RandomHFlip(p=0.5),
        RandomVFlip(p=0.1),
    ])


def get_val_transform(img_size: int = 384) -> Compose:
    """Get validation transform (no augmentation)."""
    return Compose([
        Resize((img_size, img_size)),
    ])
