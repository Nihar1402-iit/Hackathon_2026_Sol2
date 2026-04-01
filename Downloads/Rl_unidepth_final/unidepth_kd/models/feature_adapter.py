"""
Feature adapters for cross-architecture knowledge distillation.

Implements learnable adapters to align feature distributions
between teacher and student networks of different architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelAdapter(nn.Module):
    """
    Adapt features to target dimension using 1x1 convolution.
    
    For spatial features: F_adapted = Conv1x1(F)
    For tokens: F_adapted = Linear(F)
    """
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            bias: Whether to use bias
        """
        super().__init__()
        if in_dim == out_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, C, H, W) or (B, N, D)
            
        Returns:
            Adapted feature
        """
        return self.adapter(x)


class SpatialAdapter(nn.Module):
    """
    Adapt spatial resolution using interpolation.
    
    Ensures features have matching spatial dimensions for comparison.
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 mode: str = 'bilinear'):
        """
        Args:
            target_size: Target (H, W). If None, inferred at forward time
            mode: Interpolation mode ('bilinear', 'nearest')
        """
        super().__init__()
        self.target_size = target_size
        self.mode = mode
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, C, H, W)
            target: Target feature to match size. If provided, x is resized to match
            
        Returns:
            Resized feature
        """
        if target is not None:
            target_size = (target.shape[2], target.shape[3])
        elif self.target_size is not None:
            target_size = self.target_size
        else:
            return x
        
        if x.shape[2:] == target_size:
            return x
        
        return F.interpolate(x, size=target_size, mode=self.mode, align_corners=False 
                           if self.mode == 'bilinear' else None)


class FeatureNormalization(nn.Module):
    """
    Normalize features for stable comparison.
    
    F_norm = F / ||F||_2
    """
    
    def __init__(self, eps: float = 1e-6, spatial: bool = True):
        """
        Args:
            eps: Small value for numerical stability
            spatial: Whether to normalize spatially (per location) or globally
        """
        super().__init__()
        self.eps = eps
        self.spatial = spatial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, C, H, W) or (B, N, D)
            
        Returns:
            Normalized feature
        """
        if x.dim() == 4:  # Spatial features (B, C, H, W)
            # Normalize per channel
            norm = torch.norm(x, dim=1, keepdim=True)
            return x / (norm + self.eps)
        elif x.dim() == 3:  # Tokens (B, N, D)
            # Normalize per token
            norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (norm + self.eps)
        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (norm + self.eps)


class MultiScaleAdapter(nn.Module):
    """
    Adapt features across multiple scales for multi-scale KD.
    
    Processes features at original and reduced resolutions.
    """
    
    def __init__(self, in_dim: int, out_dim: int, scales: list = None):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            scales: List of scale factors (e.g., [1, 0.5, 0.25])
        """
        super().__init__()
        if scales is None:
            scales = [1.0]
        
        self.scales = scales
        self.channel_adapters = nn.ModuleDict({
            f'scale_{i}': ChannelAdapter(in_dim, out_dim)
            for i in range(len(scales))
        })
        self.spatial_adapters = nn.ModuleDict({
            f'scale_{i}': SpatialAdapter(mode='bilinear')
            for i in range(len(scales))
        })
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            dict with adapted features at each scale
        """
        outputs = {}
        for i, scale in enumerate(self.scales):
            key = f'scale_{i}'
            
            # Resize if needed
            if scale != 1.0:
                h, w = x.shape[2], x.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', 
                                      align_corners=False)
            else:
                scaled = x
            
            # Channel adaptation
            adapted = self.channel_adapters[key](scaled)
            outputs[key] = adapted
        
        return outputs


class TokenToSpatialAdapter(nn.Module):
    """
    Convert token representation to spatial feature map.
    
    Transforms (B, N, D) tokens back to (B, D, H, W) spatial features.
    Excludes CLS token and reshapes based on patch information.
    """
    
    def __init__(self, embed_dim: int, patch_size: int):
        """
        Args:
            embed_dim: Embedding dimension
            patch_size: Patch size for reshaping
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
    
    def forward(self, tokens: torch.Tensor, n_patches: int) -> torch.Tensor:
        """
        Args:
            tokens: Token tensor (B, N+1, D) including CLS token
            n_patches: Total number of patches (N in above)
            
        Returns:
            Spatial features (B, D, H, W)
        """
        B = tokens.shape[0]
        
        # Remove CLS token
        patch_tokens = tokens[:, 1:, :]  # (B, N, D)
        
        # Compute spatial dimensions
        n_patches_side = int(n_patches ** 0.5)
        
        # Reshape to spatial
        spatial = patch_tokens.reshape(B, n_patches_side, n_patches_side, self.embed_dim)
        spatial = spatial.permute(0, 3, 1, 2)  # (B, D, H, W) in patch units
        
        return spatial


class SpatialToTokenAdapter(nn.Module):
    """
    Convert spatial feature map to token representation.
    
    Flattens spatial features and optionally adds CLS token.
    """
    
    def __init__(self, add_cls_token: bool = True):
        """
        Args:
            add_cls_token: Whether to add learnable CLS token
        """
        super().__init__()
        self.add_cls_token = add_cls_token
        if add_cls_token:
            # Will be initialized per embedding dimension
            self.cls_token = None
    
    def forward(self, spatial: torch.Tensor, embed_dim: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            spatial: Spatial features (B, D, H, W)
            embed_dim: Embedding dimension (required if add_cls_token=True)
            
        Returns:
            Token tensor (B, N+1, D) or (B, N, D)
        """
        B, D, H, W = spatial.shape
        
        # Flatten to tokens
        tokens = spatial.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        if self.add_cls_token:
            if self.cls_token is None or self.cls_token.shape[-1] != D:
                self.cls_token = nn.Parameter(torch.randn(1, 1, D)).to(spatial.device)
            
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        
        return tokens


class FeatureAligner(nn.Module):
    """
    Comprehensive feature alignment module.
    
    Combines channel, spatial, and normalization adaptation.
    """
    
    def __init__(self, in_dim: int, out_dim: int, 
                 normalize: bool = True, 
                 target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            normalize: Whether to normalize features
            target_size: Target spatial size
        """
        super().__init__()
        self.channel_adapter = ChannelAdapter(in_dim, out_dim)
        self.spatial_adapter = SpatialAdapter(target_size=target_size)
        self.normalizer = FeatureNormalization() if normalize else nn.Identity()
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input feature
            target: Target feature (for spatial size matching)
            
        Returns:
            Aligned feature
        """
        x = self.channel_adapter(x)
        x = self.spatial_adapter(x, target=target)
        x = self.normalizer(x)
        return x
