"""
Depth decoder module.

Implements a hierarchical decoder that progressively upsamples
multi-scale features to produce final depth map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise separable convolution block.
    
    More efficient than standard convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
        """
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C_in, H, W)
            
        Returns:
            Output tensor (B, C_out, H, W)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and fusion.
    
    F_{i+1}^{up} = BilinearUpsample(F_{i+1})
    F_i^{fusion} = DepthwiseConv(F_i + F_{i+1}^{up})
    """
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        """
        Args:
            in_channels: Input channels from previous layer
            out_channels: Output channels
            skip_channels: Channels from skip connection
        """
        super().__init__()
        
        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Fusion convolution (after concatenation with skip)
        total_channels = in_channels + skip_channels
        self.fusion_conv = DepthwiseConvBlock(total_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input from previous layer (B, C_in, H, W)
            skip: Skip connection features (B, C_skip, H*2, W*2)
            
        Returns:
            Output (B, C_out, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # Fuse with skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Process
        x = self.fusion_conv(x)
        
        return x


class DepthHead(nn.Module):
    """
    Final depth prediction head.
    
    Predicts depth map from feature maps.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 1, num_convs: int = 2):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels (1 for single depth)
            num_convs: Number of convolutions before output
        """
        super().__init__()
        
        convs = []
        for i in range(num_convs):
            if i == 0:
                convs.append(DepthwiseConvBlock(in_channels, in_channels // 2, kernel_size=3))
            else:
                convs.append(DepthwiseConvBlock(in_channels // 2, in_channels // 2, kernel_size=3))
        
        self.convs = nn.Sequential(*convs)
        
        # Final depth projection
        self.depth_proj = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature (B, C, H, W)
            
        Returns:
            Depth map (B, 1, H, W)
        """
        x = self.convs(x)
        depth = self.depth_proj(x)
        # Use softplus to ensure positive depth
        depth = F.softplus(depth)
        return depth


class HierarchicalDecoder(nn.Module):
    """
    Multi-scale hierarchical decoder.
    
    Takes features at multiple scales and progressively upsamples
    to produce high-resolution depth map.
    
    Input: List of features at different scales
    [F_0 (finest), F_1, F_2, F_3 (coarsest)]
    
    Output: Depth map at original resolution
    """
    
    def __init__(self, feature_dims: List[int], base_channels: int = 128):
        """
        Args:
            feature_dims: List of input feature dimensions at each scale
            base_channels: Base number of channels in decoder
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.n_scales = len(feature_dims)
        
        # Initial feature adaptation
        self.adapters = nn.ModuleList([
            nn.Conv2d(dim, base_channels, kernel_size=1)
            for dim in feature_dims
        ])
        
        # Decoder blocks (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.n_scales - 1):
            block = DecoderBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                skip_channels=base_channels
            )
            self.decoder_blocks.append(block)
        
        # Depth head
        self.depth_head = DepthHead(base_channels, out_channels=1, num_convs=2)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of features from coarse to fine
                     features[0] is coarsest, features[-1] is finest
        
        Returns:
            Depth map (B, 1, H_orig, W_orig)
        """
        assert len(features) == self.n_scales, \
            f"Expected {self.n_scales} features, got {len(features)}"
        
        # Adapt features
        adapted = [adapter(feat) for adapter, feat in zip(self.adapters, features)]
        
        # Reverse to go from coarse to fine
        adapted = adapted[::-1]  # Now [finest, ..., coarsest]
        
        # Start from coarsest
        x = adapted[-1]
        
        # Progressive upsampling
        for i, block in enumerate(self.decoder_blocks):
            skip = adapted[-(i+2)]  # Get skip connection
            x = block(x, skip)
        
        # Depth prediction
        depth = self.depth_head(x)
        
        return depth


class LightweightDecoder(nn.Module):
    """
    Lightweight decoder for efficient student models.
    
    Simpler and faster than hierarchical decoder.
    """
    
    def __init__(self, in_channels: int, base_channels: int = 64):
        """
        Args:
            in_channels: Input feature channels
            base_channels: Base number of channels
        """
        super().__init__()
        
        # Quick upsampling path
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseConvBlock(in_channels, base_channels, kernel_size=3)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseConvBlock(base_channels, base_channels, kernel_size=3)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseConvBlock(base_channels, base_channels, kernel_size=3)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseConvBlock(base_channels, base_channels, kernel_size=3)
            ),
        ])
        
        # Depth head
        self.depth_head = DepthHead(base_channels, out_channels=1, num_convs=1)
    
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Input features - either a tensor (B, C, H, W) or list of tensors
            
        Returns:
            Depth map (B, 1, H_orig, W_orig)
        """
        # If x is a list, use the last (finest) feature
        if isinstance(x, list):
            x = x[-1]
        
        # Progressive upsampling
        for block in self.upsample_blocks:
            x = block(x)
        
        # Depth prediction
        depth = self.depth_head(x)
        
        return depth


def create_decoder(decoder_type: str = 'hierarchical',
                   feature_dims: Optional[List[int]] = None,
                   in_channels: Optional[int] = None,
                   base_channels: int = 128) -> nn.Module:
    """
    Create decoder with specified configuration.
    
    Args:
        decoder_type: 'hierarchical' or 'lightweight'
        feature_dims: List of feature dimensions (required for hierarchical)
        in_channels: Input channels (required for lightweight)
        base_channels: Base channels
    
    Returns:
        Decoder module
    """
    if decoder_type == 'hierarchical':
        assert feature_dims is not None, "feature_dims required for hierarchical decoder"
        return HierarchicalDecoder(feature_dims, base_channels)
    elif decoder_type == 'lightweight':
        # If in_channels not provided, use last feature dim
        if in_channels is None and feature_dims is not None:
            in_channels = feature_dims[-1]
        assert in_channels is not None, "in_channels required for lightweight decoder"
        return LightweightDecoder(in_channels, base_channels)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
