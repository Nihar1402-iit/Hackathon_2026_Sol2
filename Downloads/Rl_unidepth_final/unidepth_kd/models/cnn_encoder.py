"""
CNN Encoder for depth estimation.

Implements a ResNet-based CNN encoder with intermediate feature extraction
for cross-architecture knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ResidualBlock(nn.Module):
    """Residual block for CNN encoder."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C_in, H, W)
            
        Returns:
            Output tensor (B, C_out, H', W')
        """
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class CNNEncoder(nn.Module):
    """ResNet-based CNN encoder for feature extraction."""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 n_blocks: List[int] = None, extract_layers: List[int] = None):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (will be multiplied at each stage)
            n_blocks: Number of blocks at each stage
            extract_layers: Which layers to extract features from
        """
        super().__init__()
        
        if n_blocks is None:
            n_blocks = [2, 2, 2, 2]
        
        if extract_layers is None:
            self.extract_layers = [1, 2, 3, 4]  # All stages
        else:
            self.extract_layers = extract_layers
        
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(base_channels, base_channels, n_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, n_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, n_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, n_blocks[3], stride=2)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels: int, out_channels: int, n_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Create a residual stage."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            dict with keys:
                - features: List of extracted features
                - feature_dims: List of feature dimensions
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        features = []
        feature_dims = []
        
        # Layer 1
        x = self.layer1(x)
        if 1 in self.extract_layers:
            features.append(x)
            feature_dims.append(x.shape[1])
        
        # Layer 2
        x = self.layer2(x)
        if 2 in self.extract_layers:
            features.append(x)
            feature_dims.append(x.shape[1])
        
        # Layer 3
        x = self.layer3(x)
        if 3 in self.extract_layers:
            features.append(x)
            feature_dims.append(x.shape[1])
        
        # Layer 4
        x = self.layer4(x)
        if 4 in self.extract_layers:
            features.append(x)
            feature_dims.append(x.shape[1])
        
        return {
            'features': features,
            'feature_dims': feature_dims,
            'final_feature': x
        }


def create_cnn_encoder(model_size: str = 'resnet50',
                       extract_layers: List[int] = None) -> CNNEncoder:
    """
    Create a CNN encoder with predefined configurations.
    
    Args:
        model_size: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
        extract_layers: Which layers to extract features from
    
    Returns:
        CNNEncoder instance
    """
    configs = {
        'resnet18': {'n_blocks': [2, 2, 2, 2], 'base_channels': 64},
        'resnet34': {'n_blocks': [3, 4, 6, 3], 'base_channels': 64},
        'resnet50': {'n_blocks': [3, 4, 6, 3], 'base_channels': 64},
        'resnet101': {'n_blocks': [3, 4, 23, 3], 'base_channels': 64},
    }
    
    config = configs.get(model_size, configs['resnet50'])
    
    return CNNEncoder(
        in_channels=3,
        base_channels=config['base_channels'],
        n_blocks=config['n_blocks'],
        extract_layers=extract_layers
    )
