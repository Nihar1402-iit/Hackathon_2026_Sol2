"""
Student model for knowledge distillation.

Can use ViT or CNN as backbone with optional token operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .vit_encoder import ViTEncoder, create_vit_encoder
from .cnn_encoder import CNNEncoder, create_cnn_encoder
from .token_ops import TokenMerging, TokenPruning
from .feature_adapter import TokenToSpatialAdapter, SpatialToTokenAdapter, FeatureAligner
from .decoder import create_decoder


class StudentModel(nn.Module):
    """
    Student model for knowledge distillation.
    
    Supports:
    - ViT student (with optional token merging/pruning)
    - CNN student
    
    Produces depth map and intermediate features for KD loss computation.
    """
    
    def __init__(self, backbone_type: str = 'vit_tiny',
                 img_size: int = 384,
                 patch_size: int = 16,
                 use_token_merging: bool = False,
                 merge_ratio: float = 0.3,
                 use_token_pruning: bool = False,
                 prune_ratio: float = 0.2,
                 decoder_type: str = 'lightweight',
                 base_channels: int = 128,
                 extract_layers: Optional[List[int]] = None):
        """
        Args:
            backbone_type: 'vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                          'resnet18', 'resnet34', 'resnet50', 'resnet101'
            img_size: Input image size
            patch_size: Patch size (ViT only)
            use_token_merging: Whether to use token merging
            merge_ratio: Merging ratio if enabled
            use_token_pruning: Whether to use token pruning
            prune_ratio: Pruning ratio if enabled
            decoder_type: 'hierarchical' or 'lightweight'
            base_channels: Base decoder channels
            extract_layers: Layers to extract features from
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_token_merging = use_token_merging
        self.use_token_pruning = use_token_pruning
        self.decoder_type = decoder_type
        
        # Create backbone
        if backbone_type.startswith('vit'):
            model_size = backbone_type.replace('vit_', '')
            self.backbone = create_vit_encoder(
                model_size=model_size,
                img_size=img_size,
                patch_size=patch_size,
                extract_layers=extract_layers
            )
            self.is_vit = True
            self.embed_dim = self.backbone.embed_dim
            self.n_patches = self.backbone.n_patches
            
        elif backbone_type.startswith('resnet'):
            self.backbone = create_cnn_encoder(
                model_size=backbone_type,
                extract_layers=[1, 2, 3, 4]
            )
            self.is_vit = False
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")
        
        # Token operations (ViT only)
        if use_token_merging and self.is_vit:
            self.token_merger = TokenMerging(merge_ratio=merge_ratio, method='mass')
        else:
            self.token_merger = None
        
        if use_token_pruning and self.is_vit:
            self.token_pruner = TokenPruning(prune_ratio=prune_ratio, use_hard_pruning=False)
        else:
            self.token_pruner = None
        
        # Feature adaptation (token to spatial for ViT)
        if self.is_vit:
            self.token_to_spatial = TokenToSpatialAdapter(self.embed_dim, patch_size)
        else:
            self.token_to_spatial = None
        
        # Create decoder based on backbone type
        if self.is_vit:
            # For ViT, all features have same dimension
            n_features = len(self.backbone.extract_layers)
            feature_dims = [self.embed_dim] * n_features
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                feature_dims=feature_dims,
                base_channels=base_channels
            )
        else:
            # For CNN, use feature dimensions from backbone
            feature_dims = [64, 128, 256, 512]  # Standard ResNet dimensions
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                feature_dims=feature_dims,
                base_channels=base_channels
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            dict with keys:
                - depth: Predicted depth map (B, 1, H, W)
                - features: List of intermediate features
                - tokens: Token representation (ViT only)
                - attention_maps: Attention weights (ViT only)
        """
        if self.is_vit:
            return self._forward_vit(x)
        else:
            return self._forward_cnn(x)
    
    def _forward_vit(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for ViT student."""
        B = x.shape[0]
        
        # Backbone forward
        encoder_output = self.backbone(x)
        tokens = encoder_output['tokens']
        features = encoder_output['features']
        attention_maps = encoder_output['attention_maps']
        
        # Token operations
        if self.token_merger is not None:
            tokens, _ = self.token_merger(tokens, attention_maps[0] if attention_maps else None)
        
        if self.token_pruner is not None:
            tokens, _ = self.token_pruner(tokens)
        
        # Convert tokens to spatial features for decoder
        spatial_features = []
        for feat_tokens in features:
            spatial = self.token_to_spatial(feat_tokens, self.n_patches)
            spatial_features.append(spatial)
        
        # Decoder
        depth = self.decoder(spatial_features)
        
        return {
            'depth': depth,
            'features': spatial_features,
            'tokens': tokens,
            'attention_maps': attention_maps,
            'encoder_features': features  # Keep token-space features too
        }
    
    def _forward_cnn(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for CNN student."""
        # Backbone forward
        encoder_output = self.backbone(x)
        features = encoder_output['features']
        
        # Decoder
        depth = self.decoder(features)
        
        return {
            'depth': depth,
            'features': features,
            'tokens': None,
            'attention_maps': None,
            'encoder_features': features
        }


def create_student_model(model_config: Dict) -> StudentModel:
    """
    Create student model from config dictionary.
    
    Args:
        model_config: Configuration dictionary with keys like
                     'backbone_type', 'use_token_merging', etc.
    
    Returns:
        StudentModel instance
    """
    return StudentModel(**model_config)
