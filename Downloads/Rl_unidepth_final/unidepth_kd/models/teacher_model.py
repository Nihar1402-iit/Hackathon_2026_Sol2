"""
Teacher model for knowledge distillation.

Large, powerful model that provides supervision signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .vit_encoder import ViTEncoder, create_vit_encoder
from .cnn_encoder import CNNEncoder, create_cnn_encoder
from .feature_adapter import TokenToSpatialAdapter
from .decoder import create_decoder


class TeacherModel(nn.Module):
    """
    Teacher model for knowledge distillation.
    
    Always uses ViT or CNN backbone (typically larger than student).
    Produces supervision signals: depth, features, attention maps.
    """
    
    def __init__(self, backbone_type: str = 'vit_base',
                 img_size: int = 384,
                 patch_size: int = 16,
                 decoder_type: str = 'hierarchical',
                 base_channels: int = 128,
                 freeze_backbone: bool = True,
                 extract_layers: Optional[List[int]] = None):
        """
        Args:
            backbone_type: 'vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                          'resnet18', 'resnet34', 'resnet50', 'resnet101'
            img_size: Input image size
            patch_size: Patch size (ViT only)
            decoder_type: 'hierarchical' or 'lightweight'
            base_channels: Base decoder channels
            freeze_backbone: Whether to freeze backbone weights
            extract_layers: Layers to extract features from
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        
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
        
        # Feature adaptation (token to spatial for ViT)
        if self.is_vit:
            self.token_to_spatial = TokenToSpatialAdapter(self.embed_dim, patch_size)
        else:
            self.token_to_spatial = None
        
        # Create decoder
        if self.is_vit:
            # ViT extracts same-scale features from different layers
            # Use lightweight decoder which takes finest features
            n_features = len(self.backbone.extract_layers)
            feature_dims = [self.embed_dim] * n_features
            # For ViT, use lightweight decoder (hierarchical doesn't work with same-scale features)
            actual_decoder_type = 'lightweight' if decoder_type == 'hierarchical' else decoder_type
            self.decoder = create_decoder(
                decoder_type=actual_decoder_type,
                feature_dims=feature_dims,
                base_channels=base_channels
            )
        else:
            feature_dims = [64, 128, 256, 512]
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                feature_dims=feature_dims,
                base_channels=base_channels
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
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
        """Forward pass for ViT teacher."""
        # Backbone forward
        encoder_output = self.backbone(x)
        tokens = encoder_output['tokens']
        features = encoder_output['features']
        attention_maps = encoder_output['attention_maps']
        
        # Convert tokens to spatial features
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
            'encoder_features': features
        }
    
    def _forward_cnn(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for CNN teacher."""
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


def create_teacher_model(model_config: Dict) -> TeacherModel:
    """
    Create teacher model from config dictionary.
    
    Args:
        model_config: Configuration dictionary
    
    Returns:
        TeacherModel instance
    """
    return TeacherModel(**model_config)
