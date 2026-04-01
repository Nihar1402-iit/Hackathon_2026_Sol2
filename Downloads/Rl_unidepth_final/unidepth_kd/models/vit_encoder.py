"""
Vision Transformer Encoder for depth estimation.

Implements a pure ViT encoder with intermediate feature extraction
and attention map collection for knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings with positional encoding."""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """
        Args:
            img_size: Input image height/width (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            tokens: (B, N+1, D) - tokens with class token
            pos_embed: (1, N+1, D) - positional embeddings
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x, self.pos_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with temperature scaling."""
    
    def __init__(self, dim: int, n_heads: int = 8, attn_drop: float = 0.0, 
                 proj_drop: float = 0.0, temperature: float = 1.0):
        """
        Args:
            dim: Embedding dimension
            n_heads: Number of attention heads
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            temperature: Temperature scaling for softmax
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            out: Output tensor (B, N, D)
            attn: Attention weights (B, H, N, N)
        """
        B, N, D = x.shape
        
        # Linear projections
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with temperature scaling
        attn = (q @ k.transpose(-2, -1)) * self.scale / self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 drop_path: float = 0.0, temperature: float = 1.0):
        """
        Args:
            dim: Embedding dimension
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            drop_path: Drop path rate (stochastic depth)
            temperature: Temperature for attention
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, n_heads, attn_drop, proj_drop, temperature)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            out: Output tensor (B, N, D)
            attn: Attention weights (B, H, N, N)
        """
        # Attention block
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        
        # MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, attn_weights


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * random_tensor / keep_prob


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for feature extraction."""
    
    def __init__(self, img_size: int = 384, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, n_heads: int = 12, n_layers: int = 12,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.1,
                 attn_drop: float = 0.0, proj_drop: float = 0.1,
                 temperature: float = 1.0, extract_layers: List[int] = None):
        """
        Args:
            img_size: Input image height/width
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            drop_path_rate: Drop path rate
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            temperature: Temperature for attention softmax
            extract_layers: Layers to extract features from (default: [3, 6, 9, 11])
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        if extract_layers is None:
            self.extract_layers = [n_layers // 4 - 1, n_layers // 2 - 1, 
                                   3 * n_layers // 4 - 1, n_layers - 1]
        else:
            self.extract_layers = extract_layers
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, attn_drop, proj_drop,
                           dpr[i], temperature)
            for i in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            dict with keys:
                - tokens: All tokens (B, N+1, D)
                - features: List of extracted features at specified layers
                - attention_maps: List of attention maps at specified layers
                - pos_embed: Positional embeddings (1, N+1, D)
        """
        B = x.shape[0]
        
        # Patch embedding
        tokens, pos_embed = self.patch_embed(x)
        
        features = []
        attention_maps = []
        
        # Transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            tokens, attn = block(tokens)
            
            if layer_idx in self.extract_layers:
                features.append(self.norm(tokens))
                attention_maps.append(attn)
        
        # Final norm
        tokens = self.norm(tokens)
        
        return {
            'tokens': tokens,
            'features': features,
            'attention_maps': attention_maps,
            'pos_embed': pos_embed,
            'patch_size': self.patch_size,
            'n_patches': self.n_patches
        }


def create_vit_encoder(model_size: str = 'base', img_size: int = 384,
                      patch_size: int = 16, temperature: float = 1.0,
                      extract_layers: List[int] = None) -> ViTEncoder:
    """
    Create a ViT encoder with predefined configurations.
    
    Args:
        model_size: 'tiny', 'small', 'base', 'large'
        img_size: Input image size
        patch_size: Patch size
        temperature: Temperature for attention
        extract_layers: Layers to extract features from
    
    Returns:
        ViTEncoder instance
    """
    configs = {
        'tiny': {'embed_dim': 192, 'n_heads': 3, 'n_layers': 12, 'mlp_ratio': 4.0},
        'small': {'embed_dim': 384, 'n_heads': 6, 'n_layers': 12, 'mlp_ratio': 4.0},
        'base': {'embed_dim': 768, 'n_heads': 12, 'n_layers': 12, 'mlp_ratio': 4.0},
        'large': {'embed_dim': 1024, 'n_heads': 16, 'n_layers': 24, 'mlp_ratio': 4.0},
    }
    
    config = configs.get(model_size, configs['base'])
    
    return ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        mlp_ratio=config['mlp_ratio'],
        drop_path_rate=0.1,
        attn_drop=0.0,
        proj_drop=0.1,
        temperature=temperature,
        extract_layers=extract_layers
    )
