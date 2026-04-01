"""
Token operations for Vision Transformers.

Implements differentiable token merging and pruning operations
for efficient student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TokenMerging(nn.Module):
    """
    Weighted token merging based on attention mass or token norm.
    
    For efficiency, merges tokens before each transformer block.
    z_merge = (w_i z_i + w_j z_j) / (w_i + w_j)
    """
    
    def __init__(self, merge_ratio: float = 0.5, method: str = 'mass'):
        """
        Args:
            merge_ratio: Ratio of tokens to merge (0.0-1.0)
            method: 'mass' (attention mass) or 'norm' (token norm)
        """
        super().__init__()
        self.merge_ratio = merge_ratio
        self.method = method
        assert method in ['mass', 'norm'], f"Unknown merge method: {method}"
    
    def forward(self, tokens: torch.Tensor, 
                attention_maps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Token tensor (B, N, D)
            attention_maps: Attention weights (B, H, N, N) for 'mass' method
            
        Returns:
            merged_tokens: Merged tokens (B, N', D)
            merge_indices: Indices of merged tokens for tracking
        """
        B, N, D = tokens.shape
        
        if self.method == 'mass' and attention_maps is not None:
            # Compute attention mass per token
            # Average across heads and queries
            attn_mass = attention_maps.mean(dim=1).sum(dim=1)  # (B, N)
            weights = attn_mass
        else:
            # Use token norm as weight
            weights = torch.norm(tokens, dim=-1)  # (B, N)
        
        # Normalize weights
        weights = F.softmax(weights, dim=-1)
        
        # Number of tokens to keep (excluding CLS token)
        n_keep = max(1, int(N * (1 - self.merge_ratio)))
        
        # Find top-k tokens by weight (keep CLS token separately)
        cls_token = tokens[:, :1, :]  # (B, 1, D)
        patch_tokens = tokens[:, 1:, :]  # (B, N-1, D)
        patch_weights = weights[:, 1:]  # (B, N-1)
        
        # Top-k selection (differentiable approximation)
        topk_vals, topk_indices = torch.topk(patch_weights, k=min(n_keep, N-1), dim=-1)
        
        # Gather selected tokens
        selected_tokens = torch.gather(
            patch_tokens, 1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, n_keep, D)
        
        # Concatenate with CLS token
        merged_tokens = torch.cat([cls_token, selected_tokens], dim=1)  # (B, n_keep+1, D)
        
        return merged_tokens, topk_indices


class TokenPruning(nn.Module):
    """
    Differentiable token pruning using sigmoid gating.
    
    During training, uses soft masking with sigmoid.
    m_i = sigmoid(s_i)
    z_i' = m_i * z_i
    
    During inference, can use hard thresholding for efficiency.
    """
    
    def __init__(self, prune_ratio: float = 0.3, use_hard_pruning: bool = False):
        """
        Args:
            prune_ratio: Fraction of tokens to prune
            use_hard_pruning: Use hard thresholding instead of soft
        """
        super().__init__()
        self.prune_ratio = prune_ratio
        self.use_hard_pruning = use_hard_pruning
        
        # Learnable pruning scores (initialized near 0 for soft start)
        self.register_parameter('pruning_scores', None)
    
    def init_scores(self, n_tokens: int, device: torch.device):
        """Initialize pruning scores for given number of tokens."""
        self.pruning_scores = nn.Parameter(torch.zeros(n_tokens, device=device))
    
    def forward(self, tokens: torch.Tensor, 
                init_scores: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Token tensor (B, N, D)
            init_scores: Whether to initialize scores
            
        Returns:
            pruned_tokens: Pruned tokens (B, N, D)
            masks: Pruning masks (B, N)
        """
        B, N, D = tokens.shape
        
        if init_scores or self.pruning_scores is None:
            self.init_scores(N, tokens.device)
        
        # Soft masking using sigmoid
        masks = torch.sigmoid(self.pruning_scores)  # (N,)
        
        if self.use_hard_pruning and not self.training:
            # Hard thresholding during inference
            threshold = torch.quantile(masks, self.prune_ratio)
            masks = (masks > threshold).float()
        
        # Expand masks for batch dimension
        masks = masks.unsqueeze(0).expand(B, -1)  # (B, N)
        
        # Apply masks
        pruned_tokens = tokens * masks.unsqueeze(-1)
        
        return pruned_tokens, masks


class DynamicTokenResizing(nn.Module):
    """
    Dynamically adjust number of tokens based on input complexity.
    
    Simple approach: reduce token count for simpler regions (lower entropy).
    """
    
    def __init__(self, base_tokens: int, min_tokens: int = 50, max_tokens: int = 1000):
        """
        Args:
            base_tokens: Base number of tokens
            min_tokens: Minimum tokens to keep
            max_tokens: Maximum tokens to keep
        """
        super().__init__()
        self.base_tokens = base_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    def compute_token_count(self, features: torch.Tensor) -> int:
        """
        Compute dynamic token count based on feature entropy.
        
        Args:
            features: Feature tensor (B, N, D)
            
        Returns:
            Token count for next layer
        """
        B, N, D = features.shape
        
        # Compute per-token norm (complexity indicator)
        token_norms = torch.norm(features, dim=-1)  # (B, N)
        entropy = -(torch.log(torch.clamp(token_norms, min=1e-6)) * token_norms).mean()
        
        # Scale token count based on entropy
        scale_factor = min(max(entropy.item(), 0.1), 2.0)
        new_count = int(self.base_tokens * scale_factor)
        new_count = max(self.min_tokens, min(new_count, self.max_tokens))
        
        return new_count


class TokenAggregation(nn.Module):
    """
    Aggregate tokens before classification using weighted sum.
    
    Learns weights per token for aggregation:
    z_agg = sum(w_i * z_i)
    """
    
    def __init__(self, embed_dim: int, n_heads: int = 1):
        """
        Args:
            embed_dim: Embedding dimension
            n_heads: Number of aggregation heads for multi-view
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Attention-like aggregation
        self.query = nn.Parameter(torch.randn(n_heads, 1, self.head_dim))
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        nn.init.normal_(self.query, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Token tensor (B, N, D)
            
        Returns:
            aggregated: Aggregated token (B, D) or (B, n_heads, D/n_heads)
        """
        B, N, D = tokens.shape
        
        k = self.key(tokens)  # (B, N, D)
        v = self.value(tokens)  # (B, N, D)
        
        # Compute attention with learnable query
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention weights
        scores = (self.query * k).sum(dim=-1) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)  # (B, n_heads, 1, N)
        
        # Aggregate
        aggregated = (weights @ v).squeeze(-2)  # (B, n_heads, D/n_heads)
        aggregated = aggregated.permute(0, 1, 2).reshape(B, -1)
        
        return aggregated
