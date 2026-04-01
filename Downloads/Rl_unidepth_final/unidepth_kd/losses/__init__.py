"""Losses package."""

from .depth_loss import (
    L1DepthLoss, L2DepthLoss, ScaleInvariantLogLoss,
    GradientLoss, SSIMDepthLoss, create_depth_loss
)
from .kd_losses import (
    FeatureDistillationLoss, AttentionDistillationLoss,
    DepthDistillationLoss, RelationalDistillationLoss,
    PositionalEncodingDistillationLoss, create_kd_loss
)
from .relational_loss import (
    ChannelRelationLoss, MultiScaleRelationalLoss,
    DecoderFeatureDistillationLoss
)

__all__ = [
    'L1DepthLoss',
    'L2DepthLoss',
    'ScaleInvariantLogLoss',
    'GradientLoss',
    'SSIMDepthLoss',
    'create_depth_loss',
    'FeatureDistillationLoss',
    'AttentionDistillationLoss',
    'DepthDistillationLoss',
    'RelationalDistillationLoss',
    'PositionalEncodingDistillationLoss',
    'create_kd_loss',
    'ChannelRelationLoss',
    'MultiScaleRelationalLoss',
    'DecoderFeatureDistillationLoss',
]
