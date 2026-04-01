# UniDepth-KD: Implementation Guide

Complete implementation guide for the monocular depth estimation with Knowledge Distillation system.

## Project Overview

UniDepth-KD is a production-ready PyTorch framework for training efficient monocular depth estimation models using Knowledge Distillation (KD). It supports:

- **ViT → ViT** distillation (same architecture, different sizes)
- **ViT → CNN** cross-architecture distillation
- **Token operations** (merging, pruning) for efficiency
- **8 complementary loss functions** for comprehensive supervision
- **Mixed precision training** with gradient clipping
- **Multi-scale feature extraction** for rich supervision signals

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended

### Setup

```bash
# Clone and setup
cd unidepth_kd
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Core Components

### 1. Vision Transformer Encoder (`models/vit_encoder.py`)

**Features:**
- Configurable ViT architecture (tiny/small/base/large)
- Multi-head attention with temperature scaling
- Intermediate feature extraction at 4 depths
- Attention map collection

**Key Classes:**
- `ViTEncoder`: Main ViT implementation
- `PatchEmbedding`: Image-to-token conversion
- `MultiHeadAttention`: MHA with temperature
- `TransformerBlock`: Attention + MLP block

**Usage:**
```python
from models import create_vit_encoder

vit = create_vit_encoder(
    model_size='base',           # tiny/small/base/large
    img_size=384,
    patch_size=16,
    temperature=1.0,
    extract_layers=[3, 6, 9, 11]
)

output = vit(x)  # x: (B, 3, H, W)
# Returns:
# - tokens: (B, N+1, D) - including CLS token
# - features: List of 4 feature maps
# - attention_maps: List of attention weights
# - pos_embed: Positional embeddings
```

### 2. CNN Encoder (`models/cnn_encoder.py`)

**Features:**
- ResNet-based architecture (18/34/50/101)
- Feature extraction at 4 stages
- Backward compatibility with ViT

**Key Classes:**
- `CNNEncoder`: ResNet implementation
- `ResidualBlock`: Standard residual block

**Usage:**
```python
from models import create_cnn_encoder

cnn = create_cnn_encoder(
    model_size='resnet50',
    extract_layers=[1, 2, 3, 4]
)
```

### 3. Token Operations (`models/token_ops.py`)

**Token Merging:**
```python
from models import TokenMerging

merger = TokenMerging(merge_ratio=0.3, method='mass')
merged_tokens, indices = merger(tokens, attention_maps)
```

Reduces sequence length by merging similar tokens based on attention mass.

**Token Pruning:**
```python
from models import TokenPruning

pruner = TokenPruning(prune_ratio=0.2, use_hard_pruning=False)
pruned_tokens, masks = pruner(tokens)
```

Soft masking during training, hard thresholding during inference.

### 4. Feature Adaptation (`models/feature_adapter.py`)

**Handles cross-architecture alignment:**
```python
from models import FeatureAligner

aligner = FeatureAligner(in_dim=768, out_dim=768, normalize=True)
aligned = aligner(student_features, teacher_features)
```

Automatically:
- Projects to common dimension (1×1 conv)
- Matches spatial resolution (bilinear interpolation)
- Normalizes for stability (L2 normalization)

### 5. Decoder (`models/decoder.py`)

**Hierarchical Decoder:**
```python
from models import create_decoder

decoder = create_decoder(
    decoder_type='hierarchical',
    feature_dims=[768, 768, 768, 768],
    base_channels=128
)

depth = decoder(features)  # features: List[4x(B, D, H, W)]
```

Progressive upsampling with multi-scale fusion.

**Lightweight Decoder:**
```python
decoder = create_decoder(
    decoder_type='lightweight',
    in_channels=768,
    base_channels=128
)
```

Simpler, faster variant for efficient students.

### 6. Student & Teacher Models

**Student Model (Trainable):**
```python
from models import create_student_model

student = create_student_model({
    'backbone_type': 'vit_tiny',
    'img_size': 384,
    'use_token_merging': True,
    'merge_ratio': 0.3,
    'decoder_type': 'lightweight',
})

output = student(images)
# Returns: depth, features, tokens, attention_maps
```

**Teacher Model (Frozen):**
```python
from models import create_teacher_model

teacher = create_teacher_model({
    'backbone_type': 'vit_base',
    'freeze_backbone': True,
})

with torch.no_grad():
    output = teacher(images)
```

## Loss Functions

### Depth Losses

```python
from losses import create_depth_loss

# Scale-invariant log loss (recommended for monocular)
loss_fn = create_depth_loss('silog')
loss = loss_fn(depth_pred, depth_gt)

# Other options: 'l1', 'l2', 'gradient', 'ssim'
```

### Knowledge Distillation Losses

```python
from losses import create_kd_loss

# Feature distillation
feat_loss = create_kd_loss('feature')
loss = feat_loss(student_features, teacher_features)

# Attention distillation
attn_loss = create_kd_loss('attention')
loss = attn_loss(student_attentions, teacher_attentions)

# Depth distillation
depth_loss = create_kd_loss('depth')
loss = depth_loss(student_depth, teacher_depth)

# Relational distillation
rel_loss = create_kd_loss('relational')
loss = rel_loss(student_features, teacher_features)

# Positional encoding distillation
pos_loss = create_kd_loss('positional')
loss = pos_loss(student_pos_embed, teacher_pos_embed)
```

### Total Loss

```python
total_loss = (
    1.0 * l_depth +           # Ground truth supervision
    0.1 * l_silog +           # Scale-invariant loss
    0.5 * l_feat +            # Feature matching
    0.1 * l_attn +            # Attention matching
    0.5 * l_depth_kd +        # Depth KD
    0.1 * l_rel +             # Relational matching
    0.3 * l_decoder +         # Decoder features
    0.05 * l_pos              # Positional embeddings
)
```

All weights are configurable in `configs/config.yaml`.

## Training

### Configuration

Edit `configs/config.yaml`:

```yaml
student_model:
  backbone_type: 'vit_tiny'
  use_token_merging: true
  merge_ratio: 0.3

teacher_model:
  backbone_type: 'vit_base'
  freeze_backbone: true

training:
  epochs: 100
  batch_size: 8
  learning_rate: 1.0e-4
  use_amp: true
  use_kd: true
```

### Starting Training

```bash
python main.py train --config configs/config.yaml
```

### Resuming from Checkpoint

```bash
python main.py train --config configs/config.yaml --resume checkpoints/checkpoint_epoch_050.pth
```

### Training Loop Details

1. **Teacher forward pass** (no gradients)
   - Extracts multi-scale features
   - Collects attention maps

2. **Student forward pass** (with gradients)
   - Applies token operations
   - Extracts features
   - Produces depth

3. **Loss computation**
   - 8 complementary losses
   - Feature alignment before comparison

4. **Optimization**
   - Mixed precision (FP16/FP32)
   - Gradient clipping (norm=1.0)
   - Cosine annealing schedule

## Evaluation

### Standard Metrics

```python
from utils import compute_metrics

metrics = compute_metrics(depth_pred, depth_gt)
# Returns: {
#   'rmse': float,
#   'abs_rel': float,
#   'delta1': float,
#   'delta2': float,
#   'delta3': float,
#   'silog': float
# }
```

### Running Evaluation

```bash
python main.py eval --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

## Inference

### Single Image Inference

```python
from models import create_student_model
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
student = create_student_model({...})
checkpoint = torch.load('checkpoints/best_model.pth')
student.load_state_dict(checkpoint['student_state_dict'])
student.eval()

# Load image
image = Image.open('image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])
x = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = student(x)
    depth = output['depth']

print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
```

### Batch Inference

```bash
python main.py infer --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image test_image.jpg \
    --output depth_map.pt
```

## Model Profiling

### FLOPs and Memory

```bash
python main.py benchmark --config configs/config.yaml
```

Output includes:
- Total/trainable parameters
- FLOPs estimation
- Forward pass time (ms)
- Peak memory usage (MB)

### Custom Profiling

```python
from utils import ModelProfiler, count_parameters

student = create_student_model({...})

# Parameter count
total, trainable = count_parameters(student)
print(f"Parameters: {total:,} ({trainable:,} trainable)")

# Performance profiling
profiler = ModelProfiler(student, device='cuda')
forward_time = profiler.profile_forward((1, 3, 384, 384))
peak_mem, alloc_mem = profiler.profile_memory((1, 3, 384, 384))

print(f"Forward time: {forward_time:.2f} ms")
print(f"Peak memory: {peak_mem:.2f} MB")
```

## Visualization

```python
from utils import (
    visualize_depth, 
    save_depth_visualization,
    compare_depth_maps,
    visualize_error
)

# Single depth map
rgb = visualize_depth(depth, cmap='turbo')

# Save visualization
save_depth_visualization(depth, 'depth.png')

# Compare predictions vs ground truth
comparison = compare_depth_maps(depth_pred, depth_gt)

# Visualize error
error_map = visualize_error(depth_pred, depth_gt, error_type='abs')
```

## Dataset Support

### Mock Dataset (Testing)

```python
from data import create_dataset

dataset = create_dataset(
    dataset_type='mock',
    split='train',
    img_size=384,
    n_samples=100
)
```

### Real Datasets

```python
# NYU Depth V2
dataset = create_dataset(
    dataset_type='nyu',
    split='train',
    data_dir='./data/nyu',
    img_size=384
)

# KITTI
dataset = create_dataset(
    dataset_type='kitti',
    split='train',
    data_dir='./data/kitti',
    img_size=384
)
```

### Custom Dataset

Subclass `DepthDataset`:

```python
from data import DepthDataset

class CustomDataset(DepthDataset):
    def __getitem__(self, idx):
        # Load image and depth
        image = ...  # (3, H, W), float [0, 1]
        depth = ...  # (1, H, W), float
        mask = ...   # (1, H, W), binary
        
        return {
            'image': image,
            'depth': depth,
            'mask': mask
        }
```

## Data Transforms

```python
from data import get_train_transform, get_val_transform

train_transform = get_train_transform(img_size=384)
val_transform = get_val_transform(img_size=384)

# Manual composition
from data.transforms import Compose, Resize, RandomHFlip, RandomVFlip

transform = Compose([
    Resize((384, 384)),
    RandomHFlip(p=0.5),
    RandomVFlip(p=0.1),
])

image, depth, mask = transform(image, depth, mask)
```

## Advanced Topics

### Token Operations in Detail

**Token Merging:**
- Reduces computation by merging similar tokens
- Based on attention mass (which tokens attend together)
- Differentiable for end-to-end training
- Effective for tiny/small models

**Token Pruning:**
- Removes unimportant tokens
- Uses soft sigmoid masking during training
- Hard thresholding during inference
- Gradual effect (not sudden)

### Cross-Architecture Distillation

ViT → CNN example:

```python
student = create_student_model({
    'backbone_type': 'resnet50',  # CNN
    'decoder_type': 'hierarchical'
})

teacher = create_teacher_model({
    'backbone_type': 'vit_base',  # ViT
    'freeze_backbone': True
})

# Feature dimensions are automatically aligned
# (CNN: [64, 128, 256, 512], ViT: [768, 768, 768, 768])
```

### Numerical Stability

Ensures stable training:

1. **Log operations:** `log(x + ε)` with ε=1e-6
2. **Division:** `x / (y + ε)` with ε=1e-8
3. **Softmax:** Numerically stable implementation
4. **Normalization:** LayerNorm with ε=1e-6
5. **Mixed precision:** AMP with loss scaling

### Memory Efficiency

For limited GPU memory:

1. **Reduce batch size:** `batch_size: 4`
2. **Use token merging:** `merge_ratio: 0.5`
3. **Lightweight decoder:** `decoder_type: 'lightweight'`
4. **Reduce input size:** `img_size: 256`
5. **Gradient checkpointing:** (future enhancement)

## Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size
# In config.yaml:
training:
  batch_size: 4  # from 8

# Or use token merging
student_model:
  use_token_merging: true
  merge_ratio: 0.5  # merge more tokens
```

### Unstable Training (NaN/Inf)

```yaml
training:
  learning_rate: 5.0e-5  # reduce LR
  gradient_clip_norm: 1.0
  use_amp: true  # mixed precision helps

training:
  loss_weights:
    depth_kd: 0.1  # reduce KD weight
```

### Poor Results

1. Check that teacher is pre-trained and performs well
2. Verify ground truth depth normalization
3. Increase training epochs
4. Adjust loss weights for your dataset
5. Check feature alignment (print shapes)

## Performance Benchmarks

Example results on mock dataset:

| Model | Params | FLOPs | Forward (ms) | Memory (MB) |
|-------|--------|-------|--------------|-------------|
| ViT Tiny | 5.7M | 4.0G | 8.2 | 120 |
| ViT Small | 22.1M | 15.1G | 18.4 | 280 |
| ViT Base | 86.6M | 55.3G | 54.7 | 850 |
| ResNet18 | 11.2M | 7.3G | 5.1 | 90 |
| ResNet50 | 23.5M | 9.1G | 12.3 | 220 |

*Note: Benchmarks are approximate and depend on hardware/implementation.*

## Contributing

Areas for enhancement:
- Real dataset support (NYU, KITTI with proper loading)
- Additional architectures (DeiT, Swin, etc.)
- Pruning and quantization (excluding per requirements)
- Distillation for semantic segmentation
- Multi-task learning framework

## References

1. Dosovitskiy et al. "An Image is Worth 16x16 Words"
2. Hinton et al. "Distilling the Knowledge in a Neural Network"
3. Eigen et al. "Depth Map Prediction from a Single Image"

---

**Last Updated:** April 2026
**Version:** 1.0.0
