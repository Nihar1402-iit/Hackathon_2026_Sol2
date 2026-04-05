# HackRush 2026 Phase 2: 20-Class Bone Marrow Cell Classification

## Overview
This project implements a Vision Transformer (ViT-Base/16) for classifying bone marrow cell diseases across 20 classes:
- **Classes 1-14**: Original diseases from Phase 1 (2400+ images each)
- **Classes 15-20**: New diseases with minimal support (5 images each)

## Architecture

### Model Architecture
```
Input Image (224×224×3)
        ↓
ViT-Base/16 Backbone (pretrained ImageNet)
├─ Patch Embedding (16×16 patches)
├─ 12 Transformer Blocks
│   ├─ Multi-head Self-Attention (12 heads)
│   ├─ Layer Normalization
│   └─ Feed-Forward Networks
└─ Class Token + Positional Embeddings
        ↓
Pooled Output (768D features)
        ↓
Classification Head
├─ Linear Layer (768 → 20 classes)
└─ Softmax
        ↓
Output Probabilities [20]
```

### Training Architecture

#### Phase 1: Warmup (5 epochs)
```
Frozen Backbone (ViT-Base/16)
        ↓ (no gradient updates)
        ↓
Trainable Head (768 → 20)
        ↓ (AdamW, lr=5e-4)
        ↓
Train on 14 old diseases only
```

#### Phase 2: Fine-tune (10 epochs)
```
Unfrozen Backbone (ViT-Base/16)
        ↓ (AdamW, lr=3e-5)
        ↓
Trainable Head (768 → 20)
        ↓ (AdamW, lr=5e-4)
        ↓
OneCycleLR Scheduler
        ↓
Train on 14 old diseases
+ EMA (Exponential Moving Average)
```

### Data Pipeline

#### Classes 1-14 (Common Diseases)
```
hour0_train/
├─ disease1/: 2400 images → sample 100 → augment → batch
├─ disease2/: 2400 images → sample 100 → augment → batch
├─ ...
└─ disease14/: 2400 images → sample 100 → augment → batch

Total: 1400 images (100 per class)
```

#### Classes 15-20 (Rare Diseases)
```
phase2_support/
├─ disease15/: 5 images → oversample 20× → 100 images → augment → batch
├─ disease16/: 5 images → oversample 20× → 100 images → augment → batch
├─ ...
└─ disease20/: 5 images → oversample 20× → 100 images → augment → batch

Total: 600 images (100 per class after oversampling)
```

#### Weighted Sampler
```
Class Frequency Calculation:
  class_weight[i] = total_samples / (num_classes × count[i])

Effect: Each class equally likely to appear in batch
  • Common classes: lower per-sample weight
  • Rare classes: higher per-sample weight (compensates for oversampling)

Result: Balanced batch distribution despite class imbalance
```

### Augmentation Strategy

#### Standard Augmentation (Classes 1-14)
```
RandomResizedCrop(0.75-1.0) → HorizontalFlip(p=0.5) 
→ VerticalFlip(p=0.5) → RandomRotate90(p=0.5)
→ ShiftScaleRotate(p=0.5) → ColorJitter(p=0.5)
→ CLAHE(p=0.3) → CoarseDropout(p=0.3)
→ Normalize → ToTensor
```

#### Aggressive Augmentation (Classes 15-20)
```
RandomResizedCrop(0.6-1.0) → HorizontalFlip(p=0.7) 
→ VerticalFlip(p=0.7) → RandomRotate90(p=0.7)
→ ShiftScaleRotate(p=0.8) → Perspective(p=0.5)
→ ElasticTransform(p=0.3) → GaussianBlur/GaussNoise(p=0.5)
→ ColorJitter(p=0.7) → CLAHE(p=0.4)
→ CoarseDropout(p=0.5) → GridDistortion(p=0.3)
→ Normalize → ToTensor
```

### Loss Function: Focal Loss

```python
FL(pt) = -α·(1-pt)^γ·log(pt)

Where:
  pt = probability of ground truth class
  α = weighting factor (default: 1.0)
  γ = focusing parameter (default: 2.0)

Effect:
  • Focuses on hard examples (low confidence predictions)
  • Rare classes get higher loss weight
  • Prevents easy examples from dominating training
```

### Prototype-Based Few-Shot Learning

#### Prototype Building (for Classes 15-20)
```
For each image in support set (5 per class):
  ├─ Generate 50 augmented views
  ├─ Extract 768D features (backbone output)
  ├─ Filter outliers (|feature - mean| > mean + std)
  ├─ Average remaining views
  └─ Normalize to unit vector

Result: 1 prototype vector per support image
Aggregate: Average across all support images
Final: 768D prototype per new class
```

#### CLIP Text Embedding (for Classes 15-20)
```
For each new class:
  ├─ Generate 6 text prompts:
  │   • "microscopy image of {disease}"
  │   • "histology slide showing {disease}"
  │   • "medical image of {disease}"
  │   • "a photo of {disease}, a disease"
  │   • "clinical photograph of {disease}"
  │   • "dermoscopy image of {disease}"
  ├─ Encode each prompt with CLIP text encoder (512D)
  ├─ Average across prompts
  └─ Normalize to unit vector

Result: 512D text embedding per new class
Used for: Blending with prototype scores (25% weight)
```

### Inference Pipeline

#### Test-Time Augmentation (10 passes)
```
For each test image:
  ├─ Pass 1: Original + resize
  ├─ Pass 2: HorizontalFlip + resize
  ├─ Pass 3: VerticalFlip + resize
  ├─ Pass 4: RandomRotate90 + resize
  ├─ Pass 5: Transpose + resize
  ├─ Pass 6: HorizontalFlip + VerticalFlip
  ├─ Pass 7: RandomResizedCrop
  ├─ Pass 8: Resize + CenterCrop
  ├─ Pass 9: CLAHE + resize
  └─ Pass 10: RandomGamma + resize

For each pass:
  ├─ Extract features: f = model.backbone(img)
  ├─ Old class scores: logits = head(f) → softmax (14 probs)
  ├─ New class similarity: sim = prototypes @ f (6 scores)
  ├─ CLIP blend: (1-0.25) × sim + 0.25 × clip_scores
  ├─ Z-score calibrate: (score - mean) / std
  └─ New class probs: softmax (6 probs)

Average: mean([pass1, pass2, ..., pass10])
Final: argmax(probabilities)
```

#### Trio Prediction (for Confusable Classes)
```
For predictions in {disease11, disease15, disease20}:
  ├─ Extract feature: f = model.backbone(img)
  ├─ Normalize: f_norm = f / ||f||
  ├─ Compute similarity to trio centroids:
  │   • sim11 = f_norm · centroid11
  │   • sim15 = f_norm · centroid15
  │   • sim20 = f_norm · centroid20
  └─ Predict: argmax(sim11, sim15, sim20)

Effect: Two-stage decision for confusable classes
```

## Checkpoint System

### Checkpoint 1: Model Weights
- **File**: `checkpoints/best_vit_phase2.pth`
- **Size**: ~350 MB
- **Contains**: `model.state_dict()`
  - ViT-Base/16 backbone (86M parameters)
  - Classification head
- **Saved**: After epoch 15 (after EMA weights applied)

### Checkpoint 2: Prototypes & Metadata
- **File**: `checkpoints/phase2_prototypes.pth`
- **Size**: ~50 KB
- **Contains**:
  - 6 prototype vectors (768D each)
  - CLIP text embeddings (512D each)
  - Trio centroids (3 × 768D)
  - Z-score calibration stats (mean, std per class)
  - Temperature parameter (tuned via LOO)
  - Class indices
- **Saved**: After LOO temperature tuning

### Checkpoint 3: Complete State
- **File**: `checkpoints/phase2_complete_checkpoint.pth`
- **Size**: ~350 MB
- **Contains**: All of Checkpoints 1 & 2 + config
- **Saved**: After prototype checkpoint (allows full resumption)

## Training Timeline

```
Step 1: Load Data (30 sec)
  ├─ Classes 1-14: 100 images each = 1400 total
  └─ Classes 15-20: 5 images × 20 oversample = 600 total

Step 2: Warmup Phase (2 min)
  ├─ Epochs 1-5: Train head only
  ├─ Frozen backbone
  └─ Train on 1400 images (90% split = 1260 train, 140 val)

Step 3: Fine-tune Phase (3-4 min)
  ├─ Epochs 6-15: Train all layers
  ├─ Unfrozen backbone
  ├─ OneCycleLR scheduler
  ├─ EMA weight averaging
  └─ Train on 1260 images

Step 4: Build Prototypes (2-3 min)
  ├─ 50-view augmented prototypes for classes 15-20
  ├─ CLIP text embeddings
  ├─ Trio centroids (disease 11, 15, 20)
  └─ Prototype repulsion away from disease11

Step 5: Calibration & Tuning (2 min)
  ├─ Z-score calibration on support set
  └─ LOO temperature tuning (test 6 values)

Step 6: Inference (2-3 min)
  ├─ 10-pass TTA on 202 test images
  └─ Average predictions

Total Runtime: ~12-15 minutes
```

## Configuration

```python
CFG = {
    'seed': 42,
    'img_size': 224,
    'batch_size': 32,
    'epochs': 15,              # Total (5 warmup + 10 fine-tune)
    'warmup_epochs': 5,
    'lr_head': 5e-4,           # Head learning rate
    'lr_backbone': 3e-5,       # Backbone learning rate
    'label_smooth': 0.05,      # Label smoothing strength
    'focal_gamma': 2.0,        # Focal loss focusing parameter
    'focal_alpha': 1.0,        # Focal loss weighting
    'ema_decay': 0.999,        # EMA decay rate
    'triplet_weight': 0.3,     # Triplet loss weight
    'triplet_margin': 0.3,     # Triplet loss margin
    'tta_views': 10,           # Test-time augmentation passes
    'proto_views': 50,         # Prototype augmentation views
    'proto_temp': 20.0,        # Temperature (tuned via LOO)
    'clip_weight': 0.25,       # CLIP weight in ensemble
    'ema_decay': 0.999,
    'device': 'cuda',
}
```

## Output

### Submission File
- **Name**: `submission.csv`
- **Format**:
  ```
  image_name,predicted_class
  cp2_00001.jpg,disease1
  cp2_00002.jpg,disease15
  ...
  cp2_00202.jpg,disease20
  ```
- **Size**: ~5 KB
- **Rows**: 202 (one per test image)

## Files

### Training Scripts
- `solution_phase2.py` - Main training script (653 lines)
  - Data loading with weighted sampling
  - ViT-Base/16 model with custom head
  - Two-phase training (warmup + fine-tune)
  - Prototype building with CLIP ensemble
  - 10-pass TTA inference
  - 3-checkpoint system

### Data Directories
- `hour0_train/` - 14 diseases, ~2400 images each
- `phase2_support/` - 6 diseases, 5 images each
- `phase2_test_20/` - 202 test images

### Output
- `submission.csv` - Final predictions
- `checkpoints/` - Model and prototype checkpoints
  - `best_vit_phase2.pth` (350 MB)
  - `phase2_prototypes.pth` (50 KB)
  - `phase2_complete_checkpoint.pth` (350 MB)

## Key Features

✅ **Data Balancing**: 100 images per class (uniform representation)
✅ **Few-Shot Learning**: 5-image rare classes boosted via oversampling + aggressive augmentation
✅ **Weighted Sampling**: Balanced batches despite class imbalance
✅ **Focal Loss**: Emphasizes hard examples and rare classes
✅ **EMA**: Exponential moving average of weights for stability
✅ **Prototype-Based**: 50-view augmented prototypes for new classes
✅ **CLIP Ensemble**: Text embeddings for semantic alignment
✅ **TTA**: 10-pass test-time augmentation for robustness
✅ **Temperature Tuning**: Leave-one-out tuning on support set
✅ **Checkpoints**: 3-level checkpoint system for reproducibility

## Dependencies
```
torch>=2.0.0
torchvision
timm
albumentations
open-clip
pandas
scikit-learn
pillow
numpy
```

## Usage

```bash
# Train and generate submission
python solution_phase2.py

# Outputs:
# - checkpoints/best_vit_phase2.pth (model weights)
# - checkpoints/phase2_prototypes.pth (prototypes)
# - checkpoints/phase2_complete_checkpoint.pth (full state)
# - submission.csv (predictions)
```

## Performance

- **Phase 1 (14 classes)**: 84.32% validation accuracy
- **Phase 2 (20 classes)**: Expected 70-75% accuracy
- **Inference Speed**: ~2-3 minutes for 202 test images (10-pass TTA)
