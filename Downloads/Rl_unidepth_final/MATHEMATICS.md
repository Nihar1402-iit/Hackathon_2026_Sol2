# Mathematical Details and Formulations

## Overview

This document provides the mathematical foundations for the knowledge distillation framework used in UniDepth-KD.

## 1. Vision Transformer (ViT) Architecture

### 1.1 Patch Embedding

Given input image: **x** ∈ ℝ^(B×3×H×W)

**Step 1:** Split into patches and flatten:
- Number of patches: N = (H/P) × (W/P)
- Patch dimension: P × P × 3

**Step 2:** Linear projection to embedding space:
```
Z₀ = [CLS; E_patch(x)]  ∈ ℝ^(B×(N+1)×D)
```

Where:
- CLS: Learnable class token
- E_patch: Linear projection layer
- D: Embedding dimension

### 1.2 Positional Encoding

```
Z₀_pos = Z₀ + P_embed  ∈ ℝ^(B×(N+1)×D)
```

- P_embed: Learnable positional embeddings (1×(N+1)×D)

### 1.3 Multi-Head Attention (MHA)

For input **X** ∈ ℝ^(B×N×D):

**Linear projections:**
```
Q = X W_Q  ∈ ℝ^(B×H×N×d_k)
K = X W_K  ∈ ℝ^(B×H×N×d_k)
V = X W_V  ∈ ℝ^(B×H×N×d_v)
```

Where H is number of heads, d_k = D/H.

**Attention with temperature:**
```
A = softmax(QK^T / (√d_k · T))  ∈ ℝ^(B×H×N×N)
```

Where T is temperature parameter (default T=1.0).

**Output:**
```
MHA(X) = Concat(A_1V_1, ..., A_HV_H) W_O
```

### 1.4 Transformer Block

**Layer Normalization + MHA:**
```
Z' = Z + MHA(LN(Z))
```

**Layer Normalization + MLP:**
```
Z'' = Z' + MLP(LN(Z'))
```

MLP structure:
```
MLP(x) = GELU(x W_1) W_2
```

### 1.5 Multi-Scale Feature Extraction

Extract features at L layers:
```
F⁽ˡ⁾  ∈ ℝ^(B×(N+1)×D)    for l ∈ {l₁, l₂, l₃, l₄}
```

Typical extraction layers:
```
{l₁, l₂, l₃, l₄} = {L/4 - 1, L/2 - 1, 3L/4 - 1, L - 1}
```

## 2. Token Operations

### 2.1 Token Merging (Weighted)

**Attention mass computation:**
```
m_i = Σ_j A_{i,j}  (mass of token i)
```

Average across heads and queries:
```
w_i = Attention_mass(token_i) / Σ_j Attention_mass(token_j)
```

**Merging formula:**
```
z_merged = (w_i z_i + w_j z_j) / (w_i + w_j)
```

**Characteristics:**
- Differentiable
- Preserves attention information
- Reduces sequence length
- Maintains gradient flow

### 2.2 Token Pruning (Differentiable)

**Soft masking with sigmoid:**
```
m_i = sigmoid(s_i)  ∈ [0, 1]
z'_i = m_i · z_i
```

Where s_i are learnable pruning scores.

**During training:** Soft masking preserves gradients
**During inference:** Can use hard thresholding for efficiency

## 3. Feature Representation Conversion

### 3.1 Tokens to Spatial Features

Given token representation: **T** ∈ ℝ^(B×(N+1)×D)

Remove CLS token and reshape:
```
T_patches = T[:, 1:, :]  ∈ ℝ^(B×N×D)
```

Reshape to spatial:
```
F = reshape(T_patches, (B, H/P, W/P, D))
F = permute(F, (0, 3, 1, 2))  ∈ ℝ^(B×D×H/P×W/P)
```

### 3.2 Spatial Features to Tokens

Given spatial features: **F** ∈ ℝ^(B×D×H×W)

Flatten:
```
T = reshape(permute(F, (0, 2, 3, 1)), (B, H·W, D))
```

Add CLS token:
```
T_with_cls = [cls_token; T]  ∈ ℝ^(B×(HW+1)×D)
```

## 4. Feature Alignment

### 4.1 Channel Dimension Alignment

**Projection to common dimension:**
```
F_s' = φ_s(F_s) = Conv1×1(F_s)  ∈ ℝ^(B×D_common×H×W)
F_t' = φ_t(F_t) = Conv1×1(F_t)  ∈ ℝ^(B×D_common×H×W)
```

### 4.2 Spatial Dimension Alignment

Bilinear interpolation:
```
F_aligned = Interpolate(F, size=(H_target, W_target), mode='bilinear')
```

### 4.3 Feature Normalization

Per-sample normalization (L2):
```
F_normalized = F / ||F||_2
            = F / √(Σ_c Σ_h Σ_w F²_{c,h,w})
```

Where ||·||_2 is computed across all elements.

## 5. Depth Decoder

### 5.1 Decoder Block

Input: F_{i+1} ∈ ℝ^(B×C×H×W)
Skip connection: F_i ∈ ℝ^(B×C×2H×2W)

**Upsampling:**
```
F_{i+1}^{up} = Bilinear_Upsample(F_{i+1})  ∈ ℝ^(B×C×2H×2W)
```

**Fusion:**
```
F_concat = [F_i; F_{i+1}^{up}]  ∈ ℝ^(B×2C×2H×2W)
F_fused = DepthwiseConv(F_concat)  ∈ ℝ^(B×C×2H×2W)
```

### 5.2 Output Head

Final depth prediction:
```
D_raw = Conv1×1(F_final)  ∈ ℝ^(B×1×H×W)
D_pred = Softplus(D_raw)  ∈ ℝ^(B×1×H×W)
```

Softplus ensures positive depth: σ(x) = log(1 + e^x)

## 6. Loss Functions

### 6.1 Depth Supervision Loss (L_depth)

**L1 loss:**
```
L_L1 = (1/n) Σ_i |D_pred_i - D_gt_i|
```

### 6.2 Scale-Invariant Log Loss (L_silog)

**Invariant to unknown scale:**
```
L_silog = (1/n) Σ_i (log(D_pred_i) - log(D_gt_i))²
        - (1/n²) (Σ_i (log(D_pred_i) - log(D_gt_i)))²
```

**Advantages:**
- Handles unknown depth scale
- Suitable for monocular depth
- Mathematically well-behaved

### 6.3 Feature Distillation Loss (L_feat)

**Normalized MSE:**
```
L_feat = (1/N) Σ_i ||F_s_normalized_i - F_t_normalized_i||²

Where:
F_normalized = F / (||F||_2 + ε)
```

**Benefits:**
- Robust to feature magnitude differences
- Works across architectures
- Stable gradients

### 6.4 Attention Distillation Loss (L_attn)

**KL Divergence between attention distributions:**
```
L_attn = (1/L) Σ_{l=1}^L KL(A_t^l || A_s^l)

KL(P||Q) = Σ_i P_i log(P_i / Q_i)

Where:
A = softmax(QK^T / (√d_k · T))
```

**Temperature scaling:**
- Higher temperature: softer distributions
- Better distillation signals
- Typical range: T ∈ [1, 10]

### 6.5 Depth Distillation Loss (L_KD_depth)

**Scale-aware loss:**
```
L_KD_depth = (1/n) Σ_i (log(D_s_i) - log(D_t_i))²
```

**Advantages:**
- Scale-invariant like L_silog
- Matches log predictions
- Suitable for KD setting

### 6.6 Relational Distillation Loss (L_rel)

**Match pairwise relations:**
```
R_ij = ||F_i - F_j||_2^2

L_rel = (1/N²) Σ_i Σ_j (R_s_ij - R_t_ij)²
```

**Benefits:**
- Capture relative structure
- Handle scale differences
- More stable than absolute values

### 6.7 Positional Encoding Loss (L_pos)

**Match positional embeddings:**
```
L_pos = ||P_s - P_t||_F^2

Where ||·||_F is Frobenius norm
```

## 7. Total Loss Function

**Weighted combination:**
```
L_total = λ₁ L_depth 
        + λ₂ L_silog 
        + λ₃ L_feat 
        + λ₄ L_attn 
        + λ₅ L_KD_depth 
        + λ₆ L_rel 
        + λ₇ L_dec 
        + λ₈ L_pos
```

**Default weights:**
```
λ₁ = 1.0   (depth GT)
λ₂ = 0.1   (scale-invariant log)
λ₃ = 0.5   (feature distillation)
λ₄ = 0.1   (attention distillation)
λ₅ = 0.5   (depth distillation)
λ₆ = 0.1   (relational)
λ₇ = 0.3   (decoder)
λ₈ = 0.05  (positional)
```

## 8. Numerical Stability

### 8.1 Log Operations

```
log(x) ≈ log(x + ε)  where ε = 1e-6
```

### 8.2 Division Operations

```
x / y ≈ x / (y + ε)
```

### 8.3 Softmax Operations

```
softmax(x) = exp(x - max(x)) / Σ_i exp(x_i - max(x))
```

Prevents overflow/underflow.

### 8.4 Layer Normalization

```
LN(x) = (x - mean(x)) / √(var(x) + ε)
```

Where ε = 1e-6 for numerical stability.

## 9. Optimization

### 9.1 Mixed Precision Training

**Forward pass:** FP16 (half precision)
**Loss computation:** FP32 (full precision)
**Backward pass:** FP32
**Gradient updates:** Scaled and unscaled appropriately

### 9.2 Gradient Clipping

```
g_clipped = g / max(1, ||g|| / max_norm)
```

Where max_norm = 1.0 (prevents explosion).

### 9.3 Learning Rate Schedule

**Cosine annealing:**
```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt/T))
```

Where T = total epochs.

## 10. Evaluation Metrics

### 10.1 RMSE

```
RMSE = √((1/n) Σ_i (D_pred_i - D_gt_i)²)
```

### 10.2 Absolute Relative Error

```
AbsRel = (1/n) Σ_i |D_pred_i - D_gt_i| / D_gt_i
```

### 10.3 Accuracy at Threshold

```
δₖ = (1/n) Σ_i [max(D_pred_i/D_gt_i, D_gt_i/D_pred_i) < 1.25^k]
```

Typically k ∈ {1, 2, 3}.

## References

1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
2. Hinton et al. "Distilling the Knowledge in a Neural Network"
3. Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
4. Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"

---

**Last Updated:** April 2026
