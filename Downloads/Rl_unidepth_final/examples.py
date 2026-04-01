"""
Example usage and tutorials for UniDepth-KD.

This file demonstrates various ways to use the library.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / 'unidepth_kd'))

from models import create_vit_encoder, create_student_model, create_teacher_model
from losses import create_depth_loss, create_kd_loss
from data import create_dataset
from utils import compute_metrics, count_parameters


# ============================================================================
# Example 1: Create and inspect models
# ============================================================================
def example_1_model_creation():
    """Example 1: Creating and inspecting models."""
    print("\n" + "="*70)
    print("Example 1: Model Creation and Inspection")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ViT encoder
    vit = create_vit_encoder(
        model_size='tiny',
        img_size=384,
        patch_size=16,
        temperature=1.0
    )
    
    # Count parameters
    total, trainable = count_parameters(vit)
    print(f"\nViT Tiny:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    # Test forward pass
    vit = vit.to(device)
    x = torch.randn(2, 3, 384, 384).to(device)
    output = vit(x)
    
    print(f"\nForward pass output:")
    print(f"  Tokens shape: {output['tokens'].shape}")
    print(f"  Number of feature levels: {len(output['features'])}")
    print(f"  Feature shapes: {[f.shape for f in output['features']]}")
    print(f"  Attention maps: {len(output['attention_maps'])} levels")


# ============================================================================
# Example 2: Student and Teacher models with KD
# ============================================================================
def example_2_knowledge_distillation_setup():
    """Example 2: Setting up student-teacher KD framework."""
    print("\n" + "="*70)
    print("Example 2: Knowledge Distillation Setup")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create lightweight student
    student_config = {
        'backbone_type': 'vit_tiny',
        'img_size': 384,
        'use_token_merging': True,
        'merge_ratio': 0.3,
        'decoder_type': 'lightweight',
    }
    student = create_student_model(student_config).to(device)
    
    # Create powerful teacher
    teacher_config = {
        'backbone_type': 'vit_base',
        'img_size': 384,
        'decoder_type': 'hierarchical',
        'freeze_backbone': True,  # Frozen during training
    }
    teacher = create_teacher_model(teacher_config).to(device)
    
    # Print parameter comparison
    s_total, s_train = count_parameters(student)
    t_total, t_train = count_parameters(teacher)
    
    print(f"\nStudent Model (ViT Tiny):")
    print(f"  Parameters: {s_total:,}")
    print(f"  Trainable: {s_train:,}")
    
    print(f"\nTeacher Model (ViT Base):")
    print(f"  Parameters: {t_total:,}")
    print(f"  Trainable: {t_train:,}")
    
    print(f"\nSize reduction: {(1 - s_total/t_total) * 100:.1f}%")


# ============================================================================
# Example 3: Forward pass and loss computation
# ============================================================================
def example_3_forward_and_losses():
    """Example 3: Forward pass and KD loss computation."""
    print("\n" + "="*70)
    print("Example 3: Forward Pass and Loss Computation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    student = create_student_model({
        'backbone_type': 'vit_tiny',
        'img_size': 384,
        'decoder_type': 'lightweight',
    }).to(device)
    
    teacher = create_teacher_model({
        'backbone_type': 'vit_small',
        'freeze_backbone': True,
    }).to(device)
    
    # Create losses
    depth_loss_fn = create_depth_loss('silog').to(device)
    feat_loss_fn = create_kd_loss('feature').to(device)
    depth_kd_loss_fn = create_kd_loss('depth').to(device)
    
    # Create batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 384, 384).to(device)
    depth_gt = torch.rand(batch_size, 1, 384, 384).to(device) * 10 + 0.1
    
    # Student forward
    student_out = student(images)
    
    # Teacher forward (no gradients)
    with torch.no_grad():
        teacher_out = teacher(images)
    
    # Compute losses
    l_depth = depth_loss_fn(student_out['depth'], depth_gt)
    l_feat = feat_loss_fn(student_out['features'][0], teacher_out['features'][0])
    l_depth_kd = depth_kd_loss_fn(student_out['depth'], teacher_out['depth'])
    
    # Total loss
    total_loss = l_depth + 0.5 * l_feat + 0.5 * l_depth_kd
    
    print(f"\nLoss values:")
    print(f"  Depth GT loss: {l_depth.item():.6f}")
    print(f"  Feature KD loss: {l_feat.item():.6f}")
    print(f"  Depth KD loss: {l_depth_kd.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")


# ============================================================================
# Example 4: Training step with gradient computation
# ============================================================================
def example_4_training_step():
    """Example 4: Single training step with backprop."""
    print("\n" + "="*70)
    print("Example 4: Training Step with Backpropagation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Models
    student = create_student_model({
        'backbone_type': 'vit_tiny',
        'decoder_type': 'lightweight',
    }).to(device)
    teacher = create_teacher_model({
        'backbone_type': 'vit_small',
        'freeze_backbone': True,
    }).to(device)
    
    # Losses
    depth_loss_fn = create_depth_loss('silog').to(device)
    feat_loss_fn = create_kd_loss('feature').to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    # Batch
    images = torch.randn(2, 3, 384, 384).to(device)
    depth_gt = torch.rand(2, 1, 384, 384).to(device) * 10 + 0.1
    
    # Forward
    student_out = student(images)
    with torch.no_grad():
        teacher_out = teacher(images)
    
    # Losses
    l_depth = depth_loss_fn(student_out['depth'], depth_gt)
    l_feat = feat_loss_fn(student_out['features'][0], teacher_out['features'][0])
    total_loss = l_depth + 0.5 * l_feat
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient stats
    total_norm = 0.0
    for p in student.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Update
    optimizer.step()
    
    print(f"\nTraining step:")
    print(f"  Loss: {total_loss.item():.6f}")
    print(f"  Gradient norm: {total_norm:.6f}")
    print(f"  Updated {sum(1 for p in student.parameters() if p.requires_grad)} parameters")


# ============================================================================
# Example 5: Model with token operations
# ============================================================================
def example_5_token_operations():
    """Example 5: Student model with token merging."""
    print("\n" + "="*70)
    print("Example 5: Token Operations (Merging)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Student with token merging
    student = create_student_model({
        'backbone_type': 'vit_tiny',
        'use_token_merging': True,
        'merge_ratio': 0.4,  # Merge 40% of tokens
        'decoder_type': 'lightweight',
    }).to(device)
    
    # Forward pass
    images = torch.randn(2, 3, 384, 384).to(device)
    output = student(images)
    
    print(f"\nWith token merging (40% merge ratio):")
    print(f"  Input shape: {images.shape}")
    print(f"  Output depth shape: {output['depth'].shape}")
    print(f"  Tokens after merging: {output['tokens'].shape}")
    
    # Comparison without merging
    student_no_merge = create_student_model({
        'backbone_type': 'vit_tiny',
        'use_token_merging': False,
        'decoder_type': 'lightweight',
    }).to(device)
    
    with torch.no_grad():
        output_no_merge = student_no_merge(images)
    
    print(f"\nWithout token merging:")
    print(f"  Tokens shape: {output_no_merge['tokens'].shape}")


# ============================================================================
# Example 6: Cross-architecture KD (ViT -> CNN)
# ============================================================================
def example_6_cross_architecture_kd():
    """Example 6: ViT teacher distilling to CNN student."""
    print("\n" + "="*70)
    print("Example 6: Cross-Architecture KD (ViT → CNN)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CNN student
    student = create_student_model({
        'backbone_type': 'resnet18',  # CNN backbone
        'img_size': 384,
        'decoder_type': 'hierarchical',
    }).to(device)
    
    # ViT teacher
    teacher = create_teacher_model({
        'backbone_type': 'vit_base',
        'freeze_backbone': True,
    }).to(device)
    
    s_total, s_train = count_parameters(student)
    t_total, t_train = count_parameters(teacher)
    
    print(f"\nCross-Architecture KD Setup:")
    print(f"  Student: ResNet18 ({s_total:,} params)")
    print(f"  Teacher: ViT Base ({t_total:,} params)")
    print(f"  Student/Teacher ratio: {s_total/t_total:.2%}")
    
    # Feature adaptation is handled automatically
    images = torch.randn(2, 3, 384, 384).to(device)
    
    student_out = student(images)
    with torch.no_grad():
        teacher_out = teacher(images)
    
    print(f"\nFeature shapes:")
    print(f"  Student features: {[f.shape for f in student_out['features']]}")
    print(f"  Teacher features: {[f.shape for f in teacher_out['features']]}")


# ============================================================================
# Example 7: Evaluation and metrics
# ============================================================================
def example_7_metrics_evaluation():
    """Example 7: Computing evaluation metrics."""
    print("\n" + "="*70)
    print("Example 7: Metrics and Evaluation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create student model
    student = create_student_model({
        'backbone_type': 'vit_tiny',
        'decoder_type': 'lightweight',
    }).to(device)
    
    # Generate test data
    with torch.no_grad():
        images = torch.randn(10, 3, 384, 384).to(device)
        depth_gt_batch = []
        depth_pred_batch = []
        
        for _ in range(10):
            out = student(torch.randn(1, 3, 384, 384).to(device))
            depth_pred_batch.append(out['depth'].cpu())
            depth_gt = torch.rand(1, 1, 384, 384) * 10 + 0.1
            depth_gt_batch.append(depth_gt)
    
    # Compute metrics for each sample
    print(f"\nMetrics across {len(depth_pred_batch)} samples:")
    all_metrics = {}
    
    for i, (pred, gt) in enumerate(zip(depth_pred_batch, depth_gt_batch)):
        metrics = compute_metrics(pred, gt)
        
        for key, val in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(val)
    
    # Print averaged metrics
    print(f"\nAveraged metrics:")
    for key in ['rmse', 'abs_rel', 'delta1', 'delta2', 'delta3']:
        if key in all_metrics:
            avg = sum(all_metrics[key]) / len(all_metrics[key])
            print(f"  {key:10s}: {avg:.6f}")


# ============================================================================
# Example 8: Dataset and DataLoader
# ============================================================================
def example_8_dataset_loading():
    """Example 8: Working with datasets."""
    print("\n" + "="*70)
    print("Example 8: Dataset and DataLoader")
    print("="*70)
    
    # Create mock dataset
    dataset = create_dataset(
        dataset_type='mock',
        split='train',
        img_size=384,
        n_samples=32
    )
    
    print(f"\nMock dataset:")
    print(f"  Number of samples: {len(dataset)}")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Depths: {batch['depth'].shape}")
    if 'mask' in batch:
        print(f"  Masks: {batch['mask'].shape}")
    
    print(f"\nData ranges:")
    print(f"  Images: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"  Depths: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("UniDepth-KD Examples and Tutorials")
    print("="*70)
    
    examples = [
        ("Model Creation", example_1_model_creation),
        ("KD Setup", example_2_knowledge_distillation_setup),
        ("Forward & Losses", example_3_forward_and_losses),
        ("Training Step", example_4_training_step),
        ("Token Merging", example_5_token_operations),
        ("Cross-Arch KD", example_6_cross_architecture_kd),
        ("Metrics", example_7_metrics_evaluation),
        ("Dataset", example_8_dataset_loading),
    ]
    
    for i, (name, example_fn) in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")
