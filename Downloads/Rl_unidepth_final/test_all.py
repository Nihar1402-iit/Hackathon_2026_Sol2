#!/usr/bin/env python
"""
Comprehensive test suite and demo script.

Tests all major components and verifies correctness.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unidepth_kd.models import (
    create_vit_encoder, create_cnn_encoder,
    create_student_model, create_teacher_model
)
from unidepth_kd.losses import create_depth_loss, create_kd_loss
from unidepth_kd.data import create_dataset
from unidepth_kd.utils import (
    compute_metrics, count_parameters, visualize_depth
)


def test_vit_encoder():
    """Test Vision Transformer encoder."""
    print("\n" + "="*60)
    print("Testing ViT Encoder")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vit = create_vit_encoder(model_size='tiny', img_size=384, patch_size=16)
    vit = vit.to(device)
    
    # Forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 384, 384).to(device)
    output = vit(x)
    
    # Verify output
    assert 'tokens' in output, "Missing 'tokens' in output"
    assert 'features' in output, "Missing 'features' in output"
    assert 'attention_maps' in output, "Missing 'attention_maps' in output"
    
    tokens = output['tokens']
    features = output['features']
    attention_maps = output['attention_maps']
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Token shape: {tokens.shape}")
    print(f"✓ Features count: {len(features)}")
    print(f"✓ Feature shapes: {[f.shape for f in features]}")
    print(f"✓ Attention maps count: {len(attention_maps)}")
    print(f"✓ Attention map shapes: {[a.shape for a in attention_maps]}")
    
    total_params, trainable_params = count_parameters(vit)
    print(f"✓ Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    return True


def test_cnn_encoder():
    """Test CNN encoder."""
    print("\n" + "="*60)
    print("Testing CNN Encoder")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cnn = create_cnn_encoder(model_size='resnet18')
    cnn = cnn.to(device)
    
    # Forward pass
    x = torch.randn(2, 3, 384, 384).to(device)
    output = cnn(x)
    
    # Verify output
    assert 'features' in output, "Missing 'features' in output"
    features = output['features']
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Features count: {len(features)}")
    print(f"✓ Feature shapes: {[f.shape for f in features]}")
    
    total_params, trainable_params = count_parameters(cnn)
    print(f"✓ Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    return True


def test_student_model():
    """Test student model."""
    print("\n" + "="*60)
    print("Testing Student Model (ViT)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'backbone_type': 'vit_tiny',
        'img_size': 384,
        'patch_size': 16,
        'use_token_merging': True,
        'merge_ratio': 0.3,
        'use_token_pruning': False,
        'decoder_type': 'lightweight',
        'base_channels': 128,
    }
    
    student = create_student_model(config).to(device)
    
    # Forward pass
    x = torch.randn(2, 3, 384, 384).to(device)
    output = student(x)
    
    # Verify output
    assert 'depth' in output, "Missing 'depth' in output"
    assert 'features' in output, "Missing 'features' in output"
    
    depth = output['depth']
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Depth shape: {depth.shape}")
    print(f"✓ Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    
    total_params, trainable_params = count_parameters(student)
    print(f"✓ Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    return True


def test_teacher_model():
    """Test teacher model."""
    print("\n" + "="*60)
    print("Testing Teacher Model (ViT)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'backbone_type': 'vit_base',
        'img_size': 384,
        'patch_size': 16,
        'decoder_type': 'hierarchical',
        'base_channels': 128,
        'freeze_backbone': True,
    }
    
    teacher = create_teacher_model(config).to(device)
    
    # Check that backbone is frozen
    for param in teacher.backbone.parameters():
        assert not param.requires_grad, "Teacher backbone should be frozen"
    
    # Forward pass
    x = torch.randn(2, 3, 384, 384).to(device)
    output = teacher(x)
    
    # Verify output
    assert 'depth' in output, "Missing 'depth' in output"
    depth = output['depth']
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Depth shape: {depth.shape}")
    print(f"✓ Backbone frozen: ✓")
    
    total_params, trainable_params = count_parameters(teacher)
    print(f"✓ Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    return True


def test_depth_losses():
    """Test depth loss functions."""
    print("\n" + "="*60)
    print("Testing Depth Losses")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    depth_pred = torch.rand(2, 1, 384, 384, device=device) * 10 + 0.1
    depth_gt = torch.rand(2, 1, 384, 384, device=device) * 10 + 0.1
    
    loss_types = ['l1', 'l2', 'silog', 'gradient', 'ssim']
    
    for loss_type in loss_types:
        try:
            loss_fn = create_depth_loss(loss_type)
            loss_fn = loss_fn.to(device)
            loss = loss_fn(depth_pred, depth_gt)
            
            assert not torch.isnan(loss), f"{loss_type} loss is NaN"
            assert not torch.isinf(loss), f"{loss_type} loss is Inf"
            assert loss.item() > 0, f"{loss_type} loss should be positive"
            
            print(f"✓ {loss_type.upper():6s} loss: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ {loss_type.upper()} loss failed: {e}")
            return False
    
    return True


def test_kd_losses():
    """Test knowledge distillation losses."""
    print("\n" + "="*60)
    print("Testing KD Losses")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature distillation
    feat_s = torch.randn(2, 128, 24, 24, device=device)
    feat_t = torch.randn(2, 128, 24, 24, device=device)
    
    loss_fn = create_kd_loss('feature')
    loss_fn = loss_fn.to(device)
    loss = loss_fn(feat_s, feat_t)
    
    assert not torch.isnan(loss), "Feature loss is NaN"
    print(f"✓ Feature distillation loss: {loss.item():.6f}")
    
    # Depth distillation
    depth_s = torch.randn(2, 1, 384, 384, device=device) + 5
    depth_t = torch.randn(2, 1, 384, 384, device=device) + 5
    
    loss_fn = create_kd_loss('depth')
    loss_fn = loss_fn.to(device)
    loss = loss_fn(depth_s, depth_t)
    
    assert not torch.isnan(loss), "Depth KD loss is NaN"
    print(f"✓ Depth distillation loss: {loss.item():.6f}")
    
    # Relational distillation
    loss_fn = create_kd_loss('relational')
    loss_fn = loss_fn.to(device)
    loss = loss_fn(feat_s, feat_t)
    
    assert not torch.isnan(loss), "Relational loss is NaN"
    print(f"✓ Relational distillation loss: {loss.item():.6f}")
    
    return True


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("Testing Metrics")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    depth_pred = torch.rand(1, 1, 384, 384, device=device) * 10 + 0.1
    depth_gt = torch.rand(1, 1, 384, 384, device=device) * 10 + 0.1
    
    metrics = compute_metrics(depth_pred, depth_gt)
    
    for key, val in metrics.items():
        assert val >= 0, f"Metric {key} should be non-negative"
        print(f"✓ {key:10s}: {val:.6f}")
    
    return True


def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)
    
    dataset = create_dataset(dataset_type='mock', n_samples=10, img_size=384)
    
    assert len(dataset) == 10, "Dataset size mismatch"
    print(f"✓ Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    assert 'image' in sample, "Missing 'image' in sample"
    assert 'depth' in sample, "Missing 'depth' in sample"
    
    image = sample['image']
    depth = sample['depth']
    
    assert image.shape == (3, 384, 384), f"Wrong image shape: {image.shape}"
    assert depth.shape == (1, 384, 384), f"Wrong depth shape: {depth.shape}"
    
    print(f"✓ Image shape: {image.shape}")
    print(f"✓ Depth shape: {depth.shape}")
    
    return True


def test_visualization():
    """Test visualization utilities."""
    print("\n" + "="*60)
    print("Testing Visualization")
    print("="*60)
    
    depth = torch.rand(1, 384, 384) * 10
    
    try:
        rgb = visualize_depth(depth, cmap='turbo')
        assert rgb.shape == (384, 384, 3), f"Wrong visualization shape: {rgb.shape}"
        print(f"✓ Depth visualization shape: {rgb.shape}")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False
    
    return True


def test_end_to_end():
    """Test end-to-end training step."""
    print("\n" + "="*60)
    print("Testing End-to-End Training Step")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    student = create_student_model({
        'backbone_type': 'vit_tiny',
        'img_size': 384,
        'decoder_type': 'lightweight',
    }).to(device)
    
    teacher = create_teacher_model({
        'backbone_type': 'vit_small',
        'img_size': 384,
        'freeze_backbone': True,
    }).to(device)
    
    # Create losses
    depth_loss_fn = create_depth_loss('silog').to(device)
    feat_loss_fn = create_kd_loss('feature').to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    # Create batch
    batch = {
        'image': torch.randn(2, 3, 384, 384).to(device),
        'depth': torch.rand(2, 1, 384, 384).to(device) * 10 + 0.1,
    }
    
    # Forward pass
    student_out = student(batch['image'])
    with torch.no_grad():
        teacher_out = teacher(batch['image'])
    
    # Compute losses
    loss_depth = depth_loss_fn(student_out['depth'], batch['depth'])
    loss_feat = feat_loss_fn(
        student_out['features'][0],
        teacher_out['features'][0]
    )
    
    total_loss = loss_depth + 0.5 * loss_feat
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"✓ Depth loss: {loss_depth.item():.6f}")
    print(f"✓ Feature loss: {loss_feat.item():.6f}")
    print(f"✓ Total loss: {total_loss.item():.6f}")
    print(f"✓ Training step completed successfully")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("UNIDEPTH-KD COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    tests = [
        ("ViT Encoder", test_vit_encoder),
        ("CNN Encoder", test_cnn_encoder),
        ("Student Model", test_student_model),
        ("Teacher Model", test_teacher_model),
        ("Depth Losses", test_depth_losses),
        ("KD Losses", test_kd_losses),
        ("Metrics", test_metrics),
        ("Dataset", test_dataset),
        ("Visualization", test_visualization),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:7s} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
