#!/usr/bin/env python3
"""
Training script for NYU Depth V2 -> KITTI workflow.

Trains on NYU Depth V2 and evaluates on KITTI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
from datetime import datetime
import numpy as np

from unidepth_kd.models import create_student_model, create_teacher_model
from unidepth_kd.data.dataset import create_dataset, NYUDepthV2Dataset, KITTIDataset
from unidepth_kd.data.transforms import get_train_transform, get_val_transform
from unidepth_kd.losses.depth_loss import ScaleInvariantLogLoss
from unidepth_kd.losses.kd_losses import FeatureDistillationLoss, AttentionDistillationLoss
from unidepth_kd.utils.metrics import compute_metrics
from unidepth_kd.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NYUKITTITrainer:
    """Trainer for NYU -> KITTI workflow."""
    
    def __init__(self, config_path: str, output_dir: str = './results'):
        """Initialize trainer with config."""
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create datasets
        self._setup_datasets()
        
        # Create models
        self._setup_models()
        
        # Create trainer
        self._setup_trainer()
    
    def _setup_datasets(self):
        """Setup NYU Depth V2 training and KITTI validation/test."""
        logger.info("Setting up datasets...")
        
        data_config = self.config.get('data', {})
        img_size = data_config.get('img_size', 384)
        
        # NYU Depth V2 for training
        nyu_dir = data_config.get('nyu_dir', './data/nyu_depth_v2')
        self.train_dataset = NYUDepthV2Dataset(
            nyu_dir,
            split='train',
            img_size=img_size,
            augment=True
        )
        
        self.val_dataset = NYUDepthV2Dataset(
            nyu_dir,
            split='val',
            img_size=img_size,
            augment=False
        )
        
        # KITTI for testing
        kitti_dir = data_config.get('kitti_dir', './data/kitti')
        self.test_dataset = KITTIDataset(
            kitti_dir,
            split='val',  # KITTI uses 'val' for test set
            img_size=img_size,
            augment=False
        )
        
        # Create data loaders
        train_batch_size = data_config.get('train_batch_size', 8)
        val_batch_size = data_config.get('val_batch_size', 16)
        num_workers = data_config.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info(f"✓ Train: {len(self.train_dataset)} samples (NYU)")
        logger.info(f"✓ Val: {len(self.val_dataset)} samples (NYU)")
        logger.info(f"✓ Test: {len(self.test_dataset)} samples (KITTI)")
    
    def _setup_models(self):
        """Setup student and teacher models."""
        logger.info("Creating models...")
        
        student_config = self.config.get('student_model', {})
        teacher_config = self.config.get('teacher_model', {})
        
        self.student = create_student_model(student_config).to(self.device)
        self.teacher = create_teacher_model(teacher_config).to(self.device)
        
        # Count parameters
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        
        logger.info(f"✓ Student model: {student_params:,} parameters")
        logger.info(f"✓ Teacher model: {teacher_params:,} parameters")
    
    def _setup_trainer(self):
        """Setup trainer with optimizer and losses."""
        logger.info("Setting up training...")
        
        training_config = self.config.get('training', {})
        
        # Optimizer
        lr = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 1e-4)
        
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        epochs = training_config.get('epochs', 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Loss functions
        self.depth_loss = ScaleInvariantLogLoss()
        self.feat_loss = FeatureDistillationLoss()
        self.attn_loss = AttentionDistillationLoss()
        
        logger.info(f"✓ Optimizer: AdamW (lr={lr})")
        logger.info(f"✓ Scheduler: Cosine annealing ({epochs} epochs)")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                teacher_out = self.teacher(images)
            
            student_out = self.student(images)
            
            # Compute losses
            # Depth loss
            depth_pred = student_out['depth']
            loss_depth = self.depth_loss(depth_pred, depths, masks)
            
            # Feature distillation
            loss_feat = 0
            if 'features' in student_out and 'features' in teacher_out:
                for s_feat, t_feat in zip(student_out['features'], teacher_out['features']):
                    loss_feat += self.feat_loss(s_feat, t_feat)
                loss_feat = loss_feat / max(len(student_out['features']), 1)
            
            # Attention distillation
            loss_attn = 0
            if student_out['attention_maps'] is not None and teacher_out['attention_maps'] is not None:
                for s_attn, t_attn in zip(student_out['attention_maps'], teacher_out['attention_maps']):
                    loss_attn += self.attn_loss(s_attn, t_attn)
                loss_attn = loss_attn / max(len(student_out['attention_maps']), 1)
            
            # Total loss
            loss = loss_depth + 0.1 * loss_feat + 0.05 * loss_attn
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader, dataset_name: str = 'NYU'):
        """Validate on a dataset."""
        self.student.eval()
        
        all_metrics = {}
        pbar = tqdm(loader, desc=f'Validating on {dataset_name}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Predict
            outputs = self.student(images)
            depth_pred = outputs['depth']
            
            # Compute metrics
            metrics = compute_metrics(depth_pred, depths, masks)
            
            for key, val in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val.cpu().numpy() if torch.is_tensor(val) else val)
        
        # Average metrics
        for key in all_metrics:
            all_metrics[key] = float(np.mean(all_metrics[key]))
        
        logger.info(f"{dataset_name} Validation Metrics:")
        for key, val in all_metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        
        return all_metrics
    
    def train(self):
        """Run complete training pipeline."""
        logger.info("=" * 60)
        logger.info("Starting NYU -> KITTI Training Pipeline")
        logger.info("=" * 60)
        
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        best_metrics = None
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            
            # Validate on NYU
            if (epoch + 1) % training_config.get('val_interval', 5) == 0:
                nyu_metrics = self.validate(self.val_loader, 'NYU')
            
            # Test on KITTI
            if (epoch + 1) % training_config.get('test_interval', 10) == 0:
                kitti_metrics = self.validate(self.test_loader, 'KITTI')
            
            # Save checkpoint
            if (epoch + 1) % training_config.get('save_interval', 10) == 0:
                self._save_checkpoint(epoch)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student'])
        self.teacher.load_state_dict(checkpoint['teacher'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train on NYU, test on KITTI')
    parser.add_argument('--config', default='unidepth_kd/configs/config.yaml',
                       help='Config file path')
    parser.add_argument('--output', default='./results',
                       help='Output directory')
    parser.add_argument('--resume', default=None,
                       help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Create trainer
    trainer = NYUKITTITrainer(args.config, args.output)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
