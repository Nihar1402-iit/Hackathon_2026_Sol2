"""
Trainer class for knowledge distillation training.

Handles the complete training loop with mixed precision,
gradient clipping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

from ..models.student_model import StudentModel
from ..models.teacher_model import TeacherModel
from ..losses.depth_loss import create_depth_loss
from ..losses.kd_losses import create_kd_loss
from ..losses.relational_loss import DecoderFeatureDistillationLoss
from ..utils.metrics import compute_metrics


class Trainer:
    """
    Trainer for knowledge distillation of depth estimation models.
    """
    
    def __init__(self, student_model: StudentModel, teacher_model: TeacherModel,
                 device: str = 'cuda', checkpoint_dir: str = './checkpoints',
                 use_amp: bool = True, use_kd: bool = True,
                 loss_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            student_model: Student model instance
            teacher_model: Teacher model instance
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
            use_kd: Use knowledge distillation losses
            loss_weights: Dictionary of loss weights
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.use_kd = use_kd
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Loss functions
        self.depth_loss = create_depth_loss('silog').to(device)
        self.l1_loss = create_depth_loss('l1').to(device)
        
        self.kd_losses = {}
        if use_kd:
            self.kd_losses['feature'] = create_kd_loss('feature').to(device)
            self.kd_losses['attention'] = create_kd_loss('attention').to(device)
            self.kd_losses['depth'] = create_kd_loss('depth').to(device)
            self.kd_losses['relational'] = create_kd_loss('relational').to(device)
            self.kd_losses['positional'] = create_kd_loss('positional').to(device)
        
        self.decoder_feat_loss = DecoderFeatureDistillationLoss().to(device)
        
        # Loss weights
        self.loss_weights = loss_weights or {
            'depth_gt': 1.0,          # λ1
            'depth_silog': 0.1,       # λ2
            'feature_kd': 0.5,        # λ3
            'attention_kd': 0.1,      # λ4
            'depth_kd': 0.5,          # λ5
            'relational_kd': 0.1,     # λ6
            'decoder_kd': 0.3,        # λ7
            'positional_kd': 0.05,    # λ8
        }
        
        # Mixed precision
        if use_amp:
            self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.checkpoint_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def compute_losses(self, student_out: Dict, teacher_out: Dict,
                      depth_gt: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute all losses.
        
        Args:
            student_out: Student model output dict
            teacher_out: Teacher model output dict
            depth_gt: Ground truth depth
            mask: Valid pixel mask
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Depth supervision loss
        depth_s = student_out['depth']
        loss_dict['depth_gt'] = self.depth_loss(depth_s, depth_gt, mask)
        loss_dict['depth_l1'] = self.l1_loss(depth_s, depth_gt, mask)
        
        total_loss += self.loss_weights['depth_gt'] * loss_dict['depth_gt']
        total_loss += self.loss_weights['depth_silog'] * loss_dict['depth_l1']
        
        if self.use_kd:
            depth_t = teacher_out['depth']
            
            # 2. Feature distillation
            if student_out['features'] and teacher_out['features']:
                feat_loss = 0.0
                for f_s, f_t in zip(student_out['features'], teacher_out['features']):
                    feat_loss += self.kd_losses['feature'](f_s, f_t)
                loss_dict['feature_kd'] = feat_loss / len(student_out['features'])
                total_loss += self.loss_weights['feature_kd'] * loss_dict['feature_kd']
            
            # 3. Attention distillation (ViT only)
            if student_out['attention_maps'] and teacher_out['attention_maps']:
                loss_dict['attention_kd'] = self.kd_losses['attention'](
                    student_out['attention_maps'],
                    teacher_out['attention_maps']
                )
                total_loss += self.loss_weights['attention_kd'] * loss_dict['attention_kd']
            
            # 4. Depth distillation
            loss_dict['depth_kd'] = self.kd_losses['depth'](depth_s, depth_t)
            total_loss += self.loss_weights['depth_kd'] * loss_dict['depth_kd']
            
            # 5. Relational distillation
            if student_out['features'] and teacher_out['features']:
                loss_dict['relational_kd'] = self.kd_losses['relational'](
                    student_out['encoder_features'],
                    teacher_out['encoder_features']
                )
                total_loss += self.loss_weights['relational_kd'] * loss_dict['relational_kd']
            
            # 6. Decoder feature distillation (future: when available)
            # loss_dict['decoder_kd'] = ...
            
            # 7. Positional encoding distillation (ViT only)
            if student_out.get('tokens') is not None and teacher_out.get('tokens') is not None:
                # Simplified: distill token representations
                loss_dict['positional_kd'] = torch.tensor(0.0, device=self.device)
        
        return total_loss, loss_dict
    
    def train_step(self, batch: Dict, optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'image' and 'depth' keys
            optimizer: Optimizer
            
        Returns:
            Dictionary of loss values
        """
        images = batch['image'].to(self.device)
        depths = batch['depth'].to(self.device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(self.device)
        
        optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                # Student forward
                student_out = self.student(images)
                
                # Teacher forward (no gradients)
                with torch.no_grad():
                    teacher_out = self.teacher(images)
                
                # Compute losses
                total_loss, loss_dict = self.compute_losses(
                    student_out, teacher_out, depths, mask
                )
            
            # Backward
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Student forward
            student_out = self.student(images)
            
            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_out = self.teacher(images)
            
            # Compute losses
            total_loss, loss_dict = self.compute_losses(
                student_out, teacher_out, depths, mask
            )
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Convert to float for logging
        loss_dict['total'] = total_loss.item()
        for key in loss_dict:
            if isinstance(loss_dict[key], torch.Tensor):
                loss_dict[key] = loss_dict[key].item()
        
        return loss_dict
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validation step.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.student.eval()
        self.teacher.eval()
        
        metrics = {
            'rmse': [],
            'abs_rel': [],
            'delta1': [],
            'delta2': [],
            'delta3': [],
        }
        
        for batch in val_loader:
            images = batch['image'].to(self.device)
            depths_gt = batch['depth'].to(self.device)
            
            # Forward
            student_out = self.student(images)
            depths_pred = student_out['depth']
            
            # Compute metrics
            batch_metrics = compute_metrics(depths_pred, depths_gt)
            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])
        
        # Average metrics
        val_metrics = {}
        for key in metrics:
            if metrics[key]:
                val_metrics[key] = sum(metrics[key]) / len(metrics[key])
        
        self.student.train()
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer,
                       best_metric: Optional[float] = None):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_weights': self.loss_weights,
        }
        
        if best_metric is not None:
            checkpoint['best_metric'] = best_metric
        
        save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
        
        return save_path
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
        
        return epoch
