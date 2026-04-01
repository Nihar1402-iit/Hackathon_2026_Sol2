"""
Main training script for knowledge distillation.

Run training with configuration from YAML.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import logging
from typing import Dict, Optional

from ..models.student_model import create_student_model
from ..models.teacher_model import create_teacher_model
from .trainer import Trainer
from ..data.dataset import create_dataset


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config: Dict, resume_from: Optional[str] = None):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        resume_from: Path to checkpoint to resume from
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    student = create_student_model(config['student_model'])
    teacher = create_teacher_model(config['teacher_model'])
    
    logging.info(f"Student model: {config['student_model']['backbone_type']}")
    logging.info(f"Teacher model: {config['teacher_model']['backbone_type']}")
    
    # Create trainer
    trainer = Trainer(
        student_model=student,
        teacher_model=teacher,
        device=str(device),
        checkpoint_dir=config['training']['checkpoint_dir'],
        use_amp=config['training'].get('use_amp', True),
        use_kd=config['training'].get('use_kd', True),
        loss_weights=config['training'].get('loss_weights', None)
    )
    
    # Create optimizers
    student_optimizer = optim.AdamW(
        student.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        student_optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Load datasets
    train_dataset = create_dataset(
        dataset_type=config['data']['dataset_type'],
        split='train',
        img_size=config['data']['img_size'],
        augment=True
    )
    
    val_dataset = create_dataset(
        dataset_type=config['data']['dataset_type'],
        split='val',
        img_size=config['data']['img_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Resume training if checkpoint provided
    start_epoch = 0
    best_metric = float('inf')
    
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from, student_optimizer)
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logging.info(f"{'='*60}")
        
        # Training phase
        student.train()
        train_losses = {}
        
        for batch_idx, batch in enumerate(train_loader):
            loss_dict = trainer.train_step(batch, student_optimizer)
            
            # Accumulate losses
            for key, val in loss_dict.items():
                if key not in train_losses:
                    train_losses[key] = 0.0
                train_losses[key] += val
            
            if (batch_idx + 1) % config['training'].get('log_interval', 10) == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in train_losses.items()}
                loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} | {loss_str}")
        
        # Average training losses
        avg_train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        
        # Validation phase
        if (epoch + 1) % config['training'].get('val_interval', 1) == 0:
            val_metrics = trainer.validate(val_loader)
            
            logging.info(f"\nValidation metrics:")
            for key, val in val_metrics.items():
                logging.info(f"  {key}: {val:.4f}")
            
            # Check if best metric improved
            current_metric = val_metrics.get('rmse', float('inf'))
            if current_metric < best_metric:
                best_metric = current_metric
                trainer.save_checkpoint(epoch + 1, student_optimizer, best_metric)
                logging.info(f"New best metric: {best_metric:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['training'].get('save_interval', 5) == 0:
            trainer.save_checkpoint(epoch + 1, student_optimizer)
        
        # Update learning rate
        scheduler.step()
        
        # Log training losses
        logging.info(f"Training losses:")
        for key, val in avg_train_losses.items():
            logging.info(f"  {key}: {val:.4f}")
    
    logging.info("\nTraining complete!")
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train depth estimation model with KD')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config, resume_from=args.resume)
