"""
Main entry point for depth estimation training and inference.

Supports:
- Training with knowledge distillation
- Evaluation on test sets
- Inference on single images
- Model export
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import logging
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from .models.student_model import create_student_model
from .models.teacher_model import create_teacher_model
from .training.train import train
from .utils.flops import count_parameters, print_model_summary, ModelProfiler
from .utils.metrics import compute_metrics
from .data.dataset import create_dataset
from torch.utils.data import DataLoader


def setup_logging(log_dir: str = './logs'):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir) / 'main.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_command(config: Dict, resume_from: Optional[str] = None):
    """
    Training command.
    
    Args:
        config: Configuration dictionary
        resume_from: Path to checkpoint to resume from
    """
    logging.info("Starting training...")
    logging.info(f"Student: {config['student_model']['backbone_type']}")
    logging.info(f"Teacher: {config['teacher_model']['backbone_type']}")
    
    train(config, resume_from=resume_from)


def evaluate_command(config: Dict, checkpoint_path: str):
    """
    Evaluation command.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to trained model checkpoint
    """
    logging.info(f"Evaluating model: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    student = create_student_model(config['student_model']).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()
    
    # Create test dataset
    test_dataset = create_dataset(
        dataset_type=config['data']['dataset_type'],
        split='test',
        img_size=config['data']['img_size'],
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Evaluation loop
    all_metrics = {metric: [] for metric in ['rmse', 'abs_rel', 'delta1', 'delta2', 'delta3']}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            depths_gt = batch['depth'].to(device)
            
            # Forward
            output = student(images)
            depths_pred = output['depth']
            
            # Compute metrics
            metrics = compute_metrics(depths_pred, depths_gt)
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(metrics[key])
            
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Evaluated {batch_idx + 1} batches")
    
    # Print results
    logging.info("\n" + "="*60)
    logging.info("Test Results")
    logging.info("="*60)
    for key in all_metrics:
        avg = sum(all_metrics[key]) / len(all_metrics[key]) if all_metrics[key] else 0
        logging.info(f"{key}: {avg:.4f}")
    logging.info("="*60)


def infer_command(config: Dict, checkpoint_path: str, image_path: str,
                 output_path: Optional[str] = None):
    """
    Inference on single image.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to trained model
        image_path: Path to input image
        output_path: Path to save output depth
    """
    logging.info(f"Running inference on {image_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    student = create_student_model(config['student_model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()
    
    # Load image
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = student(image_tensor)
            depth_pred = output['depth']
        
        logging.info(f"Predicted depth shape: {depth_pred.shape}")
        logging.info(f"Depth range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
        
        # Save if output path provided
        if output_path:
            torch.save(depth_pred, output_path)
            logging.info(f"Saved depth to {output_path}")
        
        return depth_pred
    
    except ImportError:
        logging.error("PIL required for image inference. Install with: pip install pillow")


def benchmark_command(config: Dict):
    """
    Benchmark model performance (FLOPs, memory, speed).
    
    Args:
        config: Configuration dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Benchmarking on {device}")
    
    input_size = (1, 3, config['data']['img_size'], config['data']['img_size'])
    
    # Student model
    logging.info("\n" + "="*60)
    logging.info("Student Model Benchmark")
    logging.info("="*60)
    student = create_student_model(config['student_model'])
    print_model_summary(student, input_size, device=str(device))
    
    if device.type == 'cuda':
        profiler = ModelProfiler(student, device=str(device))
        forward_time = profiler.profile_forward(input_size, num_iterations=10)
        peak_mem, alloc_mem = profiler.profile_memory(input_size)
        
        logging.info(f"Forward pass time: {forward_time:.2f} ms")
        logging.info(f"Peak memory: {peak_mem:.2f} MB")
        logging.info(f"Allocated memory: {alloc_mem:.2f} MB")
    
    # Teacher model
    logging.info("\n" + "="*60)
    logging.info("Teacher Model Benchmark")
    logging.info("="*60)
    teacher = create_teacher_model(config['teacher_model'])
    print_model_summary(teacher, input_size, device=str(device))
    
    if device.type == 'cuda':
        profiler = ModelProfiler(teacher, device=str(device))
        forward_time = profiler.profile_forward(input_size, num_iterations=10)
        peak_mem, alloc_mem = profiler.profile_memory(input_size)
        
        logging.info(f"Forward pass time: {forward_time:.2f} ms")
        logging.info(f"Peak memory: {peak_mem:.2f} MB")
        logging.info(f"Allocated memory: {alloc_mem:.2f} MB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monocular Depth Estimation with Vision Transformers and Knowledge Distillation'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    infer_parser.add_argument('--image', type=str, required=True, help='Input image path')
    infer_parser.add_argument('--output', type=str, default=None, help='Output depth path')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark models')
    bench_parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.command == 'train':
        config = load_config(args.config)
        train_command(config, resume_from=args.resume)
    
    elif args.command == 'eval':
        config = load_config(args.config)
        evaluate_command(config, args.checkpoint)
    
    elif args.command == 'infer':
        config = load_config(args.config)
        infer_command(config, args.checkpoint, args.image, args.output)
    
    elif args.command == 'benchmark':
        config = load_config(args.config)
        benchmark_command(config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
