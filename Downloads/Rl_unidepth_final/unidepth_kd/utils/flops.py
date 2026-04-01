"""
FLOPs and parameter counting utilities.

Compute computational complexity and model sizes.
"""

import torch
import torch.nn as nn
from typing import Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Args:
        model: Neural network model
        
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int],
                       device: str = 'cpu'):
    """
    Print model summary including parameter count and FLOPs.
    
    Args:
        model: Neural network model
        input_size: Input tensor size (B, C, H, W)
        device: Device to compute on
    """
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Model Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Estimate FLOPs
    try:
        from fvcore.nn import FlopCounterMode
        
        model = model.to(device)
        dummy_input = torch.randn(*input_size, device=device)
        
        with FlopCounterMode(model, display=False) as flop_counter:
            _ = model(dummy_input)
        
        flops = flop_counter.total_flops
        print(f"Estimated FLOPs: {flops / 1e9:.2f}G")
    except ImportError:
        print("(Install fvcore for FLOPs counting: pip install fvcore)")
    
    print(f"{'='*60}\n")


def estimate_memory_usage(model: nn.Module, batch_size: int = 1,
                         input_size: Tuple[int, int] = (384, 384)) -> float:
    """
    Estimate memory usage in MB.
    
    Args:
        model: Neural network model
        batch_size: Batch size
        input_size: Input spatial size (H, W)
        
    Returns:
        Estimated memory in MB
    """
    total_params, _ = count_parameters(model)
    
    # Model parameters (assuming float32)
    param_memory = (total_params * 4) / (1024 ** 2)  # in MB
    
    # Activation memory (rough estimate)
    h, w = input_size
    activation_memory = (batch_size * 3 * h * w * 4) / (1024 ** 2)  # input
    
    # Add some overhead for intermediate activations
    total_memory = param_memory + activation_memory * 10
    
    return total_memory


def compare_models(model1: nn.Module, model2: nn.Module,
                  model_names: Tuple[str, str] = ('Model 1', 'Model 2')):
    """
    Compare two models side-by-side.
    
    Args:
        model1: First model
        model2: Second model
        model_names: Names of models for display
    """
    total1, train1 = count_parameters(model1)
    total2, train2 = count_parameters(model2)
    
    print(f"\n{'='*60}")
    print(f"Model Comparison")
    print(f"{'='*60}")
    
    print(f"\n{model_names[0]}:")
    print(f"  Total: {total1:,}")
    print(f"  Trainable: {train1:,}")
    
    print(f"\n{model_names[1]}:")
    print(f"  Total: {total2:,}")
    print(f"  Trainable: {train2:,}")
    
    ratio = total2 / total1 if total1 > 0 else 0
    reduction = (1 - total2 / total1) * 100 if total1 > 0 else 0
    
    print(f"\nModel 2 / Model 1 ratio: {ratio:.2f}x")
    print(f"Parameter reduction: {reduction:.1f}%")
    print(f"{'='*60}\n")


class ModelProfiler:
    """Profile model execution time and memory."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Model to profile
            device: Device to profile on
        """
        self.model = model.to(device)
        self.device = device
    
    def profile_forward(self, input_size: Tuple[int, int, int, int],
                       num_iterations: int = 10) -> float:
        """
        Profile forward pass time.
        
        Args:
            input_size: Input tensor size (B, C, H, W)
            num_iterations: Number of iterations
            
        Returns:
            Average time in milliseconds
        """
        self.model.eval()
        
        dummy_input = torch.randn(*input_size, device=self.device)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Profile
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
                end = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
                
                if start is not None:
                    start.record()
                
                _ = self.model(dummy_input)
                
                if end is not None:
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
        
        avg_time = sum(times) / len(times) if times else 0
        return avg_time
    
    def profile_memory(self, input_size: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Profile memory usage.
        
        Args:
            input_size: Input tensor size (B, C, H, W)
            
        Returns:
            (peak_memory_mb, allocated_memory_mb)
        """
        self.model.eval()
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        dummy_input = torch.randn(*input_size, device=self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        if self.device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            return peak_memory, allocated_memory
        else:
            return 0.0, 0.0
