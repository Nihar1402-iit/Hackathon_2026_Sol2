"""
Dataset handling for depth estimation.

Supports NYU Depth V2 and KITTI datasets with proper data loading.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, Optional
import os
from pathlib import Path
from PIL import Image
import cv2


class DepthDataset(Dataset):
    """
    Base depth estimation dataset.
    
    Subclass this for specific datasets.
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 img_size: int = 384, augment: bool = False):
        """
        Args:
            data_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            img_size: Target image size (square)
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # This should be implemented in subclasses
        self.samples = []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load RGB image."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_depth(self, path: Path) -> np.ndarray:
        """Load depth map (subclass specific)."""
        raise NotImplementedError
    
    def _resize(self, img: np.ndarray, depth: np.ndarray, 
                img_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and depth to target size."""
        h, w = img.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Pad to square
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        img = cv2.copyMakeBorder(img, pad_top, pad_h - pad_top, 
                                  pad_left, pad_w - pad_left,
                                  cv2.BORDER_CONSTANT, value=0)
        depth = cv2.copyMakeBorder(depth, pad_top, pad_h - pad_top,
                                    pad_left, pad_w - pad_left,
                                    cv2.BORDER_CONSTANT, value=0)
        
        return img, depth


class MockDepthDataset(Dataset):
    """
    Mock dataset for testing.
    
    Generates random images and depth maps.
    """
    
    def __init__(self, n_samples: int = 100, img_size: int = 384,
                 split: str = 'train', augment: bool = False):
        """
        Args:
            n_samples: Number of samples
            img_size: Image size
            split: Dataset split
            augment: Whether to augment
        """
        self.n_samples = n_samples
        self.img_size = img_size
        self.split = split
        self.augment = augment
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with keys:
                - image: (3, H, W) image tensor, normalized to [0, 1]
                - depth: (1, H, W) depth map, normalized to [0, 1]
                - mask: (1, H, W) valid pixel mask
        """
        # Random image
        image = torch.randn(3, self.img_size, self.img_size)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        
        # Random depth
        depth = torch.rand(1, self.img_size, self.img_size)
        
        # Random mask (80% valid)
        mask = torch.rand(1, self.img_size, self.img_size)
        mask = (mask > 0.2).float()
        
        return {
            'image': image,
            'depth': depth,
            'mask': mask
        }


class NYUDepthV2Dataset(DepthDataset):
    """
    NYU Depth V2 dataset loader.
    
    Expected structure:
    data_dir/
        train/
            rgb/           # RGB images as PNG or JPG
            depth/         # Depth maps as PNG or NPY
        val/
            rgb/
            depth/
    
    Depth maps should be in meters (float32).
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 img_size: int = 384, augment: bool = False):
        """
        Args:
            data_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            img_size: Target image size (square)
            augment: Use data augmentation
        """
        super().__init__(data_dir, split, img_size, augment)
        
        # Get file lists
        rgb_dir = self.data_dir / split / 'rgb'
        depth_dir = self.data_dir / split / 'depth'
        
        if rgb_dir.exists() and depth_dir.exists():
            # Find RGB files
            self.rgb_files = sorted(list(rgb_dir.glob('*.png')) + 
                                   list(rgb_dir.glob('*.jpg')))
            
            # Find corresponding depth files
            self.depth_files = []
            for rgb_file in self.rgb_files:
                # Try both .png and .npy extensions
                depth_png = depth_dir / (rgb_file.stem + '.png')
                depth_npy = depth_dir / (rgb_file.stem + '.npy')
                
                if depth_png.exists():
                    self.depth_files.append(depth_png)
                elif depth_npy.exists():
                    self.depth_files.append(depth_npy)
            
            # Only keep pairs where both exist
            pairs = [(r, d) for r, d in zip(self.rgb_files, self.depth_files) 
                    if d.exists()]
            if pairs:
                self.rgb_files, self.depth_files = zip(*pairs)
                self.rgb_files = list(self.rgb_files)
                self.depth_files = list(self.depth_files)
        else:
            print(f"Warning: NYU dataset dirs not found at {rgb_dir}")
            self.rgb_files = []
            self.depth_files = []
    
    def __len__(self) -> int:
        return len(self.rgb_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load NYU Depth V2 sample.
        
        Returns:
            image: (3, H, W) normalized RGB
            depth: (1, H, W) depth in meters
            mask: (1, H, W) valid pixel mask
        """
        if len(self.rgb_files) == 0:
            # Return mock data if dataset not found
            return self._get_mock_sample()
        
        # Load RGB image
        rgb_path = self.rgb_files[idx]
        img = self._load_image(rgb_path)
        
        # Load depth map
        depth_path = self.depth_files[idx]
        if depth_path.suffix == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        else:
            # PNG depth (typically stored as uint16 or uint8, scaled)
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32)
            # Scale if needed (depends on how it was saved)
            if depth.max() > 256:  # Likely uint16
                depth = depth / 1000.0  # Convert to meters
            elif depth.max() <= 1:
                depth = depth  # Already normalized
            else:
                depth = depth / 255.0  # Normalize from 0-255
        
        # Ensure proper shape
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # Resize
        img, depth = self._resize(img, depth, self.img_size)
        
        # Create mask (valid pixels)
        mask = (depth > 0).astype(np.float32)
        
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)   # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)
        
        return {
            'image': image,
            'depth': depth,
            'mask': mask
        }
    
    def _get_mock_sample(self) -> Dict[str, torch.Tensor]:
        """Return mock sample when dataset not available."""
        image = torch.randn(3, self.img_size, self.img_size)
        image = torch.clamp(image, min=0, max=1)
        depth = torch.rand(1, self.img_size, self.img_size) * 10.0
        mask = torch.ones(1, self.img_size, self.img_size)
        
        return {'image': image, 'depth': depth, 'mask': mask}


class KITTIDataset(DepthDataset):
    """
    KITTI dataset loader for depth estimation.
    
    Expected structure:
    data_dir/
        train/
            image_02/     # Left RGB images
            depth_02/     # Depth maps (from sparse LiDAR or dense completion)
        val/
            image_02/
            depth_02/
    
    Or raw KITTI structure with velodyne data.
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 img_size: int = 384, augment: bool = False):
        """
        Args:
            data_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            img_size: Target image size (square)
            augment: Use data augmentation
        """
        super().__init__(data_dir, split, img_size, augment)
        
        # Get file lists
        img_dir = self.data_dir / split / 'image_02'
        depth_dir = self.data_dir / split / 'depth_02'
        
        if img_dir.exists() and depth_dir.exists():
            # Find image files
            self.img_files = sorted(list(img_dir.glob('*.png')) + 
                                   list(img_dir.glob('*.jpg')))
            
            # Find corresponding depth files
            self.depth_files = []
            for img_file in self.img_files:
                # Try both .png and .npy extensions
                depth_png = depth_dir / (img_file.stem + '.png')
                depth_npy = depth_dir / (img_file.stem + '.npy')
                
                if depth_png.exists():
                    self.depth_files.append(depth_png)
                elif depth_npy.exists():
                    self.depth_files.append(depth_npy)
            
            # Only keep pairs
            pairs = [(i, d) for i, d in zip(self.img_files, self.depth_files) 
                    if d.exists()]
            if pairs:
                self.img_files, self.depth_files = zip(*pairs)
                self.img_files = list(self.img_files)
                self.depth_files = list(self.depth_files)
        else:
            print(f"Warning: KITTI dataset dirs not found at {img_dir}")
            self.img_files = []
            self.depth_files = []
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load KITTI sample.
        
        Returns:
            image: (3, H, W) normalized RGB
            depth: (1, H, W) depth in meters
            mask: (1, H, W) valid pixel mask (0 for invalid/occluded)
        """
        if len(self.img_files) == 0:
            # Return mock data if dataset not found
            return self._get_mock_sample()
        
        # Load RGB image
        img_path = self.img_files[idx]
        img = self._load_image(img_path)
        
        # Load depth map
        depth_path = self.depth_files[idx]
        if depth_path.suffix == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        else:
            # PNG depth (typically stored as uint16, in mm)
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32)
            # KITTI stores depth in mm, convert to meters
            depth = depth / 1000.0
        
        # Ensure proper shape
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # Resize
        img, depth = self._resize(img, depth, self.img_size)
        
        # Create mask (valid pixels, KITTI: 0 means invalid/occluded)
        mask = (depth > 0).astype(np.float32)
        
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)   # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)
        
        return {
            'image': image,
            'depth': depth,
            'mask': mask
        }
    
    def _get_mock_sample(self) -> Dict[str, torch.Tensor]:
        """Return mock sample when dataset not available."""
        image = torch.randn(3, self.img_size, self.img_size)
        image = torch.clamp(image, min=0, max=1)
        depth = torch.rand(1, self.img_size, self.img_size) * 100.0
        mask = torch.ones(1, self.img_size, self.img_size)
        
        return {'image': image, 'depth': depth, 'mask': mask}


def create_dataset(dataset_type: str = 'mock', split: str = 'train',
                   data_dir: str = './data', img_size: int = 384,
                   augment: bool = False, n_samples: int = 100) -> Dataset:
    """
    Create dataset.
    
    Args:
        dataset_type: 'mock', 'nyu', 'kitti'
        split: 'train', 'val', 'test'
        data_dir: Data directory
        img_size: Image size
        augment: Use augmentation
        n_samples: For mock dataset, number of samples
    
    Returns:
        Dataset instance
    """
    if dataset_type == 'mock':
        return MockDepthDataset(n_samples=n_samples, img_size=img_size,
                               split=split, augment=augment)
    elif dataset_type == 'nyu':
        return NYUDepthV2Dataset(data_dir, split=split, img_size=img_size,
                                augment=augment)
    elif dataset_type == 'kitti':
        return KITTIDataset(data_dir, split=split, img_size=img_size,
                           augment=augment)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
