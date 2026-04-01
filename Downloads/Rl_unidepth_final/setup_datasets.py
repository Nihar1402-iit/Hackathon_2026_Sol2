#!/usr/bin/env python3
"""
Setup script for downloading and preparing datasets.

Supports NYU Depth V2 and KITTI datasets.
"""

import os
import argparse
import shutil
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetSetup:
    """Helper class for dataset setup."""
    
    @staticmethod
    def setup_nyu_depth_v2(data_dir: str = './data'):
        """
        Setup NYU Depth V2 dataset.
        
        NYU Depth V2 can be downloaded from:
        - Official: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2_labeled.mat
        - Or preprocessed splits from community repos
        
        Expected structure after setup:
        data_dir/nyu_depth_v2/
            train/
                rgb/
                depth/
            val/
                rgb/
                depth/
        """
        nyu_dir = Path(data_dir) / 'nyu_depth_v2'
        nyu_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"NYU Depth V2 dataset directory: {nyu_dir}")
        logger.info("""
        To download NYU Depth V2:
        
        1. Download from official source:
           https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2_labeled.mat
        
        2. Or use preprocessed version from community:
           https://github.com/ialhashim/DenseDepthMap/blob/master/nyu_depth_v2_labeled.npz
        
        3. After downloading, create the following structure:
        
           nyu_depth_v2/
           ├── train/
           │   ├── rgb/       (RGB images as PNG)
           │   └── depth/     (Depth maps as PNG or NPY, in meters)
           └── val/
               ├── rgb/
               └── depth/
        
        4. Recommended: Use around 1449 images for training, 215 for validation
        
        5. Depth format:
           - Can be stored as uint16 PNG (in mm) or float32 NPY (in meters)
           - If PNG: depth_value_meters = png_value / 1000.0
           - If NPY: already in meters, clip to valid range
        """)
        
        return nyu_dir
    
    @staticmethod
    def setup_kitti(data_dir: str = './data'):
        """
        Setup KITTI dataset for depth estimation.
        
        KITTI can be downloaded from:
        https://www.cvlibs.net/datasets/kitti/eval_depth.php
        
        Expected structure:
        data_dir/kitti/
            train/
                image_02/      (Left camera images)
                depth_02/      (Ground truth depth)
            val/
                image_02/
                depth_02/
        """
        kitti_dir = Path(data_dir) / 'kitti'
        kitti_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"KITTI dataset directory: {kitti_dir}")
        logger.info("""
        To download KITTI Depth dataset:
        
        1. Register at: https://www.cvlibs.net/datasets/kitti/
        
        2. Download:
           - Raw data (~175 GB total, or select specific sequences)
           - Depth completion dataset
           - Depth maps from stereo (recommended for testing)
        
        3. After downloading, create structure:
        
           kitti/
           ├── train/
           │   ├── image_02/     (PNG images from left camera)
           │   └── depth_02/     (PNG depth maps, uint16, in mm)
           └── val/
               ├── image_02/
               └── depth_02/
        
        4. For testing:
           Place test set similarly, KITTI will use it for evaluation
        
        5. Depth format:
           - Stored as uint16 PNG in millimeters
           - 0 = invalid/occluded
           - value_meters = png_value / 1000.0
        
        6. Typical split:
           - Training: ~23k images from raw sequences
           - Validation/Test: 1000 images each from depth completion test set
        """)
        
        return kitti_dir
    
    @staticmethod
    def create_sample_dataset(data_dir: str = './data'):
        """
        Create a small sample dataset for testing.
        
        This creates synthetic data that matches the expected format.
        """
        logger.info("Creating sample dataset for testing...")
        
        # Create NYU sample
        nyu_dir = Path(data_dir) / 'nyu_depth_v2'
        for split in ['train', 'val']:
            rgb_dir = nyu_dir / split / 'rgb'
            depth_dir = nyu_dir / split / 'depth'
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 10 sample images for training, 2 for val
            n_samples = 10 if split == 'train' else 2
            for i in range(n_samples):
                # Create random RGB image
                rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(rgb_dir / f'{i:06d}.png')
                
                # Create random depth (0-10 meters, stored in mm as uint16)
                depth = np.random.randint(0, 10000, (480, 640), dtype=np.uint16)
                Image.fromarray(depth).save(depth_dir / f'{i:06d}.png')
        
        logger.info(f"✓ NYU sample dataset created at {nyu_dir}")
        
        # Create KITTI sample
        kitti_dir = Path(data_dir) / 'kitti'
        for split in ['train', 'val']:
            img_dir = kitti_dir / split / 'image_02'
            depth_dir = kitti_dir / split / 'depth_02'
            img_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 10 sample images
            n_samples = 10 if split == 'train' else 2
            for i in range(n_samples):
                # Create random RGB image (KITTI: 1242x375)
                rgb = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(img_dir / f'{i:06d}.png')
                
                # Create random depth (0-100 meters, stored in mm as uint16)
                depth = np.random.randint(0, 100000, (375, 1242), dtype=np.uint16)
                Image.fromarray(depth).save(depth_dir / f'{i:06d}.png')
        
        logger.info(f"✓ KITTI sample dataset created at {kitti_dir}")


def main():
    parser = argparse.ArgumentParser(description='Setup datasets for depth estimation')
    parser.add_argument('--data_dir', default='./data', help='Root data directory')
    parser.add_argument('--nyu', action='store_true', help='Setup NYU Depth V2')
    parser.add_argument('--kitti', action='store_true', help='Setup KITTI')
    parser.add_argument('--sample', action='store_true', help='Create sample datasets')
    parser.add_argument('--all', action='store_true', help='Setup all datasets')
    
    args = parser.parse_args()
    
    if args.all or args.sample:
        DatasetSetup.create_sample_dataset(args.data_dir)
    
    if args.all or args.nyu:
        DatasetSetup.setup_nyu_depth_v2(args.data_dir)
    
    if args.all or args.kitti:
        DatasetSetup.setup_kitti(args.data_dir)
    
    if not (args.nyu or args.kitti or args.sample or args.all):
        logger.info("No action specified. Use --help for options.")
        logger.info("\nQuick start:")
        logger.info("  python setup_datasets.py --sample  # Create sample data for testing")
        logger.info("  python setup_datasets.py --nyu     # Setup NYU Depth V2")
        logger.info("  python setup_datasets.py --kitti   # Setup KITTI")


if __name__ == '__main__':
    main()
