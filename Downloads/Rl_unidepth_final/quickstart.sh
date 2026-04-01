#!/usr/bin/env bash

# UniDepth-KD Quick Start Script
# Sets up the environment and runs basic tests

set -e

echo "================================"
echo "UniDepth-KD Quick Start"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "================================"
echo "Testing Installation"
echo "================================"
echo ""

# Run quick test
python3 << 'EOF'
import sys
sys.path.insert(0, 'unidepth_kd')

print("✓ Imports successful")

# Test basic model creation
import torch
from models.vit_encoder import create_vit_encoder
from models.student_model import create_student_model

device = torch.device('cpu')
print("✓ Using device:", device)

# Create tiny ViT
vit = create_vit_encoder('tiny', img_size=384)
print("✓ ViT encoder created")

# Create student model
student = create_student_model({
    'backbone_type': 'vit_tiny',
    'img_size': 384,
    'use_token_merging': True,
    'decoder_type': 'lightweight',
})
print("✓ Student model created")

# Test forward pass
x = torch.randn(1, 3, 384, 384)
output = student(x)
print(f"✓ Forward pass successful, depth shape: {output['depth'].shape}")

print("\n✓✓✓ Installation successful! ✓✓✓")
EOF

echo ""
echo "================================"
echo "Next Steps"
echo "================================"
echo ""
echo "1. Review the configuration:"
echo "   cat unidepth_kd/configs/config.yaml"
echo ""
echo "2. Run the test suite:"
echo "   python3 test_all.py"
echo ""
echo "3. Start training:"
echo "   python3 unidepth_kd/main.py train --config unidepth_kd/configs/config.yaml"
echo ""
echo "4. View documentation:"
echo "   cat README.md"
echo ""
