#!/bin/bash
# Fix NumPy version conflict for PyTorch 2.8+
# PyTorch 2.8+ includes NumPy 2.x which breaks compatibility

set -e

echo "=========================================="
echo "NumPy Version Fix for 4D Gaussian Splatting"
echo "=========================================="
echo ""

# Use python3/pip3 explicitly
PYTHON_CMD="${PYTHON_CMD:-python3}"
PIP_CMD="${PIP_CMD:-pip3}"

# Check current Python
echo "Step 1: Checking Python installation..."
$PYTHON_CMD --version || {
    echo "Error: Python3 not found!"
    echo "On Windows, you may need to:"
    echo "  1. Use 'py' instead of 'python3'"
    echo "  2. Or specify full path to python.exe"
    exit 1
}

# Check current NumPy
echo ""
echo "Step 2: Checking current NumPy version..."
$PYTHON_CMD -c "import numpy; print(f'Current NumPy: {numpy.__version__}')" 2>/dev/null || {
    echo "NumPy not installed yet"
}

# Check PyTorch
echo ""
echo "Step 3: Checking PyTorch version..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "PyTorch not found! Please install PyTorch first."
    exit 1
}

echo ""
echo "Step 4: Fixing NumPy version..."
echo "  PyTorch 2.8+ comes with NumPy 2.x"
echo "  We need NumPy 1.24.x for compatibility"
echo ""

# Method 1: Try direct install with force
echo "Method 1: Force installing NumPy 1.24.3..."
$PIP_CMD uninstall numpy -y 2>/dev/null || true
$PIP_CMD install numpy==1.24.3 --force-reinstall --no-deps --no-cache-dir

# Verify
$PYTHON_CMD -c "
import numpy
ver = numpy.__version__
print(f'Installed NumPy: {ver}')
if ver.startswith('1.24'):
    print('✓ Success! NumPy 1.24.x installed')
else:
    print(f'✗ Warning: NumPy {ver} installed, expected 1.24.x')
    exit(1)
" && exit 0

# Method 2: If method 1 failed
echo ""
echo "Method 2: Trying alternative install..."
$PIP_CMD install 'numpy<1.25,>=1.24' --force-reinstall --no-cache-dir

# Final verification
$PYTHON_CMD -c "
import numpy
ver = numpy.__version__
print(f'Installed NumPy: {ver}')
if ver.startswith('1.24'):
    print('✓ Success! NumPy 1.24.x installed')
else:
    print(f'✗ Failed: NumPy {ver} installed, expected 1.24.x')
    print('You may need to manually install NumPy 1.24.3')
    exit(1)
"

echo ""
echo "=========================================="
echo "NumPy fix complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the full installation: bash install.sh"
echo "  2. Or continue with training: python3 tools/train.py --data_root dataset/ --out_dir model/"