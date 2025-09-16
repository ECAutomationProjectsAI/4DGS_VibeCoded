#!/usr/bin/env python3
"""
Fix NumPy version conflicts with PyTorch 2.8+
PyTorch 2.8+ includes NumPy 2.x which breaks compatibility with our code.
This script forces NumPy 1.24.x installation.
"""

import sys
import subprocess
import importlib

def run_command(cmd, check=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode

def check_numpy():
    """Check current NumPy version."""
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None

def main():
    print("=" * 50)
    print("NumPy Version Fix for 4D Gaussian Splatting")
    print("=" * 50)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check PyTorch
    pytorch_ver = check_pytorch()
    if not pytorch_ver:
        print("❌ ERROR: PyTorch not found!")
        print("Please install PyTorch first:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    print(f"✓ PyTorch version: {pytorch_ver}")
    
    # Check current NumPy
    numpy_ver = check_numpy()
    if numpy_ver:
        print(f"Current NumPy version: {numpy_ver}")
        
        if numpy_ver.startswith("1.24"):
            print("✓ NumPy 1.24.x already installed - no fix needed!")
            return 0
        elif numpy_ver.startswith("2."):
            print("⚠️  NumPy 2.x detected - need to downgrade")
        else:
            print(f"⚠️  NumPy {numpy_ver} detected - need to change to 1.24.x")
    else:
        print("NumPy not installed")
    
    print()
    print("Fixing NumPy version...")
    print("  PyTorch 2.8+ comes with NumPy 2.x")
    print("  We need NumPy 1.24.x for compatibility")
    print()
    
    # Get pip command
    pip_cmd = sys.executable + " -m pip"
    
    # Method 1: Uninstall and reinstall
    print("Step 1: Uninstalling current NumPy...")
    stdout, stderr, code = run_command(f"{pip_cmd} uninstall numpy -y", check=False)
    
    print("Step 2: Installing NumPy 1.24.3...")
    stdout, stderr, code = run_command(
        f"{pip_cmd} install numpy==1.24.3 --force-reinstall --no-deps --no-cache-dir"
    )
    
    if code != 0:
        print("  First attempt failed, trying alternative...")
        stdout, stderr, code = run_command(
            f"{pip_cmd} install 'numpy<1.25,>=1.24' --force-reinstall --no-cache-dir"
        )
    
    # Reload numpy module
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    
    # Verify installation
    try:
        import numpy
        new_ver = numpy.__version__
        print()
        print(f"Installed NumPy version: {new_ver}")
        
        if new_ver.startswith("1.24"):
            print("✅ Success! NumPy 1.24.x installed successfully!")
            print()
            print("Next steps:")
            print("  1. Run full installation: python install.sh (or bash install.sh)")
            print("  2. Or continue with training: python tools/train.py --data_root dataset/ --out_dir model/")
            return 0
        else:
            print(f"⚠️  Warning: NumPy {new_ver} installed, but we need 1.24.x")
            print()
            print("Manual fix:")
            print(f"  {pip_cmd} uninstall numpy -y")
            print(f"  {pip_cmd} install numpy==1.24.3 --no-deps")
            return 1
            
    except ImportError as e:
        print(f"❌ ERROR: Failed to import NumPy after installation: {e}")
        return 1
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    sys.exit(main())