#!/usr/bin/env python3
"""
NumPy 1.24.3 installer for RunPod Ubuntu environment.
Specifically for: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
"""

import sys
import subprocess
import os
import shutil
import site

def run_command(cmd_list):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def clean_numpy():
    """Remove all NumPy installations from Ubuntu system."""
    print("Cleaning NumPy installations...")
    
    # Get site-packages directories
    site_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site and os.path.exists(user_site):
            site_dirs.append(user_site)
    
    # Clean numpy from all locations
    for site_dir in site_dirs:
        if os.path.exists(site_dir):
            for item in os.listdir(site_dir):
                if 'numpy' in item.lower():
                    path = os.path.join(site_dir, item)
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                            print(f"  Removed: {path}")
                        else:
                            os.remove(path)
                            print(f"  Removed: {path}")
                    except Exception as e:
                        print(f"  Could not remove {path}: {e}")

def main():
    print("=" * 60)
    print("NumPy 1.24.3 Installer for RunPod")
    print("Target: Ubuntu 22.04 with PyTorch 2.8.0")
    print("=" * 60)
    print()
    
    # Check current NumPy
    try:
        import numpy
        current = numpy.__version__
        print(f"Current NumPy: {current}")
        
        if current.startswith("1.24"):
            print("✅ NumPy 1.24.x already installed!")
            return 0
    except ImportError:
        print("NumPy not installed")
    
    print("\nStarting installation process...")
    
    # Step 1: Uninstall via pip
    print("\n1. Uninstalling NumPy via pip...")
    pip = [sys.executable, "-m", "pip"]
    
    for _ in range(2):
        stdout, stderr, code = run_command(pip + ["uninstall", "numpy", "-y"])
        if "not installed" in (stdout + stderr).lower():
            break
    
    # Step 2: Clean filesystem
    print("\n2. Cleaning filesystem...")
    clean_numpy()
    
    # Step 3: Clear pip cache
    print("\n3. Clearing pip cache...")
    cache_dir = os.path.expanduser("~/.cache/pip")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"  Cleared: {cache_dir}")
        except Exception as e:
            print(f"  Could not clear cache: {e}")
    
    run_command(pip + ["cache", "purge"])
    
    # Step 4: Install NumPy 1.24.3
    print("\n4. Installing NumPy 1.24.3...")
    
    # Try installation
    stdout, stderr, code = run_command(
        pip + ["install", "numpy==1.24.3", "--force-reinstall", "--no-cache-dir"]
    )
    
    if code != 0:
        print("  First attempt failed, trying without force-reinstall...")
        stdout, stderr, code = run_command(
            pip + ["install", "numpy==1.24.3", "--no-cache-dir"]
        )
    
    if code != 0:
        print("❌ Installation failed!")
        print(f"Error: {stderr}")
        return 1
    
    # Step 5: Verify
    print("\n5. Verifying installation...")
    
    # Clear module cache
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    
    try:
        import numpy
        version = numpy.__version__
        print(f"  Installed NumPy: {version}")
        
        if version.startswith("1.24"):
            print("\n✅ SUCCESS! NumPy 1.24.3 installed!")
            print("\nYou can now proceed with training:")
            print("  python tools/train.py --data_root dataset/ --out_dir model/")
            return 0
        else:
            print(f"\n⚠️ Wrong version installed: {version}")
            return 1
    except ImportError as e:
        print(f"\n❌ Cannot import NumPy: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())