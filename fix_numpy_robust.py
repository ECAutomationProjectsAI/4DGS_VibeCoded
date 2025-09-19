#!/usr/bin/env python3
"""
Robust NumPy version fixer for Windows and Linux.
Handles NumPy 1.x vs 2.x compatibility issues with PyTorch 2.8+.
"""

import sys
import subprocess
import os
import shutil
import site
import importlib.util

def run_command(cmd_list, check=False):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def clean_numpy_completely():
    """Remove all NumPy installations from all locations."""
    print("Performing deep clean of NumPy installations...")
    
    # Get all possible site-packages directories
    site_dirs = []
    
    # Standard site-packages
    site_dirs.extend(site.getsitepackages())
    
    # User site-packages
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site:
            site_dirs.append(user_site)
    
    # Add Python's lib directory
    import sysconfig
    site_dirs.append(sysconfig.get_paths()['purelib'])
    
    # Remove duplicates and non-existent paths
    site_dirs = [d for d in set(site_dirs) if os.path.exists(d)]
    
    removed_count = 0
    for site_dir in site_dirs:
        try:
            items_to_remove = []
            for item in os.listdir(site_dir):
                if item.lower().startswith('numpy') or 'numpy' in item.lower():
                    items_to_remove.append(os.path.join(site_dir, item))
            
            for item_path in items_to_remove:
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"  Removed: {item_path}")
                        removed_count += 1
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"  Removed: {item_path}")
                        removed_count += 1
                except PermissionError:
                    print(f"  Permission denied: {item_path}")
                except Exception as e:
                    print(f"  Could not remove {item_path}: {e}")
        except Exception as e:
            print(f"  Error scanning {site_dir}: {e}")
    
    return removed_count > 0

def main():
    print("=" * 60)
    print("Robust NumPy Fix for 4D Gaussian Splatting")
    print("=" * 60)
    print()
    
    # Detect platform
    is_windows = sys.platform.startswith('win')
    print(f"Platform: {'Windows' if is_windows else 'Linux/Unix'}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Check current NumPy
    try:
        import numpy
        current_ver = numpy.__version__
        numpy_path = numpy.__file__
        print(f"Current NumPy: {current_ver}")
        print(f"Location: {os.path.dirname(numpy_path)}")
        
        if current_ver.startswith("1.24"):
            print("✅ NumPy 1.24.x already installed!")
            return 0
    except ImportError:
        print("NumPy not currently installed")
        current_ver = None
        numpy_path = None
    
    print()
    print("Starting fix process...")
    print()
    
    # Step 1: Uninstall via pip (multiple times)
    print("Step 1: Uninstalling NumPy via pip...")
    pip_cmd = [sys.executable, "-m", "pip"]
    
    for i in range(3):
        stdout, stderr, code = run_command(pip_cmd + ["uninstall", "numpy", "-y"])
        if "not installed" in stdout.lower() or "not installed" in stderr.lower():
            break
        if code == 0:
            print(f"  Uninstall pass {i+1} successful")
    
    # Step 2: Deep clean
    print("\nStep 2: Deep cleaning NumPy from all locations...")
    if clean_numpy_completely():
        print("  Deep clean completed")
    else:
        print("  No NumPy installations found to clean")
    
    # Step 3: Clear pip cache
    print("\nStep 3: Clearing pip cache...")
    run_command(pip_cmd + ["cache", "purge"])
    
    # Clear Windows/Linux cache manually
    if is_windows:
        # Windows pip cache locations
        cache_dirs = [
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'pip', 'Cache'),
            os.path.join(os.environ.get('USERPROFILE', ''), 'pip', 'Cache'),
        ]
    else:
        # Linux pip cache location
        cache_dirs = [
            os.path.expanduser('~/.cache/pip'),
        ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"  Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"  Could not clear {cache_dir}: {e}")
    
    # Step 4: Install NumPy 1.24.3
    print("\nStep 4: Installing NumPy 1.24.3...")
    
    # Try multiple installation methods
    install_methods = [
        # Method 1: Direct version with no-deps
        pip_cmd + ["install", "numpy==1.24.3", "--no-deps", "--no-cache-dir"],
        # Method 2: Force reinstall
        pip_cmd + ["install", "numpy==1.24.3", "--force-reinstall", "--no-cache-dir"],
        # Method 3: Version range
        pip_cmd + ["install", "numpy>=1.24,<1.25", "--force-reinstall", "--no-cache-dir"],
        # Method 4: Download wheel directly (Windows)
        pip_cmd + ["install", "numpy==1.24.3", "--only-binary", ":all:", "--no-cache-dir"] if is_windows else None,
    ]
    
    success = False
    for i, method in enumerate(install_methods, 1):
        if method is None:
            continue
        print(f"  Trying installation method {i}...")
        stdout, stderr, code = run_command(method)
        if code == 0:
            print(f"  Method {i} succeeded!")
            success = True
            break
        else:
            error_msg = stderr[:200] if stderr else stdout[:200]
            print(f"  Method {i} failed: {error_msg}")
    
    if not success:
        print("\n❌ All installation methods failed!")
        print("\nManual fix required:")
        print("1. Close all Python processes")
        print("2. Run as Administrator/root:")
        print(f"   {' '.join(pip_cmd)} install numpy==1.24.3 --force-reinstall --no-cache-dir")
        return 1
    
    # Step 5: Verify installation
    print("\nStep 5: Verifying installation...")
    
    # Clear module cache
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    
    # Force reload
    try:
        import numpy
        new_ver = numpy.__version__
        print(f"  Installed NumPy: {new_ver}")
        print(f"  Location: {os.path.dirname(numpy.__file__)}")
        
        if new_ver.startswith("1.24"):
            print("\n✅ SUCCESS! NumPy 1.24.x installed successfully!")
            print("\nYou can now run:")
            print("  python tools/train.py --data_root dataset/ --out_dir model/")
            return 0
        else:
            print(f"\n⚠️ WARNING: NumPy {new_ver} installed, but we need 1.24.x")
            return 1
    except ImportError as e:
        print(f"\n❌ ERROR: Cannot import NumPy after installation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())