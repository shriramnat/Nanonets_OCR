#!/usr/bin/env python3
"""
Quick fix script to install missing torchvision dependency
"""

import subprocess
import sys

def install_torchvision():
    """Install torchvision specifically."""
    print("🔧 Installing missing torchvision dependency...")
    
    commands = [
        f"{sys.executable} -m pip install --upgrade pip",
        f"{sys.executable} -m pip install torchvision>=0.15.0",
        f"{sys.executable} -m pip install torch torchvision --upgrade"
    ]
    
    for cmd in commands:
        print(f"\nRunning: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print("✅ Command completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            continue
    
    # Test the installation
    try:
        import torchvision
        print("\n✅ TorchVision installed successfully!")
        print(f"✅ TorchVision version: {torchvision.__version__}")
        return True
    except ImportError:
        print("\n❌ TorchVision installation failed.")
        return False

def main():
    print("🚀 Nanonets OCR - Dependency Fix")
    print("=" * 40)
    print("This script will install the missing torchvision dependency.")
    
    if install_torchvision():
        print("\n🎉 Dependencies fixed successfully!")
        print("\nNow you can run:")
        print("  python nanonets_ocr_local.py --interactive")
        print("  python test_installation.py")
    else:
        print("\n❌ Failed to fix dependencies.")
        print("\nTry manual installation:")
        print("  pip install torch torchvision")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
