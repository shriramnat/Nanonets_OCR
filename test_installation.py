#!/usr/bin/env python3
"""
Test script to verify Nanonets OCR Local Runner installation
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test if Python version is compatible."""
    print("🐍 Testing Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version} is compatible!")
    return True

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n📦 Testing dependencies...")
    
    required_modules = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    missing_modules = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} is installed")
        except ImportError:
            print(f"❌ {display_name} is missing")
            missing_modules.append(display_name)
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_gpu_availability():
    """Test GPU availability."""
    print("\n🔍 Testing GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} GPU(s) available)")
            print(f"✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("ℹ️ No GPU detected. CPU mode will be used.")
            return False
    except Exception as e:
        print(f"⚠️ Error checking GPU: {e}")
        return False

def test_model_import():
    """Test if the main OCR class can be imported."""
    print("\n🔧 Testing main OCR class import...")
    try:
        from nanonets_ocr_local import NanonetsOCR
        print("✅ NanonetsOCR class imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import NanonetsOCR: {e}")
        return False

def test_file_structure():
    """Test if all required files are present."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "nanonets_ocr_local.py",
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename} found")
        else:
            print(f"❌ {filename} missing")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_model_initialization():
    """Test if the model can be initialized (without loading)."""
    print("\n🤖 Testing model initialization...")
    try:
        from nanonets_ocr_local import NanonetsOCR
        ocr = NanonetsOCR()
        print("✅ Model initialized successfully")
        print(f"✅ Model path: {ocr.model_path}")
        print(f"✅ Device: {ocr.device}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Nanonets OCR Installation Test")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("GPU Availability", test_gpu_availability),
        ("Model Import", test_model_import),
        ("File Structure", test_file_structure),
        ("Model Initialization", test_model_initialization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready to use.")
        print("\nNext steps:")
        print("1. Run: python nanonets_ocr_local.py --interactive")
        print("2. Or check: python example_usage.py")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("1. Run: python setup.py")
        print("2. Or: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
