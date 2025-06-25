"""
Setup script for Nanonets OCR Local Runner
This script helps install dependencies and set up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version} is compatible!")
    return True

def check_gpu_availability():
    """Check if CUDA is available."""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} GPU(s) available)")
            return True
        else:
            print("ℹ️ No GPU detected. Will use CPU mode.")
            return False
    except ImportError:
        print("ℹ️ PyTorch not installed yet. GPU check will be done after installation.")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies from requirements.txt...")
    
    # Install basic requirements
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing core dependencies"):
        print("\n⚠️ Some dependencies failed to install. Trying without optional packages...")
        
        # Try installing core dependencies only
        core_deps = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.36.0", 
            "pillow>=9.0.0",
            "accelerate>=0.20.0",
            "numpy>=1.21.0",
            "requests>=2.25.0",
            "tqdm>=4.62.0"
        ]
        
        for dep in core_deps:
            if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
                print(f"❌ Failed to install {dep}")
                return False
        
        print("\n✅ Core dependencies installed successfully!")
        print("ℹ️ Optional dependencies (flash-attn, bitsandbytes) may need to be installed manually for better performance.")
    
    return True

def download_model_info():
    """Provide information about model download."""
    print("\n📥 Model Download Information:")
    print("=" * 50)
    print("The Nanonets OCR model will be downloaded automatically when you first run the program.")
    print("Model size: ~3.75B parameters (approximately 7-8 GB)")
    print("Download location: ~/.cache/huggingface/transformers/")
    print("\nFirst run may take 10-30 minutes depending on your internet connection.")
    print("Subsequent runs will be much faster as the model will be cached locally.")

def create_example_script():
    """Create an example usage script."""
    example_content = '''#!/usr/bin/env python3
"""
Example usage script for Nanonets OCR Local Runner
"""

from nanonets_ocr_local import NanonetsOCR
import os

def example_single_image():
    """Example: Process a single image"""
    print("Example: Processing a single image")
    
    # Initialize OCR
    ocr = NanonetsOCR()
    ocr.load_model()
    
    # Replace with your image path
    image_path = "path/to/your/document.jpg"
    
    if os.path.exists(image_path):
        result = ocr.process_image(image_path)
        print("OCR Result:")
        print("=" * 50)
        print(result)
        
        # Save result
        with open("example_result.md", "w", encoding="utf-8") as f:
            f.write(result)
        print("\\nResult saved to: example_result.md")
    else:
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")

def example_batch_processing():
    """Example: Process multiple images"""
    print("Example: Batch processing multiple images")
    
    # Initialize OCR
    ocr = NanonetsOCR()
    ocr.load_model()
    
    # Replace with your image paths
    image_paths = [
        "path/to/document1.jpg",
        "path/to/document2.png",
        "path/to/document3.pdf"  # Note: PDF pages need to be converted to images first
    ]
    
    # Filter existing files
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if existing_paths:
        results = ocr.process_batch(existing_paths, output_dir="batch_results")
        successful = sum(1 for r in results if r["success"])
        print(f"Batch completed: {successful}/{len(results)} images processed successfully")
    else:
        print("No valid image files found. Please update the image_paths list.")

if __name__ == "__main__":
    print("Nanonets OCR Examples")
    print("=" * 30)
    
    print("\\n1. Single image example")
    print("2. Batch processing example")
    
    choice = input("\\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        example_single_image()
    elif choice == "2":
        example_batch_processing()
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
'''
    
    with open("example_usage.py", "w", encoding="utf-8") as f:
        f.write(example_content)
    
    print("\n📝 Created example_usage.py with usage examples")

def main():
    """Main setup function."""
    print("🚀 Nanonets OCR Local Runner Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU availability (if possible)
    gpu_available = check_gpu_availability()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        sys.exit(1)
    
    # Check GPU again after PyTorch installation
    if not gpu_available:
        check_gpu_availability()
    
    # Provide model download info
    download_model_info()
    
    # Create example script
    create_example_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the program: python nanonets_ocr_local.py --interactive")
    print("2. Or check examples: python example_usage.py")
    print("3. For command-line usage: python nanonets_ocr_local.py --help")
    print("\nNote: First run will download the model (~7-8 GB) and may take some time.")

if __name__ == "__main__":
    main()
