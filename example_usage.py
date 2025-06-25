#!/usr/bin/env python3
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
        print("\nResult saved to: example_result.md")
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
    
    print("\n1. Single image example")
    print("2. Batch processing example")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        example_single_image()
    elif choice == "2":
        example_batch_processing()
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
