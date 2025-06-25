"""
Nanonets OCR Local Runner
A comprehensive program to run the Nanonets-OCR-s model locally on your machine.

This program provides multiple ways to use the model:
1. Direct transformers inference
2. Batch processing of multiple images
3. Interactive mode for single images
4. Command-line interface
"""

import os
import sys
import argparse
import base64
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

try:
    from PIL import Image
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
    import torch
    import torchvision
except ImportError as e:
    print(f"Error: Missing required dependencies. Please run the following commands:")
    print(f"  pip install -r requirements.txt")
    print(f"  # Or install missing dependencies individually:")
    print(f"  pip install torch torchvision transformers pillow accelerate")
    print(f"\nSpecific error: {e}")
    print(f"\nFor GPU support, visit: https://pytorch.org/get-started/locally/")
    sys.exit(1)


class NanonetsOCR:
    """Main class for running Nanonets OCR model locally."""
    
    def __init__(self, model_path: str = "nanonets/Nanonets-OCR-s", device: str = "auto"):
        """
        Initialize the Nanonets OCR model.
        
        Args:
            model_path: HuggingFace model path
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        print(f"Initializing Nanonets OCR model: {model_path}")
        print(f"Using device: {device}")
        
    def load_model(self):
        """Load the model, processor, and tokenizer."""
        print("Loading model components...")
        
        try:
            # Load model with optimizations
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            self.model.eval()
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU-only mode...")
            
            # Fallback to CPU
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("✓ Model loaded in CPU mode!")
    
    def get_default_prompt(self) -> str:
        """Get the default OCR prompt."""
        return """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    def process_image(self, image_path: str, custom_prompt: Optional[str] = None, max_new_tokens: int = 4096) -> str:
        """
        Process a single image with OCR.
        
        Args:
            image_path: Path to the image file
            custom_prompt: Custom prompt (uses default if None)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Extracted text in markdown format
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        print(f"Processing image: {image_path}")
        
        try:
            # Load and validate image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            prompt = custom_prompt or self.get_default_prompt()
            
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Generate output
            print("Generating OCR output...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            print("✓ OCR processing completed!")
            return output_text[0]
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise
    
    def process_batch(self, image_paths: List[str], output_dir: str = "output", custom_prompt: Optional[str] = None) -> List[dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            custom_prompt: Custom prompt for all images
            
        Returns:
            List of results with metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                # Process image
                start_time = datetime.now()
                text_output = self.process_image(image_path, custom_prompt)
                end_time = datetime.now()
                
                # Prepare result
                result = {
                    "image_path": image_path,
                    "filename": os.path.basename(image_path),
                    "processed_at": start_time.isoformat(),
                    "processing_time_seconds": (end_time - start_time).total_seconds(),
                    "output_text": text_output,
                    "success": True
                }
                
                # Save individual result
                output_filename = f"{Path(image_path).stem}_ocr_result.md"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# OCR Result for {os.path.basename(image_path)}\n\n")
                    f.write(f"**Processed at:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Processing time:** {result['processing_time_seconds']:.2f} seconds\n\n")
                    f.write("## Extracted Content\n\n")
                    f.write(text_output)
                
                print(f"✓ Result saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")
                result = {
                    "image_path": image_path,
                    "filename": os.path.basename(image_path),
                    "processed_at": datetime.now().isoformat(),
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Batch processing completed! Summary saved to: {summary_path}")
        return results


def interactive_mode():
    """Run the program in interactive mode."""
    print("\n🔍 Nanonets OCR - Interactive Mode")
    print("=" * 50)
    
    # Initialize model
    ocr = NanonetsOCR()
    ocr.load_model()
    
    while True:
        print("\nOptions:")
        print("1. Process single image")
        print("2. Process multiple images (batch)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            custom_prompt = input("Enter custom prompt (press Enter for default): ").strip()
            
            try:
                result = ocr.process_image(
                    image_path, 
                    custom_prompt if custom_prompt else None
                )
                print("\n" + "="*80)
                print("OCR RESULT:")
                print("="*80)
                print(result)
                print("="*80)
                
                # Ask to save
                save = input("\nSave result to file? (y/n): ").strip().lower()
                if save == 'y':
                    output_file = f"{Path(image_path).stem}_ocr_result.md"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"✓ Result saved to: {output_file}")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            print("Enter image paths (one per line, empty line to finish):")
            image_paths = []
            while True:
                path = input().strip()
                if not path:
                    break
                image_paths.append(path)
            
            if image_paths:
                output_dir = input("Enter output directory (default: 'output'): ").strip() or "output"
                custom_prompt = input("Enter custom prompt (press Enter for default): ").strip()
                
                try:
                    results = ocr.process_batch(
                        image_paths, 
                        output_dir, 
                        custom_prompt if custom_prompt else None
                    )
                    
                    successful = sum(1 for r in results if r["success"])
                    print(f"\n✓ Batch completed: {successful}/{len(results)} images processed successfully")
                    
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("No image paths provided.")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Run Nanonets OCR model locally")
    parser.add_argument("--image", "-i", help="Single image to process")
    parser.add_argument("--batch", "-b", nargs="+", help="Multiple images to process")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--prompt", "-p", help="Custom prompt")
    parser.add_argument("--max-tokens", "-t", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    if not args.image and not args.batch:
        print("No input provided. Use --interactive for interactive mode or provide --image/--batch")
        parser.print_help()
        return
    
    # Initialize model
    print("🚀 Starting Nanonets OCR Local Runner")
    ocr = NanonetsOCR(device=args.device)
    ocr.load_model()
    
    if args.image:
        # Single image processing
        try:
            result = ocr.process_image(args.image, args.prompt, args.max_tokens)
            
            # Save result
            output_file = f"{Path(args.image).stem}_ocr_result.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"\n✓ Result saved to: {output_file}")
            print("\nOCR Result:")
            print("="*80)
            print(result)
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.batch:
        # Batch processing
        try:
            results = ocr.process_batch(args.batch, args.output, args.prompt)
            successful = sum(1 for r in results if r["success"])
            print(f"\n✓ Batch completed: {successful}/{len(results)} images processed successfully")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
