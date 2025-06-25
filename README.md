# Nanonets OCR Local Runner

A comprehensive Python program to run the powerful **Nanonets-OCR-s** model locally on your machine for advanced document OCR processing.

## Features

✨ **Advanced OCR Capabilities:**
- **LaTeX Equation Recognition** - Converts mathematical equations to LaTeX format
- **Intelligent Image Description** - Describes images within documents
- **Signature Detection** - Isolates signatures with `<signature>` tags
- **Watermark Extraction** - Detects watermarks with `<watermark>` tags
- **Smart Checkbox Handling** - Converts checkboxes to Unicode symbols (☐, ☑, ☒)
- **Complex Table Extraction** - Extracts tables in both markdown and HTML formats

🚀 **Multiple Usage Modes:**
- Interactive mode with user-friendly prompts
- Command-line interface for automation
- Batch processing for multiple images
- Single image processing
- Custom prompts support

🔧 **Technical Features:**
- Automatic GPU detection and optimization
- CPU fallback mode for systems without GPU
- Flash Attention support for faster processing
- Comprehensive error handling and logging
- Progress tracking and timing information

## Requirements

- **Python:** 3.8 or higher
- **Memory:** 8GB RAM minimum (16GB recommended for GPU)
- **Storage:** 10GB free space (for model download)
- **GPU:** CUDA-compatible GPU recommended (optional, CPU mode available)

## Installation

### Quick Setup (Recommended)

1. **Clone or download the files** to your local directory
2. **Run the setup script:**
   ```bash
   python setup.py
   ```

The setup script will:
- Check your Python version
- Install all required dependencies
- Detect GPU availability
- Create example usage files
- Provide next steps

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For GPU users (optional but recommended):**
   ```bash
   pip install flash-attn>=2.0.0
   ```

## Usage

### Interactive Mode (Easiest)

Start the interactive mode for a user-friendly experience:

```bash
python nanonets_ocr_local.py --interactive
```

This mode will guide you through:
- Processing single images
- Batch processing multiple images
- Saving results automatically

### Command Line Usage

#### Process a Single Image
```bash
python nanonets_ocr_local.py --image "path/to/your/document.jpg"
```

#### Batch Process Multiple Images
```bash
python nanonets_ocr_local.py --batch "doc1.jpg" "doc2.png" "doc3.pdf" --output "results"
```

#### Custom Prompt
```bash
python nanonets_ocr_local.py --image "document.jpg" --prompt "Extract only the text, ignore images"
```

#### Advanced Options
```bash
python nanonets_ocr_local.py --image "document.jpg" --max-tokens 8192 --device "cuda"
```

### Programmatic Usage

```python
from nanonets_ocr_local import NanonetsOCR

# Initialize the OCR model
ocr = NanonetsOCR()
ocr.load_model()

# Process a single image
result = ocr.process_image("path/to/document.jpg")
print(result)

# Batch process multiple images
image_paths = ["doc1.jpg", "doc2.png", "doc3.jpg"]
results = ocr.process_batch(image_paths, output_dir="output")
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--interactive` | - | Run in interactive mode | - |
| `--image` | `-i` | Single image to process | - |
| `--batch` | `-b` | Multiple images to process | - |
| `--output` | `-o` | Output directory | `output` |
| `--prompt` | `-p` | Custom prompt | Default OCR prompt |
| `--max-tokens` | `-t` | Maximum tokens to generate | `4096` |
| `--device` | - | Device to use (auto/cpu/cuda) | `auto` |

## Supported File Formats

- **Images:** JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Documents:** Any image format containing text, tables, equations, or graphics

> **Note:** For PDF files, convert them to images first using tools like `pdf2image` or similar.

## Output Formats

The OCR output includes:
- **Plain text** extracted naturally
- **Tables** in HTML format
- **Equations** in LaTeX format
- **Images** with descriptions in `<img>` tags
- **Signatures** in `<signature>` tags
- **Watermarks** in `<watermark>` tags
- **Page numbers** in `<page_number>` tags
- **Checkboxes** as Unicode symbols (☐ ☑ ☒)

## Performance Tips

### For GPU Users
- Ensure CUDA is properly installed
- Install `flash-attn` for faster processing:
  ```bash
  pip install flash-attn>=2.0.0
  ```
- Use `--device cuda` to force GPU usage

### For CPU Users
- Expect slower processing times (2-5x slower than GPU)
- Reduce `--max-tokens` for faster processing
- Process images in smaller batches

### Memory Optimization
- Close other applications to free up RAM
- For large batches, process in smaller chunks
- Use smaller `--max-tokens` values if memory is limited

## Model Information

- **Model:** nanonets/Nanonets-OCR-s
- **Base:** Qwen2.5-VL-3B-Instruct
- **Size:** 3.75B parameters (~7-8 GB download)
- **Cache Location:** `~/.cache/huggingface/transformers/`

### First Run
The first run will download the model automatically:
- **Download size:** ~7-8 GB
- **Time:** 10-30 minutes (depending on internet speed)
- **Storage:** Model cached locally for future use

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce tokens or use CPU
python nanonets_ocr_local.py --image "doc.jpg" --max-tokens 2048 --device cpu
```

**2. GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Slow Processing**
- Ensure GPU drivers are updated
- Install flash-attn for GPU acceleration
- Use smaller images or reduce max-tokens

**4. Module Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help

1. **Check the setup:** `python setup.py`
2. **Run examples:** `python example_usage.py`
3. **Test installation:** `python nanonets_ocr_local.py --help`

## Examples

### Example 1: Academic Paper Processing
```bash
python nanonets_ocr_local.py --image "research_paper.pdf" --max-tokens 8192
```
Perfect for extracting equations, tables, and formatted text from academic documents.

### Example 2: Form Processing
```bash
python nanonets_ocr_local.py --image "filled_form.jpg" --prompt "Extract form fields and checkbox states"
```
Ideal for processing filled forms with checkboxes and structured data.

### Example 3: Batch Document Processing
```bash
python nanonets_ocr_local.py --batch document1.jpg document2.png document3.jpg --output "processed_docs"
```
Process multiple documents efficiently with organized output.

## Contributing

Feel free to submit issues, feature requests, or improvements to this OCR runner implementation.

## License

This implementation is provided as-is for educational and research purposes. Please refer to the original Nanonets model license for commercial usage terms.

## Acknowledgments

- **Nanonets** for the amazing OCR model
- **HuggingFace** for the transformers library
- **Qwen** for the base vision-language model

---

**Ready to get started?** Run `python setup.py` to begin! 🚀
