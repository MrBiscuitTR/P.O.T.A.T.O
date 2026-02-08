# extract_text.py
"""
Text and content extraction from PDFs and images using OCR vision models.
Converts PDF pages to base64 images in memory - no temp files.
Supports parallel processing for faster batch OCR.
"""
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import chat

# Try to import fitz (PyMuPDF)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Default comprehensive extraction prompt
DEFAULT_EXTRACTION_PROMPT = """Analyze this document/image and extract ALL information:

1. **TEXT**: Extract all readable text exactly as written, preserving paragraphs and structure.

2. **TABLES**: Convert any tables to Markdown format like:
   | Header 1 | Header 2 |
   |----------|----------|
   | Cell 1   | Cell 2   |

3. **MATH & FORMULAS**: Extract all mathematical expressions using LaTeX notation:
   - Inline math: $equation$
   - Block math: $$equation$$
   - Example: The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$

4. **IMAGES & DIAGRAMS**: Describe any images, charts, graphs, or diagrams in detail:
   - What does it show?
   - Key data points or labels
   - Colors and visual elements
   - Any text within the image

5. **STRUCTURE**: Note headings, bullet points, numbered lists, and document structure.

Return everything in a clean, organized format. Do not skip any content."""


def image_file_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def pdf_to_base64_images(pdf_path: str, max_pages: int = 20, dpi_scale: float = 2.0) -> list:
    """
    Convert PDF pages to base64 encoded PNG images in memory.
    No temp files are created.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to convert
        dpi_scale: Scale factor for rendering (2.0 = 144 DPI)
    
    Returns:
        List of base64 encoded PNG strings
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    base64_images = []
    doc = fitz.open(pdf_path)
    page_count = min(doc.page_count, max_pages)
    
    for page_num in range(page_count):
        page = doc[page_num]
        # Render at higher resolution for better OCR
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes in memory
        png_bytes = pix.tobytes("png")
        b64_str = base64.b64encode(png_bytes).decode('utf-8')
        base64_images.append(b64_str)
    
    doc.close()
    return base64_images


def describe_image_with_vision(image_base64: str, model: str = 'llava:7b', prompt: str = None) -> dict:
    """
    Send a base64 image to a vision model and get description.
    
    Args:
        image_base64: Base64 encoded image string
        model: Ollama vision model to use
        prompt: Custom prompt
    
    Returns:
        dict: {'success': bool, 'text': str, 'error': str}
    """
    if prompt is None:
        prompt = DEFAULT_EXTRACTION_PROMPT
    
    try:
        response = chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64],
            }],
            options={
                'keep_alive': '30s'  # Unload model after 30 seconds of inactivity
            }
        )
        
        text = response.message.content if response.message else ''
        return {'success': True, 'text': text, 'error': None}
        
    except Exception as e:
        return {'success': False, 'text': '', 'error': str(e)}


def describe_images_batch(image_base64_list: list, model: str = 'llava:7b', prompt: str = None, 
                          parallel: int = 3, source_names: list = None) -> list:
    """
    Describe multiple images with a vision model - supports parallel processing.
    
    Args:
        image_base64_list: List of base64 encoded images
        model: Vision model to use
        prompt: Custom prompt
        parallel: Number of concurrent requests (default 3, set to 1 for sequential)
        source_names: Optional list of source filenames/identifiers for each image
    
    Returns:
        List of dicts: [{'index': int, 'source': str, 'text': str, 'success': bool}, ...]
    """
    if prompt is None:
        prompt = DEFAULT_EXTRACTION_PROMPT
    
    # Build list of tasks with metadata
    tasks = []
    for i, img_b64 in enumerate(image_base64_list):
        source = source_names[i] if source_names and i < len(source_names) else f"Image {i + 1}"
        tasks.append({
            'index': i,
            'source': source,
            'base64': img_b64
        })
    
    results = [None] * len(tasks)  # Pre-allocate to maintain order
    
    def process_single_image(task):
        """Process a single image and return result with metadata"""
        idx = task['index']
        source = task['source']
        img_b64 = task['base64']
        
        try:
            response = chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }],
            )
            
            text = response.message.content if response.message else ''
            return {
                'index': idx,
                'source': source,
                'text': text,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'index': idx,
                'source': source,
                'text': '',
                'success': False,
                'error': str(e)
            }
    
    if parallel <= 1 or len(tasks) == 1:
        # Sequential processing
        for task in tasks:
            result = process_single_image(task)
            results[result['index']] = result
    else:
        # Parallel processing with ThreadPoolExecutor
        # Ollama handles concurrent requests to the same loaded model
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                result = future.result()
                results[result['index']] = result
    
    return results


def describe_images_batch_legacy(image_base64_list: list, model: str = 'llava:7b', prompt: str = None) -> list:
    """
    Legacy function - returns list of description strings for backwards compatibility.
    """
    results = describe_images_batch(image_base64_list, model=model, prompt=prompt, parallel=3)
    
    descriptions = []
    for r in results:
        if r['success']:
            descriptions.append(f"--- {r['source']} ---\n{r['text']}")
        else:
            descriptions.append(f"--- {r['source']} ---\n[Error: {r['error']}]")
    
    return descriptions


def extract_pdf_content(pdf_path: str, model: str = 'llava:7b', max_pages: int = 20, 
                        parallel: int = 3) -> dict:
    """
    Extract all content from PDF using vision model with parallel processing.
    Converts pages to base64 in memory, sends to model, returns descriptions.
    
    Args:
        pdf_path: Path to PDF
        model: Vision model for OCR
        max_pages: Max pages to process
        parallel: Number of concurrent OCR requests (default 3)
    
    Returns:
        dict: {'success': bool, 'content': str, 'pages': int, 'page_results': list, 'error': str}
    """
    try:
        # Get PDF filename for source tracking
        pdf_name = os.path.basename(pdf_path)
        
        # Convert PDF to base64 images
        base64_images = pdf_to_base64_images(pdf_path, max_pages=max_pages)
        
        if not base64_images:
            return {'success': False, 'content': '', 'pages': 0, 'error': 'No pages converted'}
        
        # Build source names for each page
        source_names = [f"{pdf_name} - Page {i+1}" for i in range(len(base64_images))]
        
        # Get descriptions for all pages with parallel processing
        print(f"[PDF-OCR] Processing {len(base64_images)} pages from {pdf_name} (parallel={parallel})")
        results = describe_images_batch(
            base64_images, 
            model=model, 
            parallel=parallel,
            source_names=source_names
        )
        
        # Build content string maintaining page order
        content_parts = []
        for r in results:
            if r['success']:
                content_parts.append(f"--- {r['source']} ---\n{r['text']}")
            else:
                content_parts.append(f"--- {r['source']} ---\n[Error: {r['error']}]")
        
        return {
            'success': True,
            'content': '\n\n'.join(content_parts),
            'pages': len(results),
            'page_results': results,  # Detailed per-page results
            'error': None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'content': '', 'pages': 0, 'error': str(e)}


# Keep old function names for compatibility
def extract_text_from_image(image_path: str, model: str = 'llava:7b', prompt: str = None) -> dict:
    """Extract content from an image file."""
    try:
        b64 = image_file_to_base64(image_path)
        result = describe_image_with_vision(b64, model=model, prompt=prompt)
        return result
    except Exception as e:
        return {'success': False, 'text': '', 'error': str(e)}


def extract_text_from_pdf(pdf_path: str, model: str = 'llava:7b', max_pages: int = 20, 
                          temp_dir: str = None, prompt: str = None) -> dict:
    """Extract content from PDF. temp_dir is ignored - we use memory now."""
    result = extract_pdf_content(pdf_path, model=model, max_pages=max_pages)
    # Adapt return format
    return {
        'success': result['success'],
        'text': result.get('content', ''),
        'pages': result.get('pages', 0),
        'error': result.get('error')
    }


def fast_extract_pdf_text(pdf_path: str, max_pages: int = None) -> dict:
    """
    Fast PDF text extraction using PyMuPDF (fitz) if available.
    Falls back to reading page text with PyPDF2 if fitz is not installed.

    Returns:
        dict: {'success': bool, 'text': str, 'pages': int, 'error': str}
    """
    try:
        # Prefer PyMuPDF for fast, structured text extraction
        if HAS_PYMUPDF:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            if max_pages is not None:
                page_count = min(page_count, max_pages)

            parts = []
            for i in range(page_count):
                page = doc[i]
                # get_text('text') returns plain text preserving newlines
                txt = page.get_text('text')
                parts.append(f"--- Page {i+1} ---\n" + txt)

            doc.close()
            return {'success': True, 'text': '\n\n'.join(parts), 'pages': page_count, 'error': None}

        # Fallback: try PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            if max_pages is not None:
                page_count = min(page_count, max_pages)
            parts = []
            for i in range(page_count):
                page = reader.pages[i]
                txt = page.extract_text() or ''
                parts.append(f"--- Page {i+1} ---\n" + txt)
            return {'success': True, 'text': '\n\n'.join(parts), 'pages': page_count, 'error': None}
        except Exception:
            return {'success': False, 'text': '', 'pages': 0, 'error': 'No suitable PDF text extractor available (install pymupdf or PyPDF2)'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'text': '', 'pages': 0, 'error': str(e)}


def batch_process_files(files: list, model: str = 'llava:7b', parallel: int = 3, 
                        max_pdf_pages: int = 20) -> dict:
    """
    Process multiple files (images and PDFs) with parallel OCR.
    
    Args:
        files: List of dicts with {'path': str, 'filename': str, 'base64': str (optional)}
               - For images: can provide base64 directly or path
               - For PDFs: must provide path (will be converted to images)
        model: Vision model for OCR
        parallel: Number of concurrent requests
        max_pdf_pages: Max pages per PDF
    
    Returns:
        dict: {
            'success': bool,
            'results': [{'filename': str, 'content': str, 'success': bool, 'error': str}, ...],
            'total_content': str,
            'error': str
        }
    """
    all_images = []  # List of {'base64': str, 'source': str}
    
    # First pass: collect all images (expanding PDFs to pages)
    for file_info in files:
        filename = file_info.get('filename', 'unknown')
        path = file_info.get('path')
        base64_data = file_info.get('base64')
        
        ext = os.path.splitext(filename)[1].lower() if filename else ''
        
        if ext == '.pdf' and path:
            # PDF: convert pages to images
            try:
                page_images = pdf_to_base64_images(path, max_pages=max_pdf_pages)
                for i, page_b64 in enumerate(page_images):
                    all_images.append({
                        'base64': page_b64,
                        'source': f"{filename} - Page {i+1}",
                        'original_file': filename
                    })
                print(f"[BATCH] {filename}: {len(page_images)} pages queued")
            except Exception as e:
                print(f"[BATCH] {filename}: Failed to convert PDF - {e}")
                all_images.append({
                    'base64': None,
                    'source': filename,
                    'original_file': filename,
                    'error': str(e)
                })
        elif base64_data:
            # Image with base64 already provided
            all_images.append({
                'base64': base64_data,
                'source': filename,
                'original_file': filename
            })
            print(f"[BATCH] {filename}: Image queued (base64 provided)")
        elif path:
            # Image from path
            try:
                b64 = image_file_to_base64(path)
                all_images.append({
                    'base64': b64,
                    'source': filename,
                    'original_file': filename
                })
                print(f"[BATCH] {filename}: Image queued (from path)")
            except Exception as e:
                print(f"[BATCH] {filename}: Failed to load - {e}")
                all_images.append({
                    'base64': None,
                    'source': filename,
                    'original_file': filename,
                    'error': str(e)
                })
    
    # Filter out failed loads
    valid_images = [img for img in all_images if img.get('base64')]
    failed_images = [img for img in all_images if not img.get('base64')]
    
    if not valid_images:
        return {
            'success': False,
            'results': [],
            'total_content': '',
            'error': 'No valid images to process'
        }
    
    # Process all images in parallel
    print(f"[BATCH] Processing {len(valid_images)} images with {parallel} workers...")
    
    image_b64_list = [img['base64'] for img in valid_images]
    source_names = [img['source'] for img in valid_images]
    
    ocr_results = describe_images_batch(
        image_b64_list,
        model=model,
        parallel=parallel,
        source_names=source_names
    )
    
    # Group results by original file
    file_results = {}
    for i, result in enumerate(ocr_results):
        original_file = valid_images[i]['original_file']
        if original_file not in file_results:
            file_results[original_file] = {
                'filename': original_file,
                'pages': [],
                'success': True,
                'errors': []
            }
        
        if result['success']:
            file_results[original_file]['pages'].append({
                'source': result['source'],
                'text': result['text']
            })
        else:
            file_results[original_file]['errors'].append(result['error'])
    
    # Add failed files
    for img in failed_images:
        original_file = img['original_file']
        if original_file not in file_results:
            file_results[original_file] = {
                'filename': original_file,
                'pages': [],
                'success': False,
                'errors': [img.get('error', 'Unknown error')]
            }
    
    # Build final results
    final_results = []
    content_parts = []
    
    for filename, data in file_results.items():
        if data['pages']:
            content = '\n\n'.join([
                f"--- {p['source']} ---\n{p['text']}" 
                for p in data['pages']
            ])
            final_results.append({
                'filename': filename,
                'content': content,
                'pages': len(data['pages']),
                'success': True,
                'error': None
            })
            content_parts.append(f"=== {filename} ===\n{content}")
        else:
            final_results.append({
                'filename': filename,
                'content': '',
                'pages': 0,
                'success': False,
                'error': '; '.join(data['errors'])
            })
    
    return {
        'success': True,
        'results': final_results,
        'total_content': '\n\n'.join(content_parts),
        'error': None
    }


def main():
    """Interactive CLI for testing."""
    file_path = input("Enter path to PDF or image: ").strip()
    model = input("Vision model (default: llava:7b): ").strip() or 'llava:7b'
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        result = extract_pdf_content(file_path, model=model)
        if result['success']:
            print(f"\n{'='*50}\nExtracted from {result['pages']} pages:\n{'='*50}")
            print(result['content'])
        else:
            print(f"Error: {result['error']}")
    else:
        b64 = image_file_to_base64(file_path)
        result = describe_image_with_vision(b64, model=model)
        if result['success']:
            print(f"\n{'='*50}\nExtracted:\n{'='*50}")
            print(result['text'])
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()