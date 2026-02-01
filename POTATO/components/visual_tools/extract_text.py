# extract_text.py
from ollama import chat
from pdf2image import convert_from_path

def main():
    """Main function to process file with Ollama chat API."""
    # Get file path from user
    file_path = input("Enter the absolute path or url to your PDF or image file: ")
    images = convert_from_path(file_path, dpi=300)
    # temporary save in .tmp
    for i, image in enumerate(images):
        image.save(f'.tmp/img/pdfextracted/{file_path.split("/")[-1]}_page_{i}.jpg', 'JPEG')
    print(f"Processing file with Ollama chat API: {file_path}")
    # continue later
    try:
        # Use Ollama's chat API exactly as shown in documentation
        response = chat(
            model='qwen3-vl:8b', # or 'llava3.1:8b' or others that support images. transformers pipeline with dot.ocr safetensors can be used for offline too
            messages=[
                {
                    'role': 'user',
                    'content': 'What is in this image? Be concise.',
                    'images': [file_path],
                }
            ],
        )
        print("AI Analysis:")
        print(response.message.content)
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()