# Use a pipeline as a high-level helper
from transformers import pipeline
# install huggingface SAFETENSORS dot.OCR and ollama's llava 8b or qwen3-vl:8b model first to use offline

pipe = pipeline("image-text-to-text", model="tencent/HunyuanOCR")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "ex.jpg"},
            {"type": "text", "text": "What is this"}
        ]
    },
]
pipe(text=messages)