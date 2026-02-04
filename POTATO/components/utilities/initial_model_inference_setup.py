"""
Auto-detect thinking tags and output formats for different LLM models.
Runs a test inference and uses qwen2.5-coder:7b to analyze the output structure.
"""

import ollama
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from POTATO.main import MODEL_INFO_PATH, load_model_info, save_model_info


def detect_model_tags(model_name):
    """
    Run a test inference on the model and detect its tag patterns.
    Uses qwen2.5-coder:7b to analyze the output structure.
    
    Returns dict with tag patterns:
    {
        'thinking_tags': {'open': '<think>', 'close': '</think>'},
        'tool_call_format': 'xml|json|native',
        'supports_streaming_thinking': True/False
    }
    """
    print(f"\n[DETECTION] Testing {model_name}...")
    
    # Test prompt designed to elicit thinking and tool usage patterns
    test_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. For complex queries, show your reasoning process before giving your final answer."
        },
        {
            "role": "user",
            "content": "Explain step by step: What is 15 * 23 + 47?"
        }
    ]
    
    # Run inference and capture raw output
    try:
        response = ollama.chat(
            model=model_name,
            messages=test_messages,
            stream=False
        )
        
        raw_output = response.get('message', {}).get('content', '')
        
        if not raw_output:
            print(f"[WARNING] No output from {model_name}")
            return get_default_tags()
        
        print(f"[SAMPLE OUTPUT]\n{raw_output[:500]}...\n")
        
        # Use qwen2.5-coder:7b to analyze the structure
        analysis_prompt = f"""Analyze this LLM output and determine its tag structure:

OUTPUT:
```
{raw_output}
```

Identify:
1. Thinking/reasoning tags (e.g., <think></think>, <reasoning></reasoning>, or none)
2. Whether thinking appears inline or separate
3. Any special formatting patterns

Respond ONLY with valid JSON in this exact format:
{{
    "thinking_tags": {{"open": "<tag>", "close": "</tag>"}},
    "has_inline_thinking": true/false,
    "notes": "brief observation"
}}

If no thinking tags found, use: {{"open": "", "close": ""}}"""

        analysis_response = ollama.chat(
            model="qwen2.5-coder:7b",
            messages=[{"role": "user", "content": analysis_prompt}],
            stream=False
        )
        
        analysis_text = analysis_response.get('message', {}).get('content', '')
        
        # Extract JSON from response (may be wrapped in markdown)
        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            analysis_json = json.loads(analysis_text[json_start:json_end])
            
            # Validate and structure the result
            thinking_tags = analysis_json.get('thinking_tags', {'open': '', 'close': ''})
            
            result = {
                'thinking_tags': thinking_tags,
                'has_inline_thinking': analysis_json.get('has_inline_thinking', False),
                'supports_streaming_thinking': bool(thinking_tags.get('open')),
                'detection_notes': analysis_json.get('notes', ''),
                'detection_date': datetime.now().isoformat()
            }
            
            print(f"[DETECTED] Tags: {thinking_tags}")
            return result
        else:
            print(f"[WARNING] Could not parse analysis JSON")
            return get_default_tags()
            
    except Exception as e:
        print(f"[ERROR] Detection failed for {model_name}: {e}")
        return get_default_tags()


def get_default_tags():
    """Return default tag structure"""
    return {
        'thinking_tags': {'open': '<think>', 'close': '</think>'},
        'has_inline_thinking': False,
        'supports_streaming_thinking': False,
        'detection_notes': 'Using default tags',
        'detection_date': datetime.now().isoformat()
    }


def update_model_with_tags(model_name):
    """Detect tags for a model and update modelinfos.json"""
    
    # Load current model info
    model_infos = load_model_info()
    
    # Create entry if model doesn't exist
    if model_name not in model_infos:
        print(f"[INFO] Creating new entry for {model_name}")
        model_infos[model_name] = {
            "can_think": "unknown",
            "tool_calls": False,
            "vision": "unknown"
        }
    
    # Detect tags
    tag_info = detect_model_tags(model_name)
    
    # Update model info
    model_infos[model_name].update(tag_info)
    
    # Save back to file
    save_model_info(model_infos)
    
    print(f"[SUCCESS] Updated {model_name} with tag info")
    print(json.dumps(tag_info, indent=2))
    
    # Unload detection model to free VRAM
    print("[CLEANUP] Unloading qwen2.5-coder:7b...")
    try:
        ollama.chat(model="qwen2.5-coder:7b", messages=[], keep_alive=0)
        print("[CLEANUP] Detection model unloaded")
    except Exception as e:
        print(f"[WARNING] Could not unload detection model: {e}")
    
    return True


def detect_all_models():
    """Run detection on all models in modelinfos.json"""
    model_infos = load_model_info()
    
    print(f"\n{'='*60}")
    print(f"Starting tag detection for {len(model_infos)} models")
    print(f"{'='*60}")
    
    for model_name in model_infos.keys():
        # Skip if already has tag info
        if 'thinking_tags' in model_infos[model_name]:
            print(f"[SKIP] {model_name} already has tag info")
            continue
        
        update_model_with_tags(model_name)
        print()  # Blank line between models
    
    print(f"\n{'='*60}")
    print("Detection complete!")
    print(f"{'='*60}\n")
    
    # Unload detection model after all detections
    print("[CLEANUP] Unloading qwen2.5-coder:7b...")
    try:
        ollama.chat(model="qwen2.5-coder:7b", messages=[], keep_alive=0)
        print("[CLEANUP] Detection model unloaded")
    except Exception as e:
        print(f"[WARNING] Could not unload detection model: {e}")


def detect_single_model(model_name):
    """Run detection on a single model"""
    return update_model_with_tags(model_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect thinking tags for LLM models')
    parser.add_argument('--model', type=str, help='Specific model to test')
    parser.add_argument('--all', action='store_true', help='Test all models')
    
    args = parser.parse_args()
    
    if args.all:
        detect_all_models()
    elif args.model:
        detect_single_model(args.model)
    else:
        print("Usage:")
        print("  python initial_model_inference_setup.py --model <model_name>")
        print("  python initial_model_inference_setup.py --all")
