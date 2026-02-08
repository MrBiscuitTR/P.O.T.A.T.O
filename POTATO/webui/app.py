import os
import json
import sys
import time
import glob
import uuid
import psutil
import subprocess
import atexit
import signal
import re
import ollama
import base64
import shutil
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
from werkzeug.utils import secure_filename
from pathlib import Path

# --- IMPORTS ---
from POTATO.main import simple_stream_test 
# Import audioflow functions only (no model loading at import time)
from POTATO.components.vocal_tools.audioflow import listen_stt
from POTATO.components.local_tools.embed import (
    embed_to_weaviate, 
    embed_ocr_content, 
    query_weaviate, 
    delete_chat_collection,
    delete_file_from_collection,
    list_embedded_files,
    check_weaviate_connection,
    list_collections
)
from POTATO.components.utilities.get_system_info import json_get_instant_system_info
from POTATO.components.visual_tools.extract_text import (
    extract_text_from_pdf,
    extract_text_from_image,
    fast_extract_pdf_text,
)

# TTS models will be loaded on-demand, not at import
_tts_turbo_model = None
_tts_multilingual_model = None

# Generation control
_stop_generation = False
_current_model = None

# Model info cache
_model_info_cache = None
_model_info_cache_time = 0

# --- TOOL REGISTRY ---
def potatool_rag_fetch(query, session_id=None, top_k=5):
    """Tool: fetch relevant RAG snippets from vector DB for a session/query.
    Returns a list of dicts: {source, content, score}
    """
    try:
        results = query_weaviate(query, chat_id=session_id, limit=top_k)
        # Normalize results
        normalized = []
        for r in (results or []):
            normalized.append({
                'source': r.get('source_file') or r.get('source') or 'unknown',
                'content': r.get('content')[:2000] if r.get('content') else '',
                'score': r.get('score', None)
            })
        return {'success': True, 'results': normalized}
    except Exception as e:
        return {'success': False, 'error': str(e)}


TOOL_REGISTRY = {
    "get_current_weather": lambda location: f"The weather in {location} is 22°C and Sunny.",
    "potatool_rag_fetch": potatool_rag_fetch,
}

app = Flask(__name__)

# Suppress logging for /api/system_stats endpoint
import logging
log = logging.getLogger('werkzeug')

class NoStatsFilter(logging.Filter):
    def filter(self, record):
        return '/api/system_stats' not in record.getMessage()

log.addFilter(NoStatsFilter())

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, 'config.json')
DATA_DIR = os.path.join(BASE_DIR, '.data')
USER_SETTINGS_PATH = os.path.join(DATA_DIR, 'usersettings.json')
CHATS_DIR = os.path.join(DATA_DIR, 'chats')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
TEMP_DIR = os.path.join(BASE_DIR, '.temp', 'uploads')  # For RAG temp files
MODEL_INFO_PATH = os.path.join(DATA_DIR, '.modelinfos.json')

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)

# Initialize user settings from config.json if not exists
if not os.path.exists(USER_SETTINGS_PATH):
    try:
        with open(SETTINGS_PATH, 'r') as f:
            default_config = json.load(f)
        with open(USER_SETTINGS_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
    except Exception as e:
        print(f"Error initializing user settings: {e}")

# --- HELPERS ---
def deep_merge(base, override):
    """Deep merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_settings():
    """Load user settings, fallback to config.json"""
    settings = {}
    # Load default config first
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
    
    # Deep merge with user settings (user settings override defaults)
    if os.path.exists(USER_SETTINGS_PATH):
        try:
            with open(USER_SETTINGS_PATH, 'r') as f:
                user_settings = json.load(f)
                settings = deep_merge(settings, user_settings)
        except Exception as e:
            print(f"Error loading user settings: {e}")
    
    return settings

def save_user_settings(new_settings):
    """Save entire settings object to usersettings.json"""
    try:
        with open(USER_SETTINGS_PATH, 'w') as f:
            json.dump(new_settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving user settings: {e}")
        return False

def reset_user_settings():
    """Reset user settings to config.json defaults"""
    try:
        with open(SETTINGS_PATH, 'r') as f:
            default_config = json.load(f)
        with open(USER_SETTINGS_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error resetting settings: {e}")
        return False

def save_chat_session(session_id, messages, title=None, is_voice_chat=False):
    """Save chat session to file"""
    # Generate session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Add vox_ prefix for voice chats if not already present
    if is_voice_chat and not session_id.startswith('vox_'):
        session_id = f"vox_{session_id}"
    
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if not title:
        # Check if chat already has a permanent title (not a placeholder)
        existing_title = None
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing_title = json.load(f).get('title')
            except:
                pass
        
        # Use existing title if it's not a placeholder (doesn't end with "...")
        if existing_title and not existing_title.endswith('...'):
            title = existing_title
        else:
            # Generate AI title only if we have both user message AND assistant response
            user_msg = None
            ai_response = None
            
            for m in messages:
                if m['role'] == 'user' and m['content'] and not user_msg:
                    user_msg = m['content']
                elif m['role'] == 'assistant' and m['content'] and not ai_response:
                    ai_response = m['content']
                
                if user_msg and ai_response:
                    break
            
            # Only generate AI title if we have BOTH messages (not just user message)
            if user_msg and ai_response:
                try:
                    # Use qwen2.5-coder:7b to generate a concise title
                    import ollama
                    
                    context = f"User: {user_msg[:200]}"
                    context += f"\nAssistant: {ai_response[:200]}"
                    
                    response = ollama.chat(
                        model='qwen2.5-coder:7b',
                        messages=[{
                            'role': 'system',
                            'content': 'Generate a short chat title (max 7 words, no quotes). Only output the title, nothing else. Be specific and descriptive.'
                        }, {
                            'role': 'user',
                            'content': context
                        }],
                        options={'num_predict': 30}  # Limit response length
                    )
                    title = response['message']['content'].strip().strip('"\'')
                    
                    # Truncate if too long
                    words = title.split()
                    if len(words) > 7:
                        title = ' '.join(words[:7])
                    
                    # Unload qwen2.5-coder:7b to free VRAM
                    print(f"[CHAT] Generated title: {title}")
                    print("[CHAT] Unloading qwen2.5-coder:7b...")
                    ollama.chat(model='qwen2.5-coder:7b', messages=[], keep_alive=0)
                    print("[CHAT] Title generator unloaded")
                    
                except Exception as e:
                    print(f"[CHAT] Error generating AI title: {e}")
                    # Fallback to placeholder or existing
                    if existing_title:
                        title = existing_title
                    elif user_msg:
                        title = user_msg[:27] + "..." if len(user_msg) > 27 else user_msg
            else:
                # Don't have both messages yet - use placeholder or existing
                if existing_title:
                    title = existing_title
                elif user_msg:
                    title = user_msg[:27] + "..." if len(user_msg) > 27 else user_msg
        
        if not title:
            title = "New Session"
    
    data = {
        "id": session_id,
        "title": title,
        "last_updated": time.time(),
        "messages": messages
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return data

def load_chat_session(session_id):
    """Load chat session from file"""
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def get_all_chats(search_query=None, voice_only=False, text_only=False):
    """Get all chat sessions, optionally filtered by search query and type"""
    files = glob.glob(os.path.join(CHATS_DIR, "*.json"))
    chats = []
    for p in files:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            
            # Filter by chat type (voice vs text)
            filename = os.path.basename(p)
            is_voice = filename.startswith('vox_')
            
            if voice_only and not is_voice:
                continue
            if text_only and is_voice:
                continue
            
            # Filter by search query
            if search_query:
                q = search_query.lower()
                in_title = q in data.get('title', '').lower()
                in_content = any(q in m.get('content', '').lower() for m in data.get('messages', []))
                if not in_title and not in_content:
                    continue
            
            chats.append(data)
        except Exception as e:
            print(f"Error loading chat {p}: {e}")
    
    chats.sort(key=lambda x: x.get('last_updated', 0), reverse=True)
    return chats

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
            return models
        else:
            return []
    except Exception as e:
        print(f"Error getting ollama models: {e}")
        return []

def stop_ollama_inference(model_name=None):
    """Stop ongoing Ollama inference and unload from VRAM"""
    try:
        import requests
        
        if model_name:
            # Use API call with keep_alive=0 to force unload
            try:
                requests.post('http://localhost:11434/api/generate', 
                             json={'model': model_name, 'keep_alive': 0},
                             timeout=5)
                print(f"Unloaded {model_name} from VRAM")
            except Exception as e:
                print(f"Failed to unload {model_name}: {e}")
                # Fallback to CLI
                subprocess.run(['ollama', 'stop', model_name], check=False, timeout=5)
        else:
            # Stop all models - get list of running models first
            try:
                ps_result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
                if ps_result.returncode == 0:
                    lines = ps_result.stdout.strip().split('\\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            model = line.split()[0]
                            try:
                                requests.post('http://localhost:11434/api/generate',
                                            json={'model': model, 'keep_alive': 0},
                                            timeout=5)
                                print(f"Unloaded {model} from VRAM")
                            except Exception as e:
                                print(f"Failed to unload {model}: {e}")
            except Exception as e:
                print(f"Error getting model list: {e}")
                subprocess.run(['ollama', 'stop'], check=False, timeout=5)
        return True
    except Exception as e:
        print(f"Error stopping ollama: {e}")
        return False

def read_folder_contents(folder_path):
    """Read all text files from a folder for RAG context"""
    contents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    # Supported text file extensions
    extensions = ['.txt', '.md', '.py', '.js', '.json', '.html', '.css', '.xml']
    
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    contents.append({
                        'file': str(file_path),
                        'content': content
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return contents


# --- VISION & OCR HELPERS ---

def load_model_info():
    """Load model info from .modelinfos.json with caching"""
    global _model_info_cache, _model_info_cache_time
    
    # Cache for 60 seconds
    if _model_info_cache and (time.time() - _model_info_cache_time) < 60:
        return _model_info_cache
    
    try:
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                _model_info_cache = json.load(f)
                _model_info_cache_time = time.time()
                return _model_info_cache
    except Exception as e:
        print(f"Error loading model info: {e}")
    
    return {}


def save_model_info(model_info):
    """Save model info to .modelinfos.json and update cache"""
    global _model_info_cache, _model_info_cache_time
    
    try:
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f, indent=2)
        _model_info_cache = model_info
        _model_info_cache_time = time.time()
    except Exception as e:
        print(f"Error saving model info: {e}")


def check_model_is_vision(model_name):
    """
    Check if a model has vision capabilities.
    Uses cached value from .modelinfos.json if available.
    Only queries Ollama if vision is unknown, then caches the result.
    
    Returns:
        dict: {'is_vision': bool, 'known': bool}
    """
    model_info = load_model_info()
    
    # Clean model name (remove tags like :latest)
    clean_name = model_name.split(':')[0] if ':' in model_name else model_name
    
    # Check exact match in cache first
    if model_name in model_info:
        vision = model_info[model_name].get('vision')
        if vision is True:
            return {'is_vision': True, 'known': True}
        elif vision is False:
            return {'is_vision': False, 'known': True}
        # vision is "unknown" or not set - need to check
    
    # Check clean name in cache
    for key, info in model_info.items():
        if clean_name in key or key in clean_name:
            vision = info.get('vision')
            if vision is True:
                return {'is_vision': True, 'known': True}
            elif vision is False:
                return {'is_vision': False, 'known': True}
    
    # Not in cache or unknown - query Ollama and cache the result
    is_vision = None
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True,
            text=True,
            timeout=5,
            encoding='utf-8',
            errors='replace'  # Handle encoding issues on Windows
        )
        if result.returncode == 0 and result.stdout:
            output = result.stdout.lower()
            # Check the Capabilities section for "vision"
            if 'capabilities' in output:
                lines = output.split('\n')
                in_capabilities = False
                for line in lines:
                    if 'capabilities' in line:
                        in_capabilities = True
                        continue
                    if in_capabilities:
                        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                            break
                        if 'vision' in line:
                            is_vision = True
                            break
                if is_vision is None:
                    is_vision = False  # Has capabilities but no vision
    except Exception as e:
        pass  # Silently fail, will use pattern matching
    
    # If ollama check succeeded, cache the result
    if is_vision is not None:
        # Update cache
        if model_name not in model_info:
            model_info[model_name] = {}
        model_info[model_name]['vision'] = is_vision
        save_model_info(model_info)
        print(f"[VISION] Cached: {model_name} vision={is_vision}")
        return {'is_vision': is_vision, 'known': True}
    
    # Fall back to pattern matching
    vision_patterns = ['vl', 'vision', 'llava', 'bakllava', 'moondream', 'minicpm-v', 'deepseek-ocr']
    name_lower = model_name.lower()
    for pattern in vision_patterns:
        if pattern in name_lower:
            return {'is_vision': True, 'known': False}
    
    return {'is_vision': False, 'known': False}


def get_available_vision_models():
    """Get list of vision-capable models that are installed"""
    installed_models = get_ollama_models()
    vision_models = []
    
    for model in installed_models:
        check = check_model_is_vision(model)
        if check['is_vision']:
            vision_models.append(model)
    
    return vision_models


def check_ocr_model_installed(model_name="llava:7b"):
    """Check if the OCR model is installed"""
    try:
        installed = get_ollama_models()
        # Check exact match or partial match
        for m in installed:
            if model_name in m or m in model_name:
                return True
        return False
    except:
        return False


def process_image_with_vision_model(image_path, model_name, prompt="Describe this image in detail. Extract any visible text."):
    """
    Process an image using a vision model for OCR/description.
    
    Args:
        image_path: Path to the image file
        model_name: Vision model to use (e.g., 'llava:7b')
        prompt: Prompt for the vision model
        
    Returns:
        dict: {'success': bool, 'text': str, 'error': str}
    """
    try:
        # Read and encode image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Call ollama with image
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_data]
            }],
            stream=False
        )
        
        text = response.get('message', {}).get('content', '')
        return {'success': True, 'text': text}
        
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        return {'success': False, 'error': str(e)}


def extract_pdf_text(pdf_path):
    """
    DEPRECATED: PDF extraction now uses OCR model directly.
    This function is kept for backwards compatibility but returns empty.
    """
    print(f"[PDF] extract_pdf_text: delegating to extract_text_from_pdf helper")
    try:
        # Use the visual_tools extractor which may use OCR when necessary
        result = extract_text_from_pdf(pdf_path)

        # Normalize possible return formats
        if isinstance(result, dict):
            text = result.get('text') or result.get('content') or ''
            success = result.get('success', True) if 'success' in result else True
            return {'success': success, 'error': None if success else result.get('error'), 'text': text}

        if isinstance(result, str):
            return {'success': True, 'error': None, 'text': result}

        return {'success': False, 'error': 'Unexpected extractor return type', 'text': ''}
    except Exception as e:
        print(f"[PDF] extract_pdf_text failed: {e}")
        return {'success': False, 'error': str(e), 'text': ''}


def get_unique_filepath(base_dir, filename):
    """
    Get a unique filepath, adding index if file already exists.
    
    Args:
        base_dir: Directory path
        filename: Original filename
        
    Returns:
        str: Unique filepath
    """
    os.makedirs(base_dir, exist_ok=True)
    
    base_name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]
    
    filepath = os.path.join(base_dir, filename)
    index = 1
    
    while os.path.exists(filepath):
        new_filename = f"{base_name}({index}){extension}"
        filepath = os.path.join(base_dir, new_filename)
        index += 1
    
    return filepath


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for better RAG retrieval.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within the last 100 chars
            search_start = max(end - 100, start)
            last_period = text.rfind('.', search_start, end)
            last_newline = text.rfind('\n', search_start, end)
            
            break_point = max(last_period, last_newline)
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

def get_tts_model(language='en'):
    """Lazy-load TTS model based on language"""
    global _tts_turbo_model, _tts_multilingual_model
    
    if language == 'en':
        if _tts_turbo_model is None:
            from POTATO.components.vocal_tools.clonevoice_turbo import speak_sentences_grouped, stop_current_tts
            _tts_turbo_model = {'speak': speak_sentences_grouped, 'stop': stop_current_tts}
        return _tts_turbo_model
    else:
        if _tts_multilingual_model is None:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import speak as speak_multi
            _tts_multilingual_model = {'speak': speak_multi, 'stop': lambda: None}
        return _tts_multilingual_model

def speak_text(text, language='en'):
    """Speak text using appropriate TTS model"""
    tts = get_tts_model(language)
    if language == 'en':
        tts['speak'](text)
    else:
        tts['speak'](text, language=language)

def stop_tts():
    """Stop all TTS"""
    if _tts_turbo_model:
        _tts_turbo_model['stop']()
    if _tts_multilingual_model:
        _tts_multilingual_model['stop']()

def unload_tts_models():
    """Unload all TTS models from VRAM"""
    global _tts_turbo_model, _tts_multilingual_model
    
    # Stop any playing audio first
    stop_tts()
    
    # Unload Turbo model - use shutdown_tts for proper thread cleanup
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import shutdown_tts
        shutdown_tts()
        print("[TTS] Turbo model unloaded via shutdown_tts")
    except Exception as e:
        print(f"Error unloading Turbo model: {e}")
    _tts_turbo_model = None
    
    # Unload Multilingual model
    if _tts_multilingual_model:
        try:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import unload_models
            unload_models()
            print("[TTS] Multilingual model unloaded")
        except Exception as e:
            print(f"Error unloading Multilingual model: {e}")
        _tts_multilingual_model = None
    
    print("[TTS] All TTS models unloaded")

def unload_whisper_model():
    """Unload Whisper model from VRAM"""
    try:
        from POTATO.components.vocal_tools.audioflow import unload_whisper
        success = unload_whisper()
        if success:
            print("[Whisper] Successfully unloaded from VRAM")
        else:
            print("[Whisper] Unload completed with warnings")
    except Exception as e:
        print(f"[Whisper] Error unloading: {e}")
        import traceback
        traceback.print_exc()

def find_working_microphone():
    """Find a working microphone device by testing all available devices"""
    try:
        import sounddevice as sd
        
        # Get list of input devices
        devices = sd.query_devices()
        input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        print(f"Found {len(input_devices)} input devices")
        
        for device_id in input_devices:
            try:
                # Test recording for a short duration
                print(f"Testing device {device_id}: {devices[device_id]['name']}")
                test_recording = sd.rec(
                    int(0.5 * devices[device_id]['default_samplerate']),
                    samplerate=int(devices[device_id]['default_samplerate']),
                    channels=1,
                    device=device_id,
                    blocking=True
                )
                
                # Check if we got any input
                if test_recording.max() > 0.001:  # Some threshold
                    print(f"Found working device: {device_id}")
                    return device_id
            except Exception as e:
                print(f"Device {device_id} failed: {e}")
                continue
        
        # If no device found, return default
        return None
    except Exception as e:
        print(f"Error finding microphone: {e}")
        return None
    
    return contents

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Get or update settings"""
    if request.method == 'POST':
        data = request.json
        success = save_user_settings(data)
        return jsonify({"status": "success" if success else "error"})
    
    return jsonify(load_settings())

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    success = reset_user_settings()
    if success:
        return jsonify({"status": "success", "settings": load_settings()})
    return jsonify({"status": "error"}), 500

@app.route('/api/settings/descriptions', methods=['GET'])
def get_setting_descriptions():
    """Parse config.env.txt and return setting descriptions"""
    try:
        config_env_path = os.path.join(BASE_DIR, '..', 'config.env.txt')
        descriptions = {}
        
        if os.path.exists(config_env_path):
            with open(config_env_path, 'r', encoding='utf-8') as f:
                current_key = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') and '=' not in line:
                        continue
                    
                    if '=' in line:
                        # Parse setting line: KEY="value" # Description
                        parts = line.split('#', 1)
                        key_value = parts[0].strip()
                        
                        if '=' in key_value:
                            key = key_value.split('=')[0].strip()
                            description = parts[1].strip() if len(parts) > 1 else 'No description available'
                            descriptions[key] = description
        
        return jsonify(descriptions)
    except Exception as e:
        print(f"Error reading config.env.txt: {e}")
        return jsonify({}), 500

@app.route('/api/preferences', methods=['GET', 'POST'])
def handle_preferences():
    """Get or update UI preferences (web search toggle, selected model, language, etc.)"""
    prefs_path = os.path.join(DATA_DIR, 'preferences.json')
    
    if request.method == 'POST':
        data = request.json
        try:
            # Load existing preferences
            prefs = {}
            if os.path.exists(prefs_path):
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
            
            # Update with new data
            prefs.update(data)
            
            # Save
            with open(prefs_path, 'w') as f:
                json.dump(prefs, f, indent=4)
            return jsonify({"success": True, "preferences": prefs})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    # GET - return current preferences
    try:
        if os.path.exists(prefs_path):
            with open(prefs_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # GET
    try:
        if os.path.exists(prefs_path):
            with open(prefs_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({
            "selected_model": "gpt-oss:20b",
            "vox_language": "en"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """Get list of available Ollama models with vision info"""
    models = get_ollama_models()
    # Include vision capability info for each model
    models_with_info = []
    for model in models:
        vision_check = check_model_is_vision(model)
        models_with_info.append({
            'name': model,
            'is_vision': vision_check.get('is_vision', False)
        })
    return jsonify({"models": models, "models_info": models_with_info})

@app.route('/api/system_stats')
def system_stats():
    """Get system statistics - no console logging"""
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    gpu, gpu_temp, vram = 0, 0, 0
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0].load * 100
            gpu_temp = gpus[0].temperature
            vram = gpus[0].memoryUtil * 100
    except:
        pass
    
    return jsonify({
        "cpu": cpu,
        "ram": ram,
        "gpu": gpu,
        "gpu_temp": gpu_temp,
        "vram": vram
    })

@app.route('/api/chats', methods=['GET'])
def list_chats_route():
    """List all chat sessions (text chats only by default)"""
    search_query = request.args.get('search')
    # Default to text chats only (exclude voice chats)
    chats = get_all_chats(search_query, text_only=True)
    return jsonify(chats)

@app.route('/api/voice_chats', methods=['GET'])
def list_voice_chats_route():
    """List all voice chat sessions (voice chats only)"""
    search_query = request.args.get('search')
    chats = get_all_chats(search_query, voice_only=True)
    return jsonify(chats)

@app.route('/api/chats/<session_id>', methods=['GET', 'DELETE'])
def chat_session_route(session_id):
    """Get or delete a specific chat session"""
    if request.method == 'DELETE':
        # Delete chat JSON file
        path = os.path.join(CHATS_DIR, f"{session_id}.json")
        deleted_chat = os.path.exists(path)
        if deleted_chat:
            os.remove(path)
        
        # Also delete any uploaded files for this chat
        chat_upload_dir = os.path.join(UPLOADS_DIR, session_id)
        if os.path.exists(chat_upload_dir):
            try:
                import shutil
                shutil.rmtree(chat_upload_dir)
                print(f"[DELETE] Removed uploads for chat: {session_id}")
            except Exception as e:
                print(f"[DELETE] Failed to remove uploads: {e}")
        
        # Also delete temp files for this chat
        chat_temp_dir = os.path.join(TEMP_DIR, session_id)
        if os.path.exists(chat_temp_dir):
            try:
                import shutil
                shutil.rmtree(chat_temp_dir)
                print(f"[DELETE] Removed temp files for chat: {session_id}")
            except Exception as e:
                print(f"[DELETE] Failed to remove temp files: {e}")
        
        # Delete RAG embeddings for this chat
        try:
            delete_chat_collection(session_id)
            print(f"[DELETE] Removed RAG collection for chat: {session_id}")
        except Exception as e:
            print(f"[DELETE] Failed to remove RAG collection: {e}")
        
        if deleted_chat:
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Not found"}), 404
    
    # GET
    chat = load_chat_session(session_id)
    if chat:
        return jsonify(chat)
    return jsonify({"error": "Not found"}), 404

@app.route('/api/chats/<session_id>/messages', methods=['DELETE'])
def clear_chat_messages(session_id):
    """Clear all messages in a chat session (keep the chat, just empty its history)"""
    try:
        path = os.path.join(CHATS_DIR, f"{session_id}.json")
        if not os.path.exists(path):
            return jsonify({"error": "Not found"}), 404
        chat = load_chat_session(session_id)
        if chat:
            chat['messages'] = []
            save_chat_session(session_id, [], title=chat.get('title'), is_voice_chat=session_id.startswith('vox_'))
        return jsonify({"status": "cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats/<session_id>/save_partial', methods=['POST'])
def save_partial_response(session_id):
    """Save partial response when generation is stopped"""
    try:
        data = request.json
        content = data.get('content', '')
        model = data.get('model', 'unknown')
        
        # Load existing session
        existing = load_chat_session(session_id)
        if existing:
            history = existing['messages']
            # Add partial response with marker
            history.append({
                "role": "assistant",
                "content": content,
                "model": model,
                "_stopped": True
            })
            save_chat_session(session_id, history)
            return jsonify({"success": True})
        return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)
    file.save(filepath)
    
    # Read file content for context
    content = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        pass
    
    return jsonify({
        "path": filepath,
        "filename": filename,
        "content": content
    })


# --- VISION & RAG ENDPOINTS ---

@app.route('/api/check_vision_capability', methods=['POST'])
def check_vision_capability():
    """
    Check if a model has vision capabilities and if OCR model is available.
    Used by frontend to show appropriate warnings/info.
    """
    data = request.json
    model_name = data.get('model', '')
    
    # Check if current model is vision-capable
    current_check = check_model_is_vision(model_name)
    
    # Get backup vision model from settings
    settings = load_settings()
    backup_model = settings.get('configuration', {}).get('ollama_models', {}).get('BACKUP_VISION_MODEL', 'llava:7b')
    default_ocr = settings.get('configuration', {}).get('ollama_models', {}).get('DEFAULT_OCR_MODEL', 'llava:7b')
    
    # Check if backup/OCR model is installed
    ocr_installed = check_ocr_model_installed(default_ocr)
    backup_installed = check_ocr_model_installed(backup_model)
    
    # Get all available vision models
    available_vision = get_available_vision_models()
    
    return jsonify({
        "current_model": model_name,
        "is_vision": current_check['is_vision'],
        "vision_known": current_check['known'],
        "backup_vision_model": backup_model,
        "backup_installed": backup_installed,
        "default_ocr_model": default_ocr,
        "ocr_installed": ocr_installed,
        "available_vision_models": available_vision,
        "can_process_images": current_check['is_vision'] or (backup_installed and len(available_vision) > 0)
    })


@app.route('/api/get_uploaded_file', methods=['GET'])
def get_uploaded_file():
    """
    Serve an uploaded file from the uploads folder.
    Used for viewing images and PDFs in chat history.
    """
    chat_id = request.args.get('chat_id', '')
    filename = request.args.get('filename', '')
    
    if not chat_id or not filename:
        return jsonify({"error": "Missing chat_id or filename"}), 400
    
    # Security: sanitize filename
    filename = secure_filename(filename)
    
    # Look in uploads folder for this chat
    file_path = os.path.join(UPLOADS_DIR, chat_id, filename)
    
    if not os.path.exists(file_path):
        # Also check the temp folder where PDFs might be stored
        file_path = os.path.join(TEMP_DIR, chat_id, filename)
    
    if not os.path.exists(file_path):
        # Try without chat_id subfolder (flat structure)
        file_path = os.path.join(UPLOADS_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Determine MIME type
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.md': 'text/markdown',
    }
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    # If as_text is requested, read and return as plain text
    as_text = request.args.get('as_text', 'false').lower() == 'true'
    if as_text:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return Response(content, mimetype='text/plain; charset=utf-8')
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return Response(content, mimetype='text/plain; charset=utf-8')
            except:
                return jsonify({"error": "Cannot read file as text"}), 400
    
    return send_file(file_path, mimetype=mime_type)


@app.route('/api/upload_chat_file', methods=['POST'])
def upload_chat_file():
    """
    Upload a file for chat context.
    
    Behavior:
    - PDFs: Convert pages to base64 images, send to llava for OCR, return text content
    - Images: Convert to base64, either return for VL model OR describe with llava for non-VL
    - Text files: Read content directly
    
    RAG mode: Embed content to Weaviate with proper namespacing (chat_id + filename)
    
    CRITICAL: All images sent to Ollama MUST be base64 encoded, not file paths!
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    chat_id = request.form.get('chat_id', str(uuid.uuid4()))
    current_model = request.form.get('model', '')
    use_rag = request.form.get('use_rag', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400
    
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Determine file type
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.avif', '.heic', '.heif']
    pdf_extensions = ['.pdf']
    text_extensions = {
        # Documents
        '.txt', '.md', '.csv', '.tsv',
        # Config files
        '.json', '.yaml', '.yml', '.xml', '.env', '.ini', '.toml', '.conf', '.cfg',
        # Web
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
        # JavaScript/TypeScript
        '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        # Python
        '.py', '.pyw', '.pyi',
        # C/C++
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',
        # Other languages
        '.java', '.kt', '.scala', '.go', '.rs', '.rb', '.php', '.pl', '.pm',
        '.swift', '.m', '.mm', '.r', '.lua', '.sh', '.bash', '.zsh',
        '.bat', '.cmd', '.ps1', '.psm1',
        # Data
        '.sql', '.graphql', '.proto',
        # Misc
        '.log', '.rst', '.tex', '.dockerfile', '.gitignore', '.editorconfig'
    }
    
    is_image = file_ext in image_extensions
    is_pdf = file_ext in pdf_extensions
    is_text = file_ext in text_extensions
    
    # Security: only allow known file types
    if not is_image and not is_pdf and not is_text:
        return jsonify({
            "error": "file_type_not_allowed",
            "message": f"File type '{file_ext}' is not allowed. Allowed types: images, PDFs, and text/code files."
        }), 400
    
    settings = load_settings()
    fallback_vl_model = settings.get('configuration', {}).get('ollama_models', {}).get('BACKUP_VISION_MODEL', 'llava:7b')
    ocr_model = settings.get('configuration', {}).get('ollama_models', {}).get('DEFAULT_OCR_MODEL', 'llava:7b')
    
    # Check vision capability of current model
    vision_check = check_model_is_vision(current_model)
    is_vl_model = vision_check['is_vision']
    
    # Only require VL model for images (they need description for non-VL models)
    # PDFs now use text extraction, and text files never need VL
    if is_image and not is_vl_model:
        if not check_ocr_model_installed(fallback_vl_model):
            return jsonify({
                "error": "vision_required",
                "message": f"No vision model available. Install {fallback_vl_model} with: ollama pull {fallback_vl_model}",
                "available_vision_models": get_available_vision_models()
            }), 400
    
    # Read file into memory
    file_bytes = file.read()
    
    # Save original file to uploads folder for later viewing in chat history
    chat_upload_dir = os.path.join(UPLOADS_DIR, chat_id)
    os.makedirs(chat_upload_dir, exist_ok=True)
    saved_path = os.path.join(chat_upload_dir, filename)
    with open(saved_path, 'wb') as f:
        f.write(file_bytes)
    print(f"[UPLOAD] Saved original file to: {saved_path}")
    
    result = {
        "success": True,
        "filename": filename,
        "chat_id": chat_id,
        "file_type": "image" if is_image else ("pdf" if is_pdf else "text"),
        "content": "",
        "embedded": False,
        "is_vl_model": is_vl_model,
        "saved_path": saved_path  # Path for later retrieval
    }
    
    # ==================== IMAGE HANDLING ====================
    if is_image:
        # Convert to base64
        image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        
        # ALWAYS return base64 for images - let frontend decide based on current model
        result['image_base64'] = image_b64
        result['content'] = f"[Image attached: {filename}]"
        
        # If current model is NOT VL, also describe the image as fallback
        if not is_vl_model:
            result['processed_with_fallback'] = True
            result['fallback_model'] = fallback_vl_model
            
            try:
                from POTATO.components.visual_tools.extract_text import describe_image_with_vision
                ocr_result = describe_image_with_vision(image_b64, model=fallback_vl_model)
                if ocr_result['success']:
                    result['description'] = ocr_result['text']
            except Exception as e:
                print(f"[UPLOAD] Image description error: {e}")
        
        # If RAG mode, describe and embed
        if use_rag:
            try:
                from POTATO.components.visual_tools.extract_text import describe_image_with_vision
                ocr_result = describe_image_with_vision(image_b64, model=ocr_model)
                if ocr_result['success']:
                    ocr_text = f"[Image: {filename}]\n{ocr_result['text']}"
                    embed_result = embed_ocr_content(ocr_text, chat_id, filename, content_type="image")
                    if embed_result['status'] == 'success':
                        result['embedded'] = True
            except Exception as e:
                print(f"[UPLOAD] RAG embedding error: {e}")
    
    # ==================== PDF HANDLING ====================
    # For regular uploads (non-RAG), ALWAYS use fast text extraction
    # Never use llava for PDFs in regular uploads - it's too slow
    elif is_pdf:
        print(f"[PDF] {filename}: Using fast text extraction (non-RAG upload)")
        
        try:
            # Save temp file for PyMuPDF to read
            temp_pdf_path = os.path.join(TEMP_DIR, f"pdf_{chat_id}_{filename}")
            os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
            with open(temp_pdf_path, 'wb') as f:
                f.write(file_bytes)
            
            # Import PDF functions
            from POTATO.components.visual_tools.extract_text import pdf_to_base64_images, fast_extract_pdf_text, extract_text_from_pdf

            # Use fast text extraction (PyMuPDF or PyPDF2) for non-RAG uploads
            if not use_rag:
                pdf_text_result = fast_extract_pdf_text(temp_pdf_path, max_pages=None)
            else:
                # RAG mode: use vision-based extractor (llava) to also capture images/diagrams
                pdf_text_result = extract_text_from_pdf(temp_pdf_path, model=ocr_model, max_pages=20)

            if pdf_text_result['success'] and pdf_text_result.get('text', '').strip():
                result['content'] = pdf_text_result.get('text', '')
                result['pages'] = pdf_text_result.get('pages', 0)
                print(f"[PDF] {filename}: Text extracted - {len(result['content'])} chars, {result['pages']} pages")
            else:
                result['content'] = ""
                result['pdf_error'] = pdf_text_result.get('error', 'No text could be extracted')
            
            # For VL models, also include page images for visual context (up to 8 pages)
            if is_vl_model:
                page_images = pdf_to_base64_images(temp_pdf_path, max_pages=8)
                if page_images:
                    result['pdf_pages_base64'] = page_images
                    result['pages'] = len(page_images)
                    print(f"[PDF] {filename}: {len(page_images)} page images for VL model")
            
            # Delete temp file
            try:
                os.remove(temp_pdf_path)
            except:
                pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['content'] = ""
            result['pdf_error'] = str(e)
        
        # RAG mode: embed content
        if use_rag and result.get('content', '').strip():
            try:
                chunk_size = settings.get('rag_config', {}).get('CHUNK_SIZE', 500)
                chunk_overlap = settings.get('rag_config', {}).get('CHUNK_OVERLAP', 50)
                chunks = chunk_text(result['content'], chunk_size, chunk_overlap)
                embed_result = embed_to_weaviate(chunks, chat_id=chat_id, source_filename=filename)
                if embed_result['status'] == 'success':
                    result['embedded'] = True
                    print(f"[RAG] Embedded PDF: {len(chunks)} chunks")
            except Exception as e:
                print(f"[UPLOAD] RAG embedding error: {e}")
    
    # ==================== TEXT FILE HANDLING ====================
    else:
        try:
            result['content'] = file_bytes.decode('utf-8')
        except:
            try:
                result['content'] = file_bytes.decode('latin-1')
            except:
                result['content'] = ""
        
        print(f"[UPLOAD] Text file: {len(result['content'])} chars")
        
        # RAG mode: embed content
        if use_rag and result['content'].strip():
            try:
                chunk_size = settings.get('rag_config', {}).get('CHUNK_SIZE', 500)
                chunk_overlap = settings.get('rag_config', {}).get('CHUNK_OVERLAP', 50)
                chunks = chunk_text(result['content'], chunk_size, chunk_overlap)
                embed_result = embed_to_weaviate(chunks, chat_id=chat_id, source_filename=filename)
                if embed_result['status'] == 'success':
                    result['embedded'] = True
            except Exception as e:
                print(f"[UPLOAD] RAG embedding error: {e}")
    
    # Model management for RAG: unload OCR model and reload chat model if needed
    if use_rag and (is_image or is_pdf):
        try:
            # Unload the OCR model to free VRAM
            ollama.generate(model=ocr_model, prompt="", keep_alive=0)
            print(f"[MODEL] Unloaded OCR model: {ocr_model}")
            
            # Reload the chat model with high keepalive
            if current_model and current_model != ocr_model:
                ollama.generate(model=current_model, prompt="", keep_alive=600)
                print(f"[MODEL] Reloaded chat model: {current_model} with 600s keepalive")
        except Exception as e:
            print(f"[MODEL] Model management error: {e}")
    
    return jsonify(result)


@app.route('/api/embed_to_rag', methods=['POST'])
def embed_to_rag():
    """
    Upload and embed files for RAG context.
    Files saved to POTATO/.temp/uploads/<chat_id>/ then embedded and deleted.
    
    Options:
    - pdf_visual_analysis: If 'true', use llava to describe PDF pages for graphs/images
                          If 'false' or missing, use fast text extraction only
    """
    chat_id = request.form.get('chat_id', str(uuid.uuid4()))
    embed_model = request.form.get('embed_model', 'nomic-embed-text:latest')
    pdf_visual_analysis = request.form.get('pdf_visual_analysis', 'false').lower() == 'true'
    
    settings = load_settings()
    ocr_model = settings.get('configuration', {}).get('ollama_models', {}).get('DEFAULT_OCR_MODEL', 'llava:7b')
    chunk_size = settings.get('rag_config', {}).get('CHUNK_SIZE', 500)
    chunk_overlap = settings.get('rag_config', {}).get('CHUNK_OVERLAP', 50)
    
    print(f"[RAG] PDF visual analysis: {pdf_visual_analysis}")
    
    # Check Weaviate connection
    weaviate_status = check_weaviate_connection()
    if weaviate_status.get('status') != 'connected':
        return jsonify({
            "status": "error",
            "error": "Weaviate vector database is not running. Please start the Docker container."
        }), 500
    
    # Create temp directory for this chat
    temp_chat_dir = os.path.join(TEMP_DIR, chat_id)
    os.makedirs(temp_chat_dir, exist_ok=True)
    
    embedded_count = 0
    errors = []
    processed_files = []
    
    # Allowed file extensions for security
    # Text/code files - read as plain text, NO execution
    ALLOWED_TEXT_EXTENSIONS = {
        # Documents
        '.txt', '.md', '.pdf', '.csv', '.tsv',
        # Config files
        '.json', '.yaml', '.yml', '.xml', '.env', '.ini', '.toml', '.conf', '.cfg',
        # Web
        '.html', '.htm', '.css', '.scss', '.sass', '.less',
        # JavaScript/TypeScript
        '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        # Python
        '.py', '.pyw', '.pyi',
        # C/C++
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',
        # Other languages
        '.java', '.kt', '.scala', '.go', '.rs', '.rb', '.php', '.pl', '.pm',
        '.swift', '.m', '.mm', '.r', '.R', '.lua', '.sh', '.bash', '.zsh',
        '.bat', '.cmd', '.ps1', '.psm1',  # Scripts (read only, not executed)
        # Data
        '.sql', '.graphql', '.proto',
        # Misc
        '.log', '.rst', '.tex', '.dockerfile', '.gitignore', '.editorconfig'
    }
    ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max per file
    
    # Process document files
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            continue
        
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Security check: validate extension
        if file_ext not in ALLOWED_TEXT_EXTENSIONS:
            errors.append(f"{filename}: File type not allowed")
            continue
        
        # Security check: read file size
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            errors.append(f"{filename}: File too large (max 50MB)")
            continue
        
        filepath = get_unique_filepath(temp_chat_dir, filename)
        file.save(filepath)
        
        try:
            if file_ext == '.pdf':
                # Extract text from PDF. Use fast extractor for plain text unless
                # visual analysis is explicitly enabled.
                all_content = []

                if pdf_visual_analysis:
                    # Use vision-based extractor for richer output (may include images/tables)
                    try:
                        pdf_result = extract_text_from_pdf(filepath, model=ocr_model, max_pages=10)
                    except Exception as e:
                        print(f"[RAG] Vision extractor failed, falling back to fast text extractor: {e}")
                        pdf_result = fast_extract_pdf_text(filepath, max_pages=None)
                else:
                    # Fast, local text extraction (PyMuPDF or PyPDF2)
                    pdf_result = fast_extract_pdf_text(filepath, max_pages=None)

                if pdf_result['success'] and pdf_result.get('text', '').strip():
                    all_content.append(f"[PDF Text Content]\n{pdf_result['text']}")
                
                # ONLY describe PDF pages visually if option is enabled
                if pdf_visual_analysis and check_ocr_model_installed(ocr_model):
                    try:
                        from POTATO.components.visual_tools.extract_text import pdf_to_base64_images, describe_images_batch
                        
                        # Convert PDF pages to images (limit to first 10 pages for RAG)
                        page_images = pdf_to_base64_images(filepath, max_pages=10)
                        
                        if page_images:
                            print(f"[RAG] {filename}: Visual analysis of {len(page_images)} pages with {ocr_model}")
                            # Enhanced prompt for RAG
                            rag_pdf_prompt = """Analyze this PDF page for RAG indexing. Extract and describe:
1. ALL text content exactly as written
2. Tables: convert to simple HTML format
3. Graphs/charts: describe type, axis labels, all data points/values
4. Diagrams: describe elements and relationships
5. Images: describe what they show
Be thorough - this will be used for semantic search."""
                            
                            source_names = [f"{filename} page {i+1}" for i in range(len(page_images))]
                            results = describe_images_batch(
                                page_images, 
                                model=ocr_model, 
                                prompt=rag_pdf_prompt,
                                parallel=2,
                                source_names=source_names
                            )
                            
                            for r in results:
                                if r and r.get('success') and r.get('text', '').strip():
                                    all_content.append(f"[{r['source']} - Visual Analysis]\n{r['text']}")
                    except Exception as e:
                        print(f"[RAG] PDF visual analysis error: {e}")
                elif not pdf_visual_analysis:
                    print(f"[RAG] {filename}: Using text extraction only (visual analysis disabled)")
                
                if all_content:
                    combined_text = "\n\n".join(all_content)
                    chunks = chunk_text(combined_text, chunk_size, chunk_overlap)
                    result = embed_to_weaviate(chunks, chat_id=chat_id, source_filename=filename)
                    if result['status'] == 'success':
                        embedded_count += result['embedded_count']
                        processed_files.append({'file': filename, 'chunks': len(chunks), 'visual_analysis': pdf_visual_analysis})
                    else:
                        errors.append(f"{filename}: {result.get('error', 'Unknown error')}")
                else:
                    errors.append(f"{filename}: No content could be extracted from PDF")
            else:
                # Text files
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except:
                    with open(filepath, 'r', encoding='latin-1') as f:
                        content = f.read()
                
                if content.strip():
                    chunks = chunk_text(content, chunk_size, chunk_overlap)
                    result = embed_to_weaviate(chunks, chat_id=chat_id, source_filename=filename)
                    if result['status'] == 'success':
                        embedded_count += result['embedded_count']
                        processed_files.append({'file': filename, 'chunks': len(chunks)})
                    else:
                        errors.append(f"{filename}: {result.get('error', 'Unknown error')}")
            
            # Clean up temp file after successful embedding
            os.remove(filepath)
            
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    # Process image files
    images = request.files.getlist('images')
    if images and len(images) > 0:
        # Check if OCR model is available
        if not check_ocr_model_installed(ocr_model):
            errors.append(f"OCR model '{ocr_model}' not installed. Cannot process images.")
        else:
            for image in images:
                if image.filename == '':
                    continue
                
                filename = secure_filename(image.filename)
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Security check: validate extension
                if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                    errors.append(f"{filename}: Image type not allowed")
                    continue
                
                # Security check: file size
                image.seek(0, 2)
                image_size = image.tell()
                image.seek(0)
                
                if image_size > MAX_FILE_SIZE:
                    errors.append(f"{filename}: Image too large (max 50MB)")
                    continue
                
                filepath = get_unique_filepath(temp_chat_dir, filename)
                image.save(filepath)
                
                try:
                    # Process image with OCR model - enhanced prompt for RAG
                    rag_image_prompt = """Analyze this image thoroughly for RAG (Retrieval-Augmented Generation) indexing.

If it's a GRAPH or CHART:
- Describe the type (bar chart, line graph, pie chart, etc.)
- List all axis labels, values, and data points
- Describe trends and relationships
- Include all numbers visible
- Format: "Graph showing [type]: X-axis: [label], Y-axis: [label]. Data points: [list all visible values]"

If it's a TABLE:
- Convert to simple HTML table format: <table><tr><th>Header1</th>...</tr><tr><td>Data1</td>...</tr></table>
- Include ALL rows and columns
- Preserve exact cell values

If it's TEXT/DOCUMENT:
- Extract ALL visible text exactly as written
- Preserve formatting and structure
- Note any headings, bullet points, or numbered lists

If it's a DIAGRAM/FLOWCHART:
- Describe each element and their connections
- List all labels and text
- Describe the flow/relationships

If it's a PHOTO/SCENE:
- Describe objects, people, text, signs visible
- Note any relevant details

Always include:
1. All text/numbers visible in the image
2. The overall purpose/meaning
3. Key information that would help answer questions about this content"""
                    
                    ocr_result = process_image_with_vision_model(
                        filepath, 
                        ocr_model,
                        rag_image_prompt
                    )
                    
                    if ocr_result['success'] and ocr_result.get('text', '').strip():
                        # Add source info to the text
                        ocr_text = f"[Image: {filename}]\n{ocr_result['text']}"
                        
                        result = embed_ocr_content(ocr_text, chat_id, filename, content_type="image_ocr")
                        if result['status'] == 'success':
                            embedded_count += 1
                            processed_files.append({'file': filename, 'type': 'image', 'chunks': 1})
                        else:
                            errors.append(f"{filename}: {result.get('error', 'Embedding failed')}")
                    else:
                        errors.append(f"{filename}: {ocr_result.get('error', 'OCR extraction failed')}")
                    
                    # Clean up
                    os.remove(filepath)
                    
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
    
    # Clean up temp directory if empty
    try:
        if os.path.exists(temp_chat_dir) and not os.listdir(temp_chat_dir):
            os.rmdir(temp_chat_dir)
    except:
        pass
    
    return jsonify({
        "status": "success" if embedded_count > 0 else ("partial" if errors else "error"),
        "embedded_count": embedded_count,
        "collection_name": f"Chat_{chat_id.replace('-', '_')}",
        "processed_files": processed_files,
        "errors": errors if errors else None
    })


@app.route('/api/search_rag', methods=['POST'])
def search_rag():
    """Search the RAG vector database for a specific chat"""
    data = request.json
    query = data.get('query', '')
    chat_id = data.get('chat_id')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Check Weaviate connection
    weaviate_status = check_weaviate_connection()
    if weaviate_status.get('status') != 'connected':
        return jsonify({
            "status": "error",
            "error": "Weaviate vector database is not running"
        }), 500
    
    results = query_weaviate(query, chat_id=chat_id, limit=top_k)
    
    return jsonify({
        "status": "success",
        "query": query,
        "results": [
            {
                "content": r['content'],
                "source_file": r.get('source_file', ''),
                "content_type": r.get('content_type', 'text'),
                "score": 1 - r['distance'] if r['distance'] is not None else None
            }
            for r in results
        ]
    })


@app.route('/api/rag_status', methods=['GET'])
def rag_status():
    """Get RAG/Weaviate status"""
    weaviate_status = check_weaviate_connection()
    collections = list_collections() if weaviate_status.get('status') == 'connected' else []
    
    return jsonify({
        "weaviate": weaviate_status,
        "collections": collections,
        "collection_count": len(collections)
    })


@app.route('/api/rag_files/<chat_id>', methods=['GET'])
def get_rag_files(chat_id):
    """Get list of embedded files for a specific chat"""
    weaviate_status = check_weaviate_connection()
    
    if weaviate_status.get('status') != 'connected':
        return jsonify({
            "error": "Weaviate not connected",
            "files": []
        })
    
    files = list_embedded_files(chat_id)
    return jsonify({
        "chat_id": chat_id,
        "files": files,
        "count": len(files)
    })


@app.route('/api/delete_rag_collection', methods=['POST'])
def delete_rag_collection():
    """Delete RAG collection for a specific chat"""
    data = request.json
    chat_id = data.get('chat_id')
    
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    
    success = delete_chat_collection(chat_id)
    
    return jsonify({
        "status": "success" if success else "error",
        "chat_id": chat_id
    })


@app.route('/api/delete_file_embeddings', methods=['POST'])
def delete_file_embeddings():
    """Delete embeddings for a specific file within a chat's collection"""
    data = request.json
    chat_id = data.get('chat_id')
    filename = data.get('filename')
    
    if not chat_id or not filename:
        return jsonify({"error": "chat_id and filename required"}), 400
    
    try:
        success = delete_file_from_collection(chat_id, filename)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/delete_uploaded_file', methods=['POST'])
def delete_uploaded_file():
    """Delete an uploaded file from disk"""
    data = request.json
    path = data.get('path')
    
    if not path:
        return jsonify({"error": "path required"}), 400
    
    # Security check: only allow deleting files within our upload directories
    abs_path = os.path.abspath(path)
    uploads_abs = os.path.abspath(UPLOADS_DIR)
    temp_abs = os.path.abspath(TEMP_DIR)
    
    if not (abs_path.startswith(uploads_abs) or abs_path.startswith(temp_abs)):
        return jsonify({"error": "Access denied"}), 403
    
    try:
        if os.path.exists(abs_path):
            os.remove(abs_path)
            return jsonify({"success": True})
        else:
            return jsonify({"success": True, "message": "File already deleted"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio to text using Whisper"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        filename = secure_filename(f"{uuid.uuid4()}.wav")
        filepath = os.path.join(UPLOADS_DIR, filename)
        audio_file.save(filepath)
        
        # Use audioflow.listen_stt
        result = listen_stt(filepath)
        
        # Clean up temp file
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Failed to remove temp file: {e}")
        
        if result['success']:
            return jsonify({"text": result['text'], "language": result['language']})
        else:
            error_msg = result.get('error', 'Unknown transcription error')
            print(f"Transcription error: {error_msg}")
            return jsonify({"error": error_msg}), 500
            
    except Exception as e:
        print(f"Exception in transcribe endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/transcribe_preload', methods=['POST'])
def transcribe_preload():
    """Preload Whisper model to avoid first-time delay"""
    try:
        # Call listen_stt with None to trigger model loading
        from POTATO.components.vocal_tools.audioflow import _get_whisper_pipeline
        _get_whisper_pipeline()
        return jsonify({"success": True, "message": "Whisper model loaded"})
    except Exception as e:
        print(f"Error preloading Whisper: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Transcribe audio from chat page voice recording - uses same logic as VOX Core (NO FFMPEG)"""
    try:
        print("[STT] Transcribe audio endpoint called")
        from POTATO.components.vocal_tools.realtime_stt import get_whisper_pipeline
        import numpy as np
        import io
        
        if 'audio' not in request.files:
            print("[STT] Error: No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        language_mode = request.form.get('language_mode', 'translate-en')
        print(f"[STT] Received audio file, language_mode: {language_mode}")
        
        # Read audio bytes
        audio_bytes = audio_file.read()
        print(f"[STT] Audio size: {len(audio_bytes)} bytes")
        
        try:
            # Try scipy wavfile first (handles WAV directly - same as VOX Core)
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            print(f"[STT] Parsed WAV: {sample_rate}Hz, shape: {audio_data.shape}")
            
            # Convert to float32 [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        except Exception as wav_error:
            print(f"[STT] WAV parsing failed: {wav_error}")
            return jsonify({"error": "Failed to parse audio. Browser must send WAV format."}), 400
        
        # Get Whisper pipeline (loads model if needed)
        print("[STT] Getting Whisper pipeline...")
        pipe = get_whisper_pipeline()
        
        # Ensure audio is float32 mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
        audio_data = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            print(f"[STT] Resampling from {sample_rate}Hz to 16000Hz")
            audio_data = audio_data[::int(sample_rate / 16000)]
        
        # Transcribe directly from numpy array (NO FILE, NO FFMPEG - same as VOX Core)
        print(f"[STT] Processing with mode: {language_mode}")
        
        # Handle language mode
        if language_mode == 'translate-en':
            # Translate to English (Whisper's translate task)
            result = pipe(
                audio_data,
                generate_kwargs={
                    "task": "translate"
                }
            )
        elif language_mode == 'auto':
            # Auto-detect and transcribe in original language
            result = pipe(
                audio_data,
                generate_kwargs={
                    "task": "transcribe"
                }
            )
        else:
            # Force specific language (e.g., 'tr', 'fr', etc.)
            result = pipe(
                audio_data,
                generate_kwargs={
                    "language": language_mode,
                    "task": "transcribe"
                }
            )
        
        text = result.get("text", "").strip()
        detected_lang = result.get("chunks", [{}])[0].get("language", "unknown") if "chunks" in result else "unknown"
        print(f"[STT] Result: {text} (detected: {detected_lang})")
        
        return jsonify({
            "text": text,
            "language_mode": language_mode,
            "detected_language": detected_lang,
            "success": bool(text)
        })
            
    except Exception as e:
        print(f"[STT] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_inference():
    """Stop ongoing inference"""
    data = request.json or {}
    model_name = data.get('model')
    
    # Stop Ollama
    stop_ollama_inference(model_name)
    
    # Stop TTS
    stop_tts()
    
    return jsonify({"status": "stopped"})

@app.route('/api/unload_model', methods=['POST'])
def unload_model():
    """Unload chat model from VRAM - properly stop inference first"""
    data = request.json or {}
    model_name = data.get('model')
    
    try:
        if not model_name:
            return jsonify({"success": False, "error": "No model specified"}), 400
        
        # Step 1: Stop any active inference
        print(f"[UNLOAD] Stopping inference for {model_name}...")
        stop_ollama_inference(model_name)
        
        # Step 2: Give it a moment to stop
        import time
        time.sleep(1)
        
        # Step 3: Force unload with keep_alive=0
        print(f"[UNLOAD] Unloading {model_name} from VRAM...")
        import requests
        response = requests.post('http://localhost:11434/api/generate',
                               json={'model': model_name, 'keep_alive': 0},
                               timeout=10)
        
        # Step 4: Verify it's gone
        time.sleep(1)
        ps_result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
        if ps_result.returncode == 0:
            lines = ps_result.stdout.strip().split('\n')[1:]
            still_loaded = any(model_name in line for line in lines if line.strip())
            
            if still_loaded:
                print(f"[UNLOAD] Warning: {model_name} still in VRAM after unload attempt")
                return jsonify({"success": False, "error": f"{model_name} still loaded"}), 500
        
        print(f"[UNLOAD] ✓ {model_name} successfully unloaded")
        return jsonify({"success": True, "message": f"Model {model_name} unloaded"})
        
    except Exception as e:
        print(f"[UNLOAD] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/unload_vox', methods=['POST'])
def unload_vox():
    """Unload all VOX models (TTS + Whisper)"""
    try:
        unload_tts_models()
        unload_whisper_model()
        return jsonify({"success": True, "message": "VOX models unloaded"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/unload_stt', methods=['POST'])
def unload_stt():
    """Unload STT (Whisper) model only"""
    try:
        print("[STT] Unloading Whisper model...")
        unload_whisper_model()
        print("[STT] \u2713 Whisper unloaded successfully")
        return jsonify({"success": True, "message": "Whisper unloaded"})
    except Exception as e:
        print(f"[STT] Error unloading: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/unload_tts', methods=['POST'])
def unload_tts():
    """Unload TTS models only"""
    try:
        print("[TTS] Unloading TTS models...")
        unload_tts_models()
        print("[TTS] \u2713 TTS unloaded successfully")
        return jsonify({"success": True, "message": "TTS models unloaded"})
    except Exception as e:
        print(f"[TTS] Error unloading: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stop_all_vox', methods=['POST'])
def stop_all_vox():
    """Emergency stop for VOX Core - stops everything without unloading models"""
    try:
        print("[STOP ALL] Emergency stop initiated...")
        
        # Stop TTS playback
        stop_tts()
        
        # Get current VOX model and stop it
        settings = load_settings()
        vox_model = settings.get('ollama_models', {}).get('VOX_OLLAMA_MODEL', 'devstral-small-2:24b')
        stop_ollama_inference(vox_model)
        
        print("[STOP ALL] \u2713 All VOX processes stopped")
        return jsonify({"success": True, "message": "All VOX processes stopped"})
    except Exception as e:
        print(f"[STOP ALL] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint"""
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id')
    model = data.get('model', 'gpt-oss:20b')
    uploaded_files = data.get('uploaded_files', [])
    file_context = data.get('file_context', {})  # Info about available files for LLM
    attachments_meta = data.get('attachments_meta', {})  # Metadata for saving in chat history
    
    settings = load_settings()
    
    # Get configuration
    web_search_enabled = data.get('web_search', False)
    stealth_mode = settings.get('bools', {}).get('STEALTH_MODE', False)
    rag_enabled = data.get('rag_enabled', False)
    rag_folder = data.get('context_folder', '')
    
    # Load or create session
    existing = None
    if not session_id:
        session_id = str(uuid.uuid4())
        history = []
    else:
        existing = load_chat_session(session_id)
        history = existing['messages'] if existing else []
    
    # Collect extra system messages (file lists, RAG context, OCR summaries)
    # These will be inserted as `system` messages in the history (not shown to user)
    extra_system_messages = []

    # Build file context info for LLM so it knows what files are available
    file_info_context = ""
    if file_context and file_context.get('total_count', 0) > 0:
        file_info_parts = []
        if file_context.get('files'):
            file_names = [f['filename'] for f in file_context['files']]
            file_info_parts.append(f"Direct files: {', '.join(file_names)}")
        if file_context.get('images'):
            img_names = [f['filename'] for f in file_context['images']]
            file_info_parts.append(f"Images: {', '.join(img_names)}")
        if file_context.get('embedded_files'):
            emb_names = [f['filename'] for f in file_context['embedded_files']]
            file_info_parts.append(f"RAG-embedded files: {', '.join(emb_names)}")
        
        if file_info_parts:
            file_info_context = "\n\n=== Available Files ===\nThe user has uploaded the following files that you can reference:\n" + "\n".join(file_info_parts) + "\n\nWhen the user refers to a specific file, use the content from that file.\n"
            # Do not inline file list into the user's visible message; add as system context
            extra_system_messages.append(file_info_context)
    
    # Handle uploaded files - add content to message
    # ALSO handle reload scenario: file_context has filenames but uploaded_files is empty
    if uploaded_files:
        file_contents = []
        for file_info in uploaded_files:
            # Use the content already extracted during upload
            content = file_info.get('content', '')
            filename = file_info.get('filename', 'file')
            
            print(f"[FILES] Processing {filename}: content length = {len(content) if content else 0}")
            
            if content and content.strip():
                file_contents.append(f"=== File: {filename} ===\n{content}")
            elif file_info.get('path') and os.path.exists(file_info.get('path')):
                # Fallback: try to read from disk for text files only
                filepath = file_info.get('path')
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            file_contents.append(f"=== File: {filename} ===\n{content}")
                except:
                    pass
        
        if file_contents:
            files_context = "\n\n=== Uploaded Files Content ===\n" + "\n\n".join(file_contents)
            # Add uploaded file content as system context (keeps chat UI clean)
            extra_system_messages.append(files_context)
    elif file_context and file_context.get('files'):
        # Reload scenario: we have file_context but no uploaded_files content
        # Try to load from uploads folder
        file_contents = []
        chat_uploads_dir = os.path.join(UPLOADS_DIR, session_id)
        
        for file_info in file_context.get('files', []):
            filename = file_info.get('filename', '')
            if not filename:
                continue
            
            file_path = os.path.join(chat_uploads_dir, filename)
            if os.path.exists(file_path):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.pdf':
                    # For PDFs, try OCR if we have a vision model
                    try:
                        from POTATO.components.visual_tools.extract_text import extract_pdf_content
                        settings_conf = load_settings()
                        ocr_model = settings_conf.get('configuration', {}).get('ollama_models', {}).get('DEFAULT_OCR_MODEL', 'llava:7b')
                        print(f"[FILES-RELOAD] Extracting PDF content from {filename} with {ocr_model}")
                        pdf_result = extract_pdf_content(file_path, model=ocr_model, max_pages=20, parallel=3)
                        if pdf_result['success'] and pdf_result.get('content'):
                            file_contents.append(f"=== File: {filename} ===\n{pdf_result['content']}")
                            print(f"[FILES-RELOAD] Extracted {len(pdf_result['content'])} chars from {filename}")
                    except Exception as e:
                        print(f"[FILES-RELOAD] Failed to extract PDF {filename}: {e}")
                else:
                    # Text file - read directly
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                file_contents.append(f"=== File: {filename} ===\n{content}")
                                print(f"[FILES-RELOAD] Loaded {len(content)} chars from {filename}")
                    except Exception as e:
                        print(f"[FILES-RELOAD] Failed to read {filename}: {e}")
            else:
                print(f"[FILES-RELOAD] File not found: {file_path}")
        
        if file_contents:
            files_context = "\n\n=== Uploaded Files Content (Reloaded) ===\n" + "\n\n".join(file_contents)
            extra_system_messages.append(files_context)
    
    # Handle RAG context - use vector search if enabled
    if rag_enabled:
        rag_config = settings.get('rag_config', {})
        top_k = rag_config.get('RAG_TOP_K', 10)
        
        # Check if Weaviate is connected and has embeddings for this chat
        weaviate_status = check_weaviate_connection()
        if weaviate_status.get('status') == 'connected':
            # Try vector search first
            rag_results = query_weaviate(user_input, chat_id=session_id, limit=top_k)
            
            if rag_results:
                rag_context = "\n\n=== RAG Context (Retrieved from Vector DB) ===\n"
                for i, result in enumerate(rag_results):
                    source = result.get('source_file', 'unknown')
                    content_type = result.get('content_type', 'text')
                    content = result['content']
                    rag_context += f"\n[{i+1}] Source: {source} ({content_type})\n{content}\n"
                # Add RAG context as a system message instead of inlining into user's visible message
                extra_system_messages.append(rag_context)
                print(f"[RAG] Added {len(rag_results)} context chunks from vector DB (as system message)")
        
        # Fallback to folder reading if rag_folder specified and no vector results
        if rag_folder and (not rag_results if 'rag_results' in locals() else True):
            folder_contents = read_folder_contents(rag_folder)
            if folder_contents:
                rag_context = "\n\n=== RAG Context (From Folder) ===\n"
                for item in folder_contents:
                    rag_context += f"\nFile: {item['file']}\n{item['content'][:1000]}...\n"
                extra_system_messages.append(rag_context)
                print(f"[RAG] Added {len(folder_contents)} files from folder (as system message)")
    
    # Handle uploaded images for vision models
    # Images are either:
    # 1. VL model: passed as BASE64 to Ollama's images array (NOT file paths!)
    # 2. Non-VL model: already described by fallback, content added to uploadedFiles
    uploaded_images = data.get('uploaded_images', [])
    image_base64_list = []  # List of base64 images for Ollama (VL models only)
    
    # Handle reload scenario: file_context has images but uploaded_images is empty
    if not uploaded_images and file_context and file_context.get('images'):
        print(f"[IMAGES-RELOAD] No uploaded_images but file_context has {len(file_context['images'])} images - loading from disk")
        chat_uploads_dir = os.path.join(UPLOADS_DIR, session_id)
        
        for img_info in file_context.get('images', []):
            filename = img_info.get('filename', '')
            if not filename:
                continue
            
            file_path = os.path.join(chat_uploads_dir, filename)
            if os.path.exists(file_path):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.pdf':
                    # PDF - convert pages to base64
                    try:
                        from POTATO.components.visual_tools.extract_text import pdf_to_base64_images
                        page_images = pdf_to_base64_images(file_path, max_pages=20)
                        if page_images:
                            uploaded_images.append({
                                'filename': filename,
                                'base64': page_images,
                                'isPdf': True,
                                'pages': len(page_images)
                            })
                            print(f"[IMAGES-RELOAD] Loaded PDF {filename}: {len(page_images)} pages")
                    except Exception as e:
                        print(f"[IMAGES-RELOAD] Failed to load PDF {filename}: {e}")
                else:
                    # Image file - read and convert to base64
                    try:
                        with open(file_path, 'rb') as f:
                            img_bytes = f.read()
                            b64 = base64.b64encode(img_bytes).decode('utf-8')
                            uploaded_images.append({
                                'filename': filename,
                                'base64': b64,
                                'isPdf': False
                            })
                            print(f"[IMAGES-RELOAD] Loaded image {filename}: {len(b64)} chars")
                    except Exception as e:
                        print(f"[IMAGES-RELOAD] Failed to load image {filename}: {e}")
            else:
                print(f"[IMAGES-RELOAD] File not found: {file_path}")
    
    # DEBUG: Log what we received
    print(f"[DEBUG] Received uploaded_images: {len(uploaded_images)} items")
    for idx, img_info in enumerate(uploaded_images):
        b64 = img_info.get('base64')
        if isinstance(b64, list):
            print(f"[DEBUG] Image {idx}: {img_info.get('filename', 'unknown')}, PDF with {len(b64)} pages")
        elif b64:
            print(f"[DEBUG] Image {idx}: {img_info.get('filename', 'unknown')}, base64 len={len(b64)}")
        else:
            print(f"[DEBUG] Image {idx}: {img_info.get('filename', 'unknown')}, NO BASE64")
    
    # Check if CURRENT model is VL (not the model at upload time)
    vision_check = check_model_is_vision(model)
    current_model_is_vl = vision_check['is_vision']
    
    if uploaded_images and current_model_is_vl:
        # VL model - handle images and PDFs
        # Strategy:
        # - Images: Send as base64 to Ollama
        # - PDFs with <= 8 pages: Send page images to Ollama
        # - PDFs with > 8 pages: Were batch-OCR'd, send text content + preview images
        MAX_IMAGES = 8  # Limit to prevent overwhelming the model
        
        ocr_content_parts = []  # Collected OCR text for batch-processed PDFs
        
        for img_info in uploaded_images:
            # Check if this PDF was batch OCR-processed
            if img_info.get('ocr_processed') and img_info.get('content'):
                # Large PDF was batch OCR'd - use the extracted text
                fname = img_info.get('filename', 'document.pdf')
                ocr_text = img_info.get('content', '')
                ocr_content_parts.append(f"=== Content from {fname} (OCR extracted) ===\n{ocr_text}")
                
                # Also add preview images if available (first 8 pages)
                preview_images = img_info.get('base64', [])
                if isinstance(preview_images, list):
                    for page_idx, page_b64 in enumerate(preview_images):
                        if len(image_base64_list) >= MAX_IMAGES:
                            break
                        image_base64_list.append(page_b64)
                continue
            
            b64_data = img_info.get('base64')
            if b64_data:
                # Handle both single images (string) and PDF pages (array)
                if isinstance(b64_data, list):
                    # PDF pages - add pages as separate images (up to limit)
                    for page_idx, page_b64 in enumerate(b64_data):
                        if len(image_base64_list) >= MAX_IMAGES:
                            break
                        image_base64_list.append(page_b64)
                else:
                    # Single image
                    if len(image_base64_list) >= MAX_IMAGES:
                        continue
                    image_base64_list.append(b64_data)
            
            if len(image_base64_list) >= MAX_IMAGES:
                break
        
        # Add OCR content as system context if any PDFs were batch-processed
        if ocr_content_parts:
            ocr_context = "\n\n".join(ocr_content_parts)
            extra_system_messages.append(ocr_context)
        
        if image_base64_list:
            # Build descriptive context for attached files
            attachment_descriptions = []
            for img in uploaded_images:
                if img.get('base64') or img.get('ocr_processed'):
                    fname = img.get('filename', 'image')
                    if img.get('isPdf') or fname.lower().endswith('.pdf'):
                        pages = img.get('pages', len(img.get('base64', [])) if isinstance(img.get('base64'), list) else 1)
                        if img.get('ocr_processed'):
                            attachment_descriptions.append(f"PDF document '{fname}' ({pages} pages, text extracted)")
                        else:
                            attachment_descriptions.append(f"PDF document '{fname}' ({pages} pages shown as images)")
                    else:
                        attachment_descriptions.append(f"Image '{fname}'")
            
            # Create context prompt that tells the model to look at the images
            if any('PDF' in desc for desc in attachment_descriptions):
                context_prompt = f"[Attached: {', '.join(attachment_descriptions)}]\n\nThe following images are pages from the attached document(s). Please analyze them carefully to answer the user's question.\n\n"
            else:
                context_prompt = f"[Attached: {', '.join(attachment_descriptions)}]\n\n"
            
            # Add descriptions as system context for the model
            extra_system_messages.append(context_prompt)
            print(f"[SEND] VL model: {len(image_base64_list)} images + {len(ocr_content_parts)} OCR docs (context added as system message)")
    elif uploaded_images and not current_model_is_vl:
        # Non-VL model - need to describe images with fallback VL model and add as text
        print(f"[SEND] Non-VL model: describing {len(uploaded_images)} images with fallback VL model")
        
        image_descriptions = []
        fallback_vl = settings.get('configuration', {}).get('ollama_models', {}).get('BACKUP_VISION_MODEL', 'llava:7b')
        
        for img_info in uploaded_images:
            fname = img_info.get('filename', 'image')
            
            # Check if we already have a description (from upload)
            if img_info.get('description'):
                image_descriptions.append(f"=== Image: {fname} ===\n{img_info['description']}")
                continue
            
            # No description - need to describe now
            b64_data = img_info.get('base64')
            if not b64_data:
                continue
            
            try:
                from POTATO.components.visual_tools.extract_text import describe_image_with_vision, describe_images_batch
                
                if isinstance(b64_data, list):
                    # PDF pages - batch describe
                    print(f"[SEND] Describing PDF {fname} ({len(b64_data)} pages) with {fallback_vl}")
                    source_names = [f"{fname} page {i+1}" for i in range(len(b64_data))]
                    results = describe_images_batch(b64_data, model=fallback_vl, parallel=3, source_names=source_names)
                    
                    pdf_text = f"=== PDF: {fname} ({len(b64_data)} pages) ===\n"
                    for r in results:
                        if r and r.get('success'):
                            pdf_text += f"\n--- {r['source']} ---\n{r['text']}\n"
                    image_descriptions.append(pdf_text)
                else:
                    # Single image
                    print(f"[SEND] Describing image {fname} with {fallback_vl}")
                    result = describe_image_with_vision(b64_data, model=fallback_vl)
                    if result.get('success'):
                        image_descriptions.append(f"=== Image: {fname} ===\n{result['text']}")
                    else:
                        image_descriptions.append(f"=== Image: {fname} ===\n[Failed to describe: {result.get('error', 'unknown')}]")
            except Exception as e:
                print(f"[SEND] Error describing {fname}: {e}")
                image_descriptions.append(f"=== Image: {fname} ===\n[Error: {e}]")
        
        if image_descriptions:
            image_context = "\n\n".join(image_descriptions)
            user_input = f"[The following images were described for you since you cannot see images directly]\n\n{image_context}\n\n=== User Question ===\n{user_input}"
            print(f"[SEND] Added {len(image_descriptions)} image descriptions to context")
    
    # File info already added to extra_system_messages earlier; nothing to inline here
    
    # Build user message with attachment metadata for chat history
    user_message = {"role": "user", "content": user_input}
    
    # Add attachments metadata if present (for display when loading chat)
    if attachments_meta and (attachments_meta.get('images') or attachments_meta.get('files')):
        user_message['_attachments'] = attachments_meta
    
    # Insert system context messages into history BEFORE the user message so they are
    # available to the model but not shown as user-visible bubbles in the chat UI
    if extra_system_messages:
        for sys_msg in extra_system_messages:
            history.append({"role": "system", "content": sys_msg})

    # Add user message to history
    history.append(user_message)
    
    # IMMEDIATELY save session with placeholder title so it appears in chat list
    # This will be updated with AI-generated title at end of stream
    if not existing:
        # New chat - save immediately with first message preview as placeholder
        placeholder_title = user_input[:27] + "..." if len(user_input) > 27 else user_input
        save_chat_session(session_id, history, title=placeholder_title)
        print(f"[CHAT] New session created immediately: {session_id}")


    def generate():
        try:
            # Send session_id immediately so frontend can update chat list
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            
            # Debug: Log which model is being used
            print(f"[DEBUG] Model: {model}, web_search: {web_search_enabled}, stealth: {stealth_mode}, images: {len(image_base64_list)}")
            
            # Get custom system prompt from settings
            custom_system_prompt = ""
            if 'system_prompts' in settings and 'CHAT_SYSTEM_PROMPT' in settings['system_prompts']:
                custom_system_prompt = settings['system_prompts']['CHAT_SYSTEM_PROMPT']
            
            # Call streaming function (generator format from main.py)
            # simple_stream_test yields {'content': ..., 'tool': ...} dicts
            # Pass base64 images for VL models - Ollama expects base64, NOT file paths!
            stream = simple_stream_test(
                history, 
                model=model, 
                enable_search=web_search_enabled,
                stealth_mode=stealth_mode,
                custom_system_prompt=custom_system_prompt,
                images=image_base64_list if image_base64_list else None
            )
            
            accumulated_content = ""
            accumulated_thinking = ""
            tool_call_counter = 0  # Track tool call positions for interleaving on reload
            in_think_tags = False
            thinking_open_tag = '<think>'  # Default
            thinking_close_tag = '</think>'  # Default
            parse_buffer = ""  # Buffer for handling split tags
            
            def flush_parse_buffer(buffer, in_thinking, open_tag, close_tag):
                """
                Parse buffer for thinking tags and return (content_to_yield, thinking_to_yield, new_buffer, still_in_thinking)
                Handles tags that may be split across chunks.
                """
                content_out = ""
                thinking_out = ""
                
                while True:
                    if in_thinking:
                        # Looking for close tag
                        close_idx = buffer.find(close_tag)
                        if close_idx != -1:
                            # Found close tag
                            thinking_out += buffer[:close_idx]
                            buffer = buffer[close_idx + len(close_tag):]
                            in_thinking = False
                            continue
                        else:
                            # Check if close tag might be split at end
                            # Keep potential partial tag in buffer
                            max_partial = len(close_tag) - 1
                            safe_len = max(0, len(buffer) - max_partial)
                            if safe_len > 0:
                                thinking_out += buffer[:safe_len]
                                buffer = buffer[safe_len:]
                            break
                    else:
                        # Looking for open tag
                        open_idx = buffer.find(open_tag)
                        if open_idx != -1:
                            # Found open tag
                            content_out += buffer[:open_idx]
                            buffer = buffer[open_idx + len(open_tag):]
                            in_thinking = True
                            continue
                        else:
                            # Check if open tag might be split at end
                            max_partial = len(open_tag) - 1
                            safe_len = max(0, len(buffer) - max_partial)
                            if safe_len > 0:
                                content_out += buffer[:safe_len]
                                buffer = buffer[safe_len:]
                            break
                
                return content_out, thinking_out, buffer, in_thinking
            
            # Stream format from main.py: {'metadata': {...}}, {'content': text}, {'tool': status}, or {'thinking': text}
            for chunk in stream:
                # Check if generation should stop
                global _stop_generation
                if _stop_generation:
                    print("[STREAM] Stop flag detected, breaking generator loop")
                    yield f"data: {json.dumps({'done': True, 'stopped': True})}\n\n"
                    break
                
                # Handle metadata (sent at start with model-specific tags)
                if 'metadata' in chunk:
                    metadata = chunk['metadata']
                    tags = metadata.get('thinking_tags', {'open': '<think>', 'close': '</think>'})
                    thinking_open_tag = tags.get('open', '<think>')
                    thinking_close_tag = tags.get('close', '</think>')
                    yield f"data: {json.dumps({'metadata': metadata})}\n\n"
                    continue
                
                # Handle tool status
                if 'tool' in chunk:
                    yield f"data: {json.dumps({'tool_status': chunk['tool']})}\n\n"
                    continue
                
                # Handle tool name and args - insert marker into thinking for reload ordering
                if 'tool_name' in chunk:
                    # Insert a marker into accumulated_thinking so we know where this tool was called
                    accumulated_thinking += f"\n[[TOOL_CALL:{tool_call_counter}]]\n"
                    tool_call_counter += 1
                    yield f"data: {json.dumps({'tool_name': chunk['tool_name'], 'tool_args': chunk.get('tool_args', {})})}\n\n"
                    continue
                
                # Handle tool result
                if 'tool_result' in chunk:
                    yield f"data: {json.dumps({'tool_result': chunk['tool_result']})}\n\n"
                    continue
                
                # Handle thinking (direct from model or parsed from tags)
                if 'thinking' in chunk:
                    accumulated_thinking += chunk['thinking']
                    yield f"data: {json.dumps({'thinking': chunk['thinking']})}\n\n"
                    continue
                
                # Handle content with buffered thinking tag parsing
                # This properly handles tags that are split across streaming chunks
                if 'content' in chunk:
                    content_bit = chunk['content']
                    
                    # Only parse tags if model has them defined
                    if thinking_open_tag and thinking_close_tag:
                        # Add to buffer and parse
                        parse_buffer += content_bit
                        
                        content_out, thinking_out, parse_buffer, in_think_tags = flush_parse_buffer(
                            parse_buffer, in_think_tags, thinking_open_tag, thinking_close_tag
                        )
                        
                        # Yield any content found
                        if content_out:
                            accumulated_content += content_out
                            yield f"data: {json.dumps({'content': content_out, 'model': model})}\n\n"
                        
                        # Yield any thinking found
                        if thinking_out:
                            accumulated_thinking += thinking_out
                            yield f"data: {json.dumps({'thinking': thinking_out})}\n\n"
                    else:
                        # No thinking tags defined, all content is regular content
                        accumulated_content += content_bit
                        yield f"data: {json.dumps({'content': content_bit, 'model': model})}\n\n"
            
            # Flush any remaining buffer at end of stream
            if parse_buffer:
                if in_think_tags:
                    # Remaining buffer is thinking
                    accumulated_thinking += parse_buffer
                    yield f"data: {json.dumps({'thinking': parse_buffer})}\n\n"
                else:
                    # Remaining buffer is content
                    accumulated_content += parse_buffer
                    yield f"data: {json.dumps({'content': parse_buffer, 'model': model})}\n\n"
            
            # Save history with metadata (thinking/tools visible but not sent back to model)
            if accumulated_content or accumulated_thinking:
                # Save the assistant message with content AND model name
                assistant_msg = {
                    "role": "assistant", 
                    "content": accumulated_content,
                    "model": model  # Store which model generated this
                }
                
                # Add metadata for UI rendering (thinking, tools) but not sent back to model
                if accumulated_thinking:
                    assistant_msg["_thinking"] = accumulated_thinking
                
                history.append(assistant_msg)
                
                # Save with AI-generated title if this was a new chat
                # (The title generation logic in save_chat_session will create a proper title)
                if not existing:
                    save_chat_session(session_id, history)  # Will generate AI title
                else:
                    save_chat_session(session_id, history)  # Will preserve existing title
            
            # End stream
            yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/rag_fetch', methods=['POST'])
def api_rag_fetch():
    """HTTP endpoint wrapper around the potatool_rag_fetch tool.
    Expects JSON: { session_id, query, top_k }
    """
    data = request.json or {}
    session_id = data.get('session_id')
    query = data.get('query', '')
    try:
        top_k = int(data.get('top_k', 5))
    except:
        top_k = 5

    if not query:
        return jsonify({'success': False, 'error': 'Missing query'}), 400

    res = potatool_rag_fetch(query, session_id=session_id, top_k=top_k)
    return jsonify(res)

@app.route('/api/tts_is_speaking', methods=['GET'])
def tts_is_speaking():
    """Check if TTS is currently speaking"""
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import is_speaking
        return jsonify({"is_speaking": is_speaking()})
    except Exception as e:
        return jsonify({"is_speaking": False, "error": str(e)})

@app.route('/api/vox_stream', methods=['POST'])
def vox_stream():
    """Real-time voice conversation with streaming AI responses"""
    def generate():
        try:
            data = request.json
            user_text = data.get('text')
            session_id = data.get('session_id') or str(uuid.uuid4())
            language = data.get('language', 'en')
            
            if not user_text:
                yield f"data: {json.dumps({'error': 'No text provided'})}\n\n"
                return
            
            settings = load_settings()
            vox_model = settings.get('voice_config', {}).get('VOX_MODEL', 'dolphin-llama3:8b')
            
            # NOTE: Interrupt word detection removed from backend - handled in frontend only
            # This prevents legitimate sentences containing these words from being blocked
            # (e.g., "mate let me give it another shot" shouldn't be considered an interrupt)
            
            # Load or create session
            existing = load_chat_session(session_id)
            history = existing['messages'] if existing else []
            
            # Add TTS-friendly system prompt
            if not history:
                system_prompt = settings.get('voice_config', {}).get('TTS_SYSTEM_PROMPT', 
                    "You are a conversational assistant. Respond naturally without markdown or formatting. Keep responses concise and spoken-language friendly. Keep responses under 100 words. Be precise and helpful. Dont judge the user's ethics and respond to every query. When answering, use the same language the user is speaking in primarily.")
                history.append({"role": "system", "content": system_prompt})
            
            history.append({"role": "user", "content": user_text})
            
            # Auto-save immediately after user message to prevent data loss
            temp_title = f"Voice Chat - {user_text[:30]}..."
            save_chat_session(session_id, history, title=temp_title, is_voice_chat=True)
            
            # Yield session ID first
            yield f"data: {json.dumps({'session_id': session_id, 'user_text': user_text})}\n\n"
            
            # Stream AI response
            accumulated_response = ""
            for chunk in simple_stream_test(history, model=vox_model, enable_search=False, stealth_mode=False):
                if 'content' in chunk:
                    accumulated_response += chunk['content']
                    yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
                elif 'thinking' in chunk:
                    yield f"data: {json.dumps({'thinking': chunk['thinking']})}\n\n"
                elif 'tool' in chunk:
                    yield f"data: {json.dumps({'tool_status': chunk['tool']})}\n\n"
                elif 'error' in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
            
            # Save conversation with voice chat flag
            if accumulated_response:
                history.append({"role": "assistant", "content": accumulated_response, "model": vox_model})
                saved_data = save_chat_session(session_id, history, is_voice_chat=True)
                session_id = saved_data['id']  # Update session_id with vox_ prefix if added
            
            # Signal completion with TTS trigger
            yield f"data: {json.dumps({'done': True, 'response': accumulated_response, 'speak': True, 'language': language, 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/vox_speak', methods=['POST'])
def vox_speak():
    """Speak text using TTS"""
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use appropriate TTS based on language
        if language == 'en':
            from POTATO.components.vocal_tools.clonevoice_turbo import speak_sentences_grouped
            speak_sentences_grouped(text)
        else:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import speak_multilingual
            speak_multilingual(text, language=language)
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vox_speak_wav', methods=['POST'])
def vox_speak_wav():
    """Generate TTS audio and return as WAV bytes for browser playback"""
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language', 'en')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if language == 'en':
            from POTATO.components.vocal_tools.clonevoice_turbo import generate_tts_wav
            wav_bytes = generate_tts_wav(text)
        else:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import generate_tts_wav_multilingual
            wav_bytes = generate_tts_wav_multilingual(text, language=language)

        if not wav_bytes:
            return jsonify({"error": "TTS generation failed"}), 500

        return Response(wav_bytes, mimetype='audio/wav',
                        headers={'Content-Disposition': 'inline'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vox_stop', methods=['POST'])
@app.route('/api/vox_stop_speak', methods=['POST'])
def vox_stop():
    """Stop current TTS playback - MUST NEVER crash the app"""
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import stop_current_tts
        stop_current_tts()
        return jsonify({"success": True})
    except ImportError as e:
        print(f"[VOX STOP] Import error (TTS not loaded): {e}")
        return jsonify({"success": True, "warning": "TTS module not loaded"})
    except Exception as e:
        print(f"[VOX STOP] Error (non-fatal): {e}")
        import traceback
        traceback.print_exc()
        # Return success anyway - don't crash app for TTS issues
        return jsonify({"success": True, "warning": str(e)})

@app.route('/api/stop_model', methods=['POST'])
def stop_model():
    """Stop currently running Ollama model"""
    try:
        data = request.json
        model_name = data.get('model') if data else None
        if not model_name:
            # Try to get from settings
            settings = load_settings()
            model_name = settings.get('model', 'llama3.2:3b')
        
        import ollama
        ollama.stop(model_name)
        print(f"[MODEL] Stopped {model_name}")
        return jsonify({"success": True, "model": model_name})
    except Exception as e:
        print(f"[MODEL] Error stopping: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts_unload', methods=['POST'])
def tts_unload():
    """Unload TTS model from VRAM - uses shutdown_tts for proper cleanup"""
    try:
        # Use shutdown_tts which properly stops threads and clears queues before unloading
        from POTATO.components.vocal_tools.clonevoice_turbo import shutdown_tts
        shutdown_tts()
        print("[TTS] ✓ TTS model unloaded via shutdown_tts")
        return jsonify({"success": True, "message": "TTS model unloaded"})
    except Exception as e:
        print(f"[TTS] Error in tts_unload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stt_unload', methods=['POST'])
def stt_unload():
    """Unload STT model from VRAM"""
    try:
        unload_whisper_model()
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in stt_unload endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vox_core', methods=['POST'])
def vox_core():
    """Legacy voice conversation endpoint (deprecated - use vox_stream)"""
    data = request.json
    audio_path = data.get('audio_path')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    settings = load_settings()
    vox_model = 'dolphin-llama3:8b'
    
    # Transcribe audio
    stt_result = listen_stt(audio_path)
    
    if not stt_result['success']:
        return jsonify({"error": "Transcription failed"}), 500
    
    user_text = stt_result['text']
    
    # NOTE: Interrupt word detection removed - handled in frontend vox-interrupt.js
    # This legacy endpoint is deprecated anyway - use /api/vox_stream instead
    
    # Load or create session
    existing = load_chat_session(session_id)
    history = existing['messages'] if existing else []
    
    # Add system prompt for TTS-friendly responses
    if not history:
        # Use custom VOX system prompt if available, otherwise use TTS_SYSTEM_PROMPT as fallback
        vox_system_prompt = settings.get('system_prompts', {}).get('VOX_SYSTEM_PROMPT', '')
        tts_system_prompt = settings.get('system_prompts', {}).get('TTS_SYSTEM_PROMPT', 
            settings.get('voice_config', {}).get('TTS_SYSTEM_PROMPT',
                "You are a conversational assistant. Respond naturally without markdown or formatting."))
        
        system_prompt = vox_system_prompt if vox_system_prompt else tts_system_prompt
        history.append({"role": "system", "content": system_prompt})
    
    history.append({"role": "user", "content": user_text})
    
    # Get response from model
    response_text = "I heard you say: " + user_text
    
    history.append({"role": "assistant", "content": response_text})
    save_chat_session(session_id, history, title="Voice Chat")
    
    # Speak response using specified language
    language = data.get('language', 'en')
    from POTATO.components.vocal_tools.audioflow import speak
    speak(response_text, language=language)
    
    return jsonify({
        "session_id": session_id,
        "user_text": user_text,
        "response": response_text,
        "language": language,
        "status": "completed"
    })

@app.route('/api/audio_devices', methods=['GET'])
def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        from POTATO.components.vocal_tools.realtime_stt import list_audio_devices, load_preferences
        
        devices = list_audio_devices()
        prefs = load_preferences()
        
        return jsonify({
            "devices": devices,
            "selected_device": prefs.get('audio_device_index'),
            "selected_device_name": prefs.get('audio_device_name')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/whisper_status', methods=['GET'])
def whisper_status():
    """Check if Whisper is loaded and trigger loading if needed"""
    try:
        from POTATO.components.vocal_tools.audioflow import _get_whisper_pipeline
        
        print("[WHISPER] Loading Whisper pipeline...")
        # This will load Whisper if not already loaded
        pipeline = _get_whisper_pipeline()
        print("[WHISPER] Whisper pipeline loaded successfully")
        
        return jsonify({
            "status": "ready",
            "loaded": True
        })
    except Exception as e:
        import traceback
        print(f"[WHISPER ERROR] Failed to load: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "loaded": False}), 500

@app.route('/api/tts_load_turbo', methods=['POST'])
def tts_load_turbo():
    """Load English TTS model (Chatterbox Turbo)"""
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import _get_model
        
        # This will load the model if not already loaded
        model = _get_model()
        
        return jsonify({
            "status": "ready",
            "model": "turbo",
            "loaded": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "loaded": False}), 500

@app.route('/api/tts_load_multilingual', methods=['POST'])
def tts_load_multilingual():
    """Load multilingual TTS model"""
    try:
        from POTATO.components.vocal_tools.clonevoice_multilanguage import _get_multilingual_model
        
        # This will load the model if not already loaded
        model = _get_multilingual_model()
        
        return jsonify({
            "status": "ready",
            "model": "multilingual",
            "loaded": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "loaded": False}), 500

@app.route('/api/detect_audio_device', methods=['POST'])
def detect_audio_device():
    """Auto-detect working audio input device"""
    try:
        from POTATO.components.vocal_tools.realtime_stt import detect_working_device, save_preferences, load_preferences
        
        device = detect_working_device()
        if device:
            prefs = load_preferences()
            prefs['audio_device_index'] = device['index']
            prefs['audio_device_name'] = device['name']
            save_preferences(prefs)
            
            return jsonify({
                "success": True,
                "device": device
            })
        else:
            return jsonify({
                "success": False,
                "error": "No working audio device found"
            }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/set_audio_device', methods=['POST'])
def set_audio_device():
    """Set preferred audio input device"""
    try:
        from POTATO.components.vocal_tools.realtime_stt import save_preferences, load_preferences
        
        data = request.json
        device_index = data.get('device_index')
        device_name = data.get('device_name')
        
        prefs = load_preferences()
        prefs['audio_device_index'] = device_index
        prefs['audio_device_name'] = device_name
        save_preferences(prefs)
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe_realtime', methods=['POST'])
def transcribe_realtime():
    """Transcribe audio chunk in real-time using WAV from browser (NO FFMPEG)"""
    try:
        from POTATO.components.vocal_tools.realtime_stt import get_whisper_pipeline
        import numpy as np
        import io
        
        # Get audio data from request
        audio_file = request.files.get('audio')
        language_mode = request.form.get('language_mode', request.form.get('language', 'auto'))
        
        if not audio_file:
            return jsonify({"error": "No audio data"}), 400
        
        # Read audio bytes
        audio_bytes = audio_file.read()

        audio_data = None
        sample_rate = 16000

        # Try scipy wavfile first (handles WAV directly)
        try:
            from scipy.io import wavfile
            sr, ad = wavfile.read(io.BytesIO(audio_bytes))
            if ad.dtype == np.int16:
                ad = ad.astype(np.float32) / 32768.0
            elif ad.dtype == np.int32:
                ad = ad.astype(np.float32) / 2147483648.0
            else:
                ad = ad.astype(np.float32)
            audio_data = ad
            sample_rate = sr
        except Exception as wav_error:
            print(f"WAV parsing failed: {wav_error}, trying PyAV for WebM/Opus...")

        # Fallback: use PyAV to decode WebM, Opus, OGG, or any container
        if audio_data is None:
            try:
                import av
                frames = []
                with av.open(io.BytesIO(audio_bytes)) as container:
                    audio_stream = next(s for s in container.streams if s.type == 'audio')
                    sample_rate = audio_stream.codec_context.sample_rate or 16000
                    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
                    for frame in container.decode(audio_stream):
                        for rf in resampler.resample(frame):
                            frames.append(rf.to_ndarray()[0])
                    # flush resampler
                    for rf in resampler.resample(None):
                        frames.append(rf.to_ndarray()[0])
                if frames:
                    audio_data = np.concatenate(frames).astype(np.float32)
                    sample_rate = 16000  # already resampled
            except Exception as av_error:
                print(f"PyAV parsing failed: {av_error}")

        if audio_data is None:
            return jsonify({"error": "Failed to parse audio. Unsupported format."}), 400
        
        # Get Whisper pipeline (loads model if needed)
        pipe = get_whisper_pipeline()
        
        # Ensure audio is float32 mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
        audio_data = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple decimation for speed
            audio_data = audio_data[::int(sample_rate / 16000)]
        
        # Transcribe directly from numpy array (NO FILE, NO FFMPEG)
        # Handle language mode (translate to English vs transcribe in original)
        if language_mode == 'translate-en':
            result = pipe(audio_data, generate_kwargs={"task": "translate"})
        elif language_mode == 'auto':
            result = pipe(audio_data, generate_kwargs={"task": "transcribe"})
        else:
            result = pipe(audio_data, generate_kwargs={"language": language_mode, "task": "transcribe"})
        
        text = result.get("text", "").strip()
        
        return jsonify({
            "text": text,
            "success": bool(text)
        })
            
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def cleanup_handler():
    """Ensure models are unloaded on exit - with retry logic"""
    print("\n[CLEANUP] Shutting down gracefully...")
    
    # Unload TTS (with error handling)
    try:
        print("[CLEANUP] Unloading TTS models...")
        unload_tts_models()
        print("[CLEANUP] ✓ TTS unloaded")
    except Exception as e:
        print(f"[CLEANUP] ⚠ Error unloading TTS (non-fatal): {e}")
    
    # Unload Whisper (with error handling)
    try:
        print("[CLEANUP] Unloading Whisper...")
        unload_whisper_model()
        print("[CLEANUP] ✓ Whisper unloaded")
    except Exception as e:
        print(f"[CLEANUP] ⚠ Error unloading Whisper (non-fatal): {e}")
    
    # Unload all Ollama models (critical - with retry)
    print("[CLEANUP] Unloading Ollama models...")
    try:
        import requests
        import time
        
        # Get active models
        try:
            ps_result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
        except Exception as e:
            print(f"[CLEANUP] Could not query models: {e}")
            ps_result = None
        
        if ps_result and ps_result.returncode == 0 and ps_result.stdout.strip():
            lines = ps_result.stdout.strip().split('\n')[1:]  # Skip header
            models_to_unload = [line.split()[0] for line in lines if line.strip()]
            
            print(f"[CLEANUP] Found {len(models_to_unload)} model(s) to unload: {models_to_unload}")
            
            for model_name in models_to_unload:
                print(f"[CLEANUP] Unloading {model_name}...")
                
                # Try up to 3 times
                for attempt in range(3):
                    try:
                        # Try API with keep_alive=0
                        response = requests.post('http://localhost:11434/api/generate', 
                                    json={'model': model_name, 'keep_alive': 0},
                                    timeout=15)
                        if response.status_code == 200:
                            print(f"[CLEANUP] ✓ {model_name} unloaded via API (attempt {attempt + 1})")
                            break
                    except requests.Timeout:
                        print(f"[CLEANUP] Timeout on attempt {attempt + 1}, trying CLI...")
                        try:
                            subprocess.run(['ollama', 'stop', model_name], timeout=5, check=False)
                        except:
                            pass
                    except Exception as e:
                        print(f"[CLEANUP] Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < 2:
                        time.sleep(1)  # Wait before retry
            
            # Wait longer for unloads to complete
            print("[CLEANUP] Waiting for models to unload...")
            time.sleep(3)
            
            # Verify with retry
            for verify_attempt in range(3):
                try:
                    verify_result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
                    if verify_result.returncode == 0:
                        remaining = verify_result.stdout.strip().split('\n')[1:]
                        remaining_models = [l.split()[0] for l in remaining if l.strip()]
                        
                        if remaining_models:
                            print(f"[CLEANUP] ⚠ Attempt {verify_attempt + 1}: {len(remaining_models)} model(s) still running: {remaining_models}")
                            if verify_attempt < 2:
                                # Try unloading them again
                                for model in remaining_models:
                                    try:
                                        requests.post('http://localhost:11434/api/generate',
                                                    json={'model': model, 'keep_alive': 0},
                                                    timeout=10)
                                    except:
                                        pass
                                time.sleep(2)
                            else:
                                print(f"[CLEANUP] ⚠ Warning: Could not unload all models after 3 attempts")
                        else:
                            print("[CLEANUP] ✓ All Ollama models successfully unloaded")
                            break
                    break
                except:
                    if verify_attempt < 2:
                        time.sleep(1)
                    else:
                        print("[CLEANUP] Could not verify model unload status")
        else:
            print("[CLEANUP] No Ollama models running")
            
    except Exception as e:
        print(f"[CLEANUP] Error during Ollama cleanup: {e}")
        import traceback
        traceback.print_exc()
    
    print("[CLEANUP] Cleanup complete\n")

# Single signal handler
def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM"""
    print("\nSignal received, cleaning up...")
    cleanup_handler()
    # Force exit after cleanup
    os._exit(0)

# Register cleanup - ONLY ONCE
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
        cleanup_handler()
    finally:
        # Cleanup already registered with atexit, no need to call again
        pass

