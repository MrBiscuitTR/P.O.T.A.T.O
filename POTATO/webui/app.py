import os
import json
import time
import glob
import uuid
import psutil
import subprocess
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from pathlib import Path

# --- IMPORTS ---
from POTATO.main import simple_stream_test 
# Import audioflow functions only (no model loading at import time)
from POTATO.components.vocal_tools.audioflow import listen_stt
from POTATO.components.local_tools.embed import embed_to_weaviate
from POTATO.components.utilities.get_system_info import json_get_instant_system_info

# TTS models will be loaded on-demand, not at import
_tts_turbo_model = None
_tts_multilingual_model = None

# MCP server process
_mcp_server_process = None

# --- TOOL REGISTRY ---
TOOL_REGISTRY = {
    "get_current_weather": lambda location: f"The weather in {location} is 22°C and Sunny.",
}

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, 'config.json')
DATA_DIR = os.path.join(BASE_DIR, '.data')
USER_SETTINGS_PATH = os.path.join(DATA_DIR, 'usersettings.json')
CHATS_DIR = os.path.join(DATA_DIR, 'chats')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
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
def load_settings():
    """Load user settings, fallback to config.json"""
    settings = {}
    # Load default config first
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
    
    # Override with user settings
    if os.path.exists(USER_SETTINGS_PATH):
        try:
            with open(USER_SETTINGS_PATH, 'r') as f:
                user_settings = json.load(f)
                settings.update(user_settings)
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

def save_chat_session(session_id, messages, title=None):
    """Save chat session to file"""
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if not title:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    title = json.load(f).get('title')
            except:
                pass
        if not title:
            # Generate title from first user msg
            for m in messages:
                if m['role'] == 'user':
                    title = m['content'][:30] + "..." if len(m['content']) > 30 else m['content']
                    break
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

def get_all_chats(search_query=None):
    """Get all chat sessions, optionally filtered by search query"""
    files = glob.glob(os.path.join(CHATS_DIR, "*.json"))
    chats = []
    for p in files:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            
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
    """Stop ongoing Ollama inference"""
    try:
        if model_name:
            subprocess.run(['ollama', 'stop', model_name], timeout=5)
        else:
            # Stop all models
            subprocess.run(['ollama', 'ps'], capture_output=True, timeout=5)
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
    
    # Unload Turbo model
    if _tts_turbo_model:
        try:
            from POTATO.components.vocal_tools.clonevoice_turbo import unload_model
            unload_model()
        except Exception as e:
            print(f"Error unloading Turbo model: {e}")
        _tts_turbo_model = None
    
    # Unload Multilingual model
    if _tts_multilingual_model:
        try:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import unload_models
            unload_models()
        except Exception as e:
            print(f"Error unloading Multilingual model: {e}")
        _tts_multilingual_model = None
    
    print("All TTS models unloaded")

def unload_whisper_model():
    """Unload Whisper model from VRAM"""
    try:
        from POTATO.components.vocal_tools import audioflow
        if hasattr(audioflow, '_whisper_pipeline') and audioflow._whisper_pipeline is not None:
            del audioflow._whisper_pipeline
            audioflow._whisper_pipeline = None
            import torch
            torch.cuda.empty_cache()
            print("Whisper model unloaded")
    except Exception as e:
        print(f"Error unloading Whisper: {e}")

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

@app.route('/api/preferences', methods=['GET', 'POST'])
def handle_preferences():
    """Get or update UI preferences (selected model, language, etc.)"""
    prefs_path = os.path.join(DATA_DIR, 'preferences.json')
    
    if request.method == 'POST':
        data = request.json
        try:
            with open(prefs_path, 'w') as f:
                json.dump(data, f, indent=4)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
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
    """Get list of available Ollama models"""
    models = get_ollama_models()
    return jsonify({"models": models})

@app.route('/api/system_stats')
def system_stats():
    """Get system statistics"""
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
    """List all chat sessions"""
    search_query = request.args.get('search')
    chats = get_all_chats(search_query)
    return jsonify(chats)

@app.route('/api/chats/<session_id>', methods=['GET', 'DELETE'])
def chat_session_route(session_id):
    """Get or delete a specific chat session"""
    if request.method == 'DELETE':
        path = os.path.join(CHATS_DIR, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Not found"}), 404
    
    # GET
    chat = load_chat_session(session_id)
    if chat:
        return jsonify(chat)
    return jsonify({"error": "Not found"}), 404

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
    """Unload chat model from VRAM"""
    data = request.json or {}
    model_name = data.get('model')
    
    try:
        stop_ollama_inference(model_name)
        return jsonify({"success": True, "message": f"Model {model_name} unloaded"})
    except Exception as e:
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

@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint"""
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id')
    model = data.get('model', 'gpt-oss:20b')
    uploaded_files = data.get('uploaded_files', [])
    
    settings = load_settings()
    
    # Get configuration
    web_search_enabled = data.get('web_search', False)
    stealth_mode = settings.get('bools', {}).get('STEALTH_MODE', False)
    rag_enabled = data.get('rag_enabled', False)
    rag_folder = data.get('context_folder', '')
    
    # Load or create session
    if not session_id:
        session_id = str(uuid.uuid4())
        history = []
    else:
        existing = load_chat_session(session_id)
        history = existing['messages'] if existing else []
    
    # Handle uploaded files - add content to message
    if uploaded_files:
        file_contents = []
        for file_info in uploaded_files:
            filepath = file_info.get('path')
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents.append(f"File: {file_info.get('filename', filepath)}\n{content}")
                except:
                    pass
        
        if file_contents:
            files_context = "\n\n=== Uploaded Files ===\n" + "\n\n".join(file_contents)
            user_input = files_context + "\n\n" + user_input
    
    # Handle RAG context
    if rag_enabled and rag_folder:
        folder_contents = read_folder_contents(rag_folder)
        if folder_contents:
            rag_context = "\n\n=== RAG Context ===\n"
            for item in folder_contents:
                rag_context += f"\nFile: {item['file']}\n{item['content'][:500]}...\n"
            user_input = rag_context + "\n\n" + user_input
    
    # Add user message to history
    history.append({"role": "user", "content": user_input})
    
    def generate():
        try:
            # Call streaming function with thinking enabled
            stream = simple_stream_test(history, stream=True, think=True)
            
            accumulated_content = ""
            
            for chunk in stream:
                # Ollama streaming format: chunk contains 'message' dict with 'content', 'role', etc.
                if 'message' in chunk:
                    message = chunk['message']
                    
                    # Handle content
                    if 'content' in message:
                        content_bit = message['content']
                        accumulated_content += content_bit
                        yield f"data: {json.dumps({'content': content_bit, 'model': model})}\n\n"
                    
                    # Handle tool calls
                    if 'tool_calls' in message:
                        tool_calls = message['tool_calls']
                        for tool in tool_calls:
                            fn_name = tool.get('function', {}).get('name', '')
                            args = tool.get('function', {}).get('arguments', {})
                            yield f"data: {json.dumps({'tool_status': f'Calling {fn_name}...'})}\n\n"
                            
                            # Execute tool
                            if fn_name in TOOL_REGISTRY:
                                try:
                                    if isinstance(args, str):
                                        tool_args = json.loads(args)
                                    else:
                                        tool_args = args
                                    
                                    result = TOOL_REGISTRY[fn_name](**tool_args)
                                    yield f"data: {json.dumps({'tool_result': str(result)})}\n\n"
                                except Exception as e:
                                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                # Check if done
                if chunk.get('done', False):
                    break
            
            # Save history
            if accumulated_content:
                history.append({"role": "assistant", "content": accumulated_content})
                save_chat_session(session_id, history)
            
            # End stream
            yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/vox_core', methods=['POST'])
def vox_core():
    """Voice conversation endpoint"""
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
    
    # Check for interrupt words
    interrupt_words = ['okay thanks', 'stop', 'enough', 'thank you', 'bye', 'done']
    if any(word in user_text.lower() for word in interrupt_words):
        stop_tts()
        return jsonify({"status": "interrupted", "text": user_text})
    
    # Load or create session
    existing = load_chat_session(session_id)
    history = existing['messages'] if existing else []
    
    # Add system prompt for TTS-friendly responses
    if not history:
        system_prompt = settings.get('voice_config', {}).get('TTS_SYSTEM_PROMPT', 
            "You are a conversational assistant. Respond naturally without markdown or formatting.")
        history.append({"role": "system", "content": system_prompt})
    
    history.append({"role": "user", "content": user_text})
    
    # Get response from model (non-streaming for voice)
    # TODO: Implement actual ollama call
    response_text = "I heard you say: " + user_text
    
    history.append({"role": "assistant", "content": response_text})
    save_chat_session(session_id, history, title="Voice Chat")
    
    # Speak response using specified language
    language = data.get('language', 'en')
    speak_text(response_text, language=language)
    
    return jsonify({
        "session_id": session_id,
        "user_text": user_text,
        "response": response_text,
        "language": language,
        "status": "completed"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
