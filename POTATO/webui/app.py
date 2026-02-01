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
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    title = json.load(f).get('title')
            except:
                pass
        if not title:
            # Generate title from first user msg using AI
            for m in messages:
                if m['role'] == 'user' and m['content']:
                    try:
                        # Use Ollama to generate a concise title
                        import ollama
                        response = ollama.chat(
                            model='qwen2.5-coder:7b',  # Fast model for title generation
                            messages=[{
                                'role': 'system',
                                'content': 'Generate a very short title (max 25 chars) for this conversation. Just output the title, nothing else.'
                            }, {
                                'role': 'user',
                                'content': m['content'][:200]  # First 200 chars of message
                            }]
                        )
                        title = response['message']['content'].strip().strip('"\'')[:30]
                    except:
                        # Fallback to truncated first message
                        title = m['content'][:27] + "..." if len(m['content']) > 27 else m['content']
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
        if model_name:
            # First stop the model
            subprocess.run(['ollama', 'stop', model_name], timeout=5)
            # Then explicitly unload it by setting keep_alive to 0
            try:
                ollama.chat(model=model_name, messages=[], keep_alive=0)
            except:
                pass  # Model might already be stopped
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

def start_mcp_server():
    """Start the MCP SearXNG server"""
    global _mcp_server_process
    
    if _mcp_server_process and _mcp_server_process.poll() is None:
        print("MCP server already running")
        return
    
    try:
        mcp_script = os.path.join(BASE_DIR, 'MCP', 'searx_mcp.py')
        if not os.path.exists(mcp_script):
            print(f"MCP script not found: {mcp_script}")
            return
        
        # Get Python executable from current environment
        python_exe = os.path.join(BASE_DIR, '.venv', 'Scripts', 'python.exe')
        if not os.path.exists(python_exe):
            python_exe = 'python'  # Fallback to system python
        
        _mcp_server_process = subprocess.Popen(
            [python_exe, mcp_script],
            cwd=os.path.join(BASE_DIR, 'MCP'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        print(f"MCP server started with PID: {_mcp_server_process.pid}")
    except Exception as e:
        print(f"Error starting MCP server: {e}")

def stop_mcp_server():
    """Stop the MCP SearXNG server"""
    global _mcp_server_process
    
    if _mcp_server_process and _mcp_server_process.poll() is None:
        try:
            _mcp_server_process.terminate()
            _mcp_server_process.wait(timeout=5)
            print("MCP server stopped")
        except subprocess.TimeoutExpired:
            _mcp_server_process.kill()
            print("MCP server killed")
        except Exception as e:
            print(f"Error stopping MCP server: {e}")
        finally:
            _mcp_server_process = None

def cleanup_on_exit():
    """Cleanup function to run on app shutdown"""
    print("\nCleaning up...")
    
    # Stop MCP server
    stop_mcp_server()
    
    # Unload all models
    try:
        unload_tts_models()
    except:
        pass
    
    try:
        unload_whisper_model()
    except:
        pass
    
    print("Cleanup complete")

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
    """Get list of available Ollama models"""
    models = get_ollama_models()
    return jsonify({"models": models})

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
    existing = None
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
            
            # Start/stop MCP server based on web search setting AND stealth mode
            if web_search_enabled and not stealth_mode:
                start_mcp_server()
            else:
                stop_mcp_server()
            
            # Debug: Log which model is being used
            print(f"[DEBUG] Model: {model}, web_search: {web_search_enabled}, stealth: {stealth_mode}")
            
            # Call streaming function (generator format from main.py)
            # simple_stream_test yields {'content': ..., 'tool': ...} dicts
            stream = simple_stream_test(
                history, 
                model=model, 
                enable_search=web_search_enabled,
                stealth_mode=stealth_mode
            )
            
            accumulated_content = ""
            accumulated_thinking = ""
            in_think_tags = False
            thinking_open_tag = '<think>'  # Default
            thinking_close_tag = '</think>'  # Default
            
            # Stream format from main.py: {'metadata': {...}}, {'content': text}, {'tool': status}, or {'thinking': text}
            for chunk in stream:
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
                
                # Handle thinking (direct from model or parsed from tags)
                if 'thinking' in chunk:
                    accumulated_thinking += chunk['thinking']
                    yield f"data: {json.dumps({'thinking': chunk['thinking']})}\n\n"
                    continue
                
                # Handle content with dynamic thinking tag parsing
                if 'content' in chunk:
                    content_bit = chunk['content']
                    
                    # Only parse tags if model has them defined
                    if thinking_open_tag and thinking_close_tag:
                        # Check if we're entering/exiting think tags
                        if thinking_open_tag in content_bit:
                            in_think_tags = True
                            # Split content before and after open tag
                            parts = content_bit.split(thinking_open_tag, 1)
                            if parts[0]:  # Content before opening tag
                                accumulated_content += parts[0]
                                yield f"data: {json.dumps({'content': parts[0], 'model': model})}\n\n"
                            content_bit = parts[1] if len(parts) > 1 else ""
                        
                        if thinking_close_tag in content_bit:
                            in_think_tags = False
                            # Split thinking and content after close tag
                            parts = content_bit.split(thinking_close_tag, 1)
                            if parts[0]:  # Thinking content
                                accumulated_thinking += parts[0]
                                yield f"data: {json.dumps({'thinking': parts[0]})}\n\n"
                            if len(parts) > 1 and parts[1]:  # Content after closing tag
                                accumulated_content += parts[1]
                                yield f"data: {json.dumps({'content': parts[1], 'model': model})}\n\n"
                        elif in_think_tags:
                            # We're inside think tags, accumulate thinking
                            accumulated_thinking += content_bit
                            yield f"data: {json.dumps({'thinking': content_bit})}\n\n"
                        else:
                            # Regular content outside think tags
                            accumulated_content += content_bit
                            yield f"data: {json.dumps({'content': content_bit, 'model': model})}\n\n"
                    else:
                        # No thinking tags defined, all content is regular content
                        accumulated_content += content_bit
                        yield f"data: {json.dumps({'content': content_bit, 'model': model})}\n\n"
            
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
                    "You are a conversational assistant. Respond naturally without markdown or formatting. Keep responses concise and spoken-language friendly.")
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
    """Unload TTS model from VRAM"""
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import unload_model
        unload_model()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stt_unload', methods=['POST'])
def stt_unload():
    """Unload STT model from VRAM"""
    try:
        from POTATO.components.vocal_tools.audioflow import unload_whisper
        unload_whisper()
        return jsonify({"success": True})
    except Exception as e:
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
    
    # Check for interrupt words
    interrupt_words = ['okay thanks', 'alright stop', 'okay enough', 'thank you', 'bye', 'done', 'shut up']
    if any(word in user_text.lower() for word in interrupt_words):
        from POTATO.components.vocal_tools.clonevoice_turbo import stop_current_tts
        stop_current_tts()
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
        
        try:
            # Try scipy wavfile first (handles WAV directly)
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            # Convert to float32 [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        except Exception as wav_error:
            print(f"WAV parsing failed: {wav_error}")
            return jsonify({"error": "Failed to parse audio. Please check microphone."}), 400
        
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

# Register cleanup handlers
atexit.register(cleanup_on_exit)

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM"""
    print("\nShutting down gracefully...")
    cleanup_on_exit()
    exit(0)

def cleanup_handler():
    """Ensure models are unloaded on exit"""
    print("\n[CLEANUP] Shutting down gracefully...")
    try:
        stop_tts()
        unload_tts_models()
        unload_whisper_model()
        unload_model()
        unload_vox()
        stop_inference()
        
    except:
        pass
    sys.exit(0)


# Register cleanup
atexit.register(cleanup_handler)
# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
        cleanup_on_exit()
        cleanup_handler()
    finally:
        cleanup_on_exit()
        cleanup_handler()

