import os
import json
import time
import glob
import uuid
import psutil
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

# --- IMPORTS ---
# We import your function, but we will call it with specific parameters
from POTATO.main import simple_stream_test 
from POTATO.components.vocal_tools.clonevoice_turbo import speak_sentences_grouped, stop_current_tts, shutdown_tts

# --- TOOL REGISTRY ---
# Map the string names from LLM to actual python functions in your components
TOOL_REGISTRY = {
    # Example mapping:
    # "generate_image": components.visual_tools.image_gen.generate,
    "get_current_weather": lambda location: f"The weather in {location} is 22°C and Sunny.", # Mock for testing
}

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, 'config.json')
print(SETTINGS_PATH)
DATA_DIR = os.path.join(BASE_DIR, '.data')
USER_SETTINGS_PATH = os.path.join(DATA_DIR, 'usersettings.json')
CHATS_DIR = os.path.join(DATA_DIR, 'chats')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)

# --- HELPERS ---
def load_settings():
    settings = {}
    try:
        with open(SETTINGS_PATH, 'r') as f: settings = json.load(f)
    except: pass
    if os.path.exists(USER_SETTINGS_PATH):
        try:
            with open(USER_SETTINGS_PATH, 'r') as f: settings.update(json.load(f))
        except: pass
    return settings

def save_user_setting(key, value):
    current = {}
    if os.path.exists(USER_SETTINGS_PATH):
        try:
            with open(USER_SETTINGS_PATH, 'r') as f: current = json.load(f)
        except: pass
    current[key] = value
    with open(USER_SETTINGS_PATH, 'w') as f: json.dump(current, f, indent=4)

def save_chat_session(session_id, messages, title=None):
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if not title:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: title = json.load(f).get('title')
            except: pass
        if not title:
            # Generate title from first user msg
            for m in messages:
                if m['role'] == 'user':
                    title = m['content'][:30] + "..." if len(m['content']) > 30 else m['content']
                    break
            if not title: title = "New Session"
            
    data = {"id": session_id, "title": title, "last_updated": time.time(), "messages": messages}
    with open(path, 'w') as f: json.dump(data, f, indent=4)
    return data

def load_chat_session(session_id):
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f: return json.load(f)
    return None

def get_all_chats(search_query=None):
    files = glob.glob(os.path.join(CHATS_DIR, "*.json"))
    chats = []
    for p in files:
        try:
            with open(p, 'r') as f: data = json.load(f)
            # Filter
            if search_query:
                q = search_query.lower()
                in_content = any(q in m.get('content', '').lower() for m in data.get('messages', []))
                if q not in data.get('title', '').lower() and not in_content:
                    continue
            chats.append(data)
        except: pass
    chats.sort(key=lambda x: x['last_updated'], reverse=True)
    return chats

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.json
        for k, v in data.items(): save_user_setting(k, v)
        return jsonify({"status": "success"})
    return jsonify(load_settings())

@app.route('/api/system_stats')
def system_stats():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu, gpu_temp, vram = 0, 0, 0
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0].load * 100
            gpu_temp = gpus[0].temperature
            vram = gpus[0].memoryUtil*100
    except: pass
    return jsonify({"cpu": cpu, "ram": ram, "gpu": gpu, "gpu_temp": gpu_temp, 'vram': vram})

@app.route('/api/chats', methods=['GET'])
def list_chats_route():
    return jsonify(get_all_chats(request.args.get('search')))

@app.route('/api/chats/<session_id>', methods=['GET'])
def load_chat_route(session_id):
    c = load_chat_session(session_id)
    return jsonify(c) if c else (jsonify({"error": "Not found"}), 404)

# --- UPLOAD ROUTES ---
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No filename"}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOADS_DIR, filename)
    file.save(path)
    return jsonify({"path": path, "filename": filename})

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    # Placeholder for Whisper integration
    if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
    # Save audio, call Whisper, return text
    return jsonify({"text": "[Voice Input Placeholder]"})

# --- STREAMING CHAT ROUTE ---
@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id')
    rag_enabled = data.get('rag_enabled', False)
    rag_folder = data.get('context_folder')
    
    if not session_id:
        session_id = str(uuid.uuid4())
        history = []
    else:
        existing = load_chat_session(session_id)
        history = existing['messages'] if existing else []

    # Handle RAG Context Injection
    if rag_enabled and rag_folder:
        user_input = f"[SYSTEM: Accessing RAG Context from {rag_folder}]\n{user_input}"

    history.append({"role": "user", "content": user_input})

    def generate():
        # Call the imported function with stream=True
        # This assumes your updated POTATO.main.simple_stream_test supports receiving 'tools' kwarg implicitly
        # or we rely on it being hardcoded there. 
        # For this to work dynamically with tool handling logic as per your docs snippet:
        
        # We start the stream
        stream = simple_stream_test(history, stream=True, think=False)
        
        accumulated_content = ""
        accumulated_thinking = ""
        
        for chunk in stream:
            # 1. Handle Thinking
            if 'thinking' in chunk.get('message', {}): # Assuming model supports it
                think_bit = chunk['message']['thinking']
                accumulated_thinking += think_bit
                yield f"data: {json.dumps({'thinking': think_bit})}\n\n"
            
            # 2. Handle Content
            if 'content' in chunk['message']:
                content_bit = chunk['message']['content']
                accumulated_content += content_bit
                yield f"data: {json.dumps({'content': content_bit})}\n\n"
            
            # 3. Handle Tool Calls (Accumulation logic needed here if streamed)
            if 'tool_calls' in chunk['message']:
                tool_calls = chunk['message']['tool_calls']
                for tool in tool_calls:
                    fn_name = tool.function.name
                    args = tool.function.arguments
                    yield f"data: {json.dumps({'tool_status': f'Calling {fn_name}...'})}\n\n"
                    
                    # Execute Tool
                    if fn_name in TOOL_REGISTRY:
                        try:
                            # Parse args if they are JSON string, otherwise use as dict
                            if isinstance(args, str): tool_args = json.loads(args)
                            else: tool_args = args
                                
                            result = TOOL_REGISTRY[fn_name](**tool_args)
                            
                            # For simple streaming, we just notify UI of result
                            # In a real loop, you'd feed this back to LLM.
                            yield f"data: {json.dumps({'tool_result': str(result)})}\n\n"
                            
                        except Exception as e:
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Save History
        history.append({"role": "assistant", "content": accumulated_content})
        save_chat_session(session_id, history)
        
        # End Stream
        yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)