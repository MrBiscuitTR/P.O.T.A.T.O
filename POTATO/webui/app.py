# POTATO/webui/app.py
def run_flask_app():
    from flask import Flask, render_template, request, jsonify
    import threading
    import sys
    import os
    import json
    from glob import glob
    import time
    import POTATO.main as main

    app = Flask(__name__, template_folder='templates', static_folder='static')

    CHAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.data', 'chats'))
    os.makedirs(CHAT_DIR, exist_ok=True)

    def load_chat(chat_id):
        path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_chat(chat_id, history):
        path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def list_chats():
        chats = []
        for file in glob(os.path.join(CHAT_DIR, "*.json")):
            chat_id = os.path.splitext(os.path.basename(file))[0]
            with open(file, "r", encoding="utf-8") as f:
                history = json.load(f)
                title = history[0]["content"] if history else chat_id
                chats.append({"id": chat_id, "title": title})
        return chats

    chat_history = []
    current_chat_id = "default"

    @app.route('/')
    def index():
        system_status = main.initial_system_status
        chats = list_chats()
        return render_template(
            'index.html',
            system_status=system_status,
            chat_history=chat_history,
            chats=chats,
            current_chat_id=current_chat_id
        )

    @app.route('/api/system_info')
    def api_system_info():
        info = main.json_get_instant_system_info()
        ram = info.get("ram", {})
        gpus = info.get("gpus", [])
        cpu = info.get("cpu", {})
        filtered = {
            "ram": {
                "available_gb": ram.get("available_gb"),
                "total_gb": ram.get("total_gb"),
                "usage_percent": ram.get("usage_percent")
            },
            "gpus": [
                {
                    "id_name": f'{gpu.get("id")}: {gpu.get("name")}',
                    "load_percent": gpu.get("load_percent"),
                    "memoryFree_MB": gpu.get("memoryFree_MB"),
                    "memoryTotal_MB": gpu.get("memoryTotal_MB"),
                    "temperature_C": gpu.get("temperature_C")
                } for gpu in gpus
            ],
            "cpu": {
                "usage_percent": cpu.get("usage_percent")
            }
        }
        return jsonify(filtered)

    @app.route('/api/model_info')
    def api_model_info():
        return jsonify({
            "name": "Ollama LLM",
            "version": "1.0",
            "status": "Ready",
            "context_length": 4096,
            "other": {}
        })

    @app.route('/api/chats', methods=['GET'])
    def api_list_chats():
        return jsonify(list_chats())

    @app.route('/api/chats/<chat_id>', methods=['GET'])
    def api_get_chat(chat_id):
        return jsonify(load_chat(chat_id))

    @app.route('/api/chats/new', methods=['POST'])
    def api_new_chat():
        nonlocal chat_history, current_chat_id
        chat_id = str(int(time.time()))
        chat_history = []
        current_chat_id = chat_id
        save_chat(chat_id, chat_history)
        return jsonify({"id": chat_id})

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        nonlocal chat_history, current_chat_id
        data = request.json
        user_message = data.get('message', '')
        chat_id = data.get('chat_id', current_chat_id)
        chat_history = load_chat(chat_id)
        ai_response = f"Echo: {user_message}"
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "ai", "content": ai_response})
        save_chat(chat_id, chat_history)
        current_chat_id = chat_id
        return jsonify({"response": ai_response, "tokens": len(user_message.split()) + 2, "chat_history": chat_history})

    app.run(host='localhost', port=3169 , debug=False, use_reloader=True)

if __name__ == "__main__":
    run_flask_app()