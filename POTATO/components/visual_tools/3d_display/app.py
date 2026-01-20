from flask import Flask, render_template, send_from_directory, jsonify
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models/list")
def list_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    files = [f for f in os.listdir(models_dir) if f.lower().endswith('.obj')]
    return jsonify(files)

@app.route("/models/<path:filename>")
def models(filename):
    return send_from_directory("models", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3754, debug=True)
