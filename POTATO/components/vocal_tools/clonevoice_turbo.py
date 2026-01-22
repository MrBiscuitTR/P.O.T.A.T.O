"""Supports: [clear throat][sigh][shush][cough][groan][sniff][gasp][chuckle][laugh]"""

"""
Chatterbox-Turbo - Fully offline, real-time sentence-by-sentence TTS
- Loads via .from_local() → no Hugging Face token/network ever
- Speaks input text sentence by sentence (preserves abbreviations)
- Voice cloning from sample1cagan.wav if present
- GPU-accelerated with bfloat16 + autocast
"""
import sys
import os
import re
import time
import torch
import queue
import threading
import torchaudio as ta
import sounddevice as sd
from pathlib import Path

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    print("Package 'chatterbox-tts' not installed.")
    print("Run:  pip install chatterbox-tts --no-deps")
    exit(1)

# Folders & paths
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR   = SCRIPT_DIR / ".temp"
TEMP_DIR.mkdir(exist_ok=True)

SAMPLES_DIR = SCRIPT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

AUDIO_PROMPT_PATH = SAMPLES_DIR / "Wheatley.wav"

if not AUDIO_PROMPT_PATH.is_file():
    print(f"Note: No reference audio at {AUDIO_PROMPT_PATH}")
    print("→ Voice cloning disabled (default voice).")
    AUDIO_PROMPT_PATH = None

# Local model path
CKPT_DIR = r"C:\Users\cagan\.cache\huggingface\hub\models--ResembleAI--chatterbox-turbo\snapshots\749d1c1a46eb10492095d68fbcf55691ccf137cd"
ckpt_path = Path(CKPT_DIR)

if not ckpt_path.is_dir() or not (ckpt_path / "t3_turbo_v1.safetensors").exists():
    print(f"ERROR: Invalid or missing checkpoint: {CKPT_DIR}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = ChatterboxTurboTTS.from_local(ckpt_dir=str(ckpt_path), device=device)
print("Model loaded.\n")

# Abbreviation-safe sentence splitter
ABBREVIATIONS = [
    r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bDr\.', r'\bProf\.', r'\bSr\.', r'\bJr\.',
    r'\bSt\.', r'\bAve\.', r'\bRd\.', r'\bBlvd\.', r'\bInc\.', r'\bLtd\.', r'\bCo\.',
    r'\bU\.S\.', r'\bU\.K\.', r'\bF\.B\.I\.', r'\bC\.I\.A\.', r'\betc\.', r'\bvs\.'
]

def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    protected = {}
    for i, pat in enumerate(ABBREVIATIONS):
        for m in re.finditer(pat, text, re.IGNORECASE):
            ph = f"__ABBR_{i}_{len(protected)}__"
            text = text[:m.start()] + ph + text[m.end():]
            protected[ph] = m.group(0)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    restored = []
    for s in sentences:
        for ph, orig in protected.items():
            s = s.replace(ph, orig)
        s = s.strip()
        if s:
            restored.append(s)

    if len(restored) <= 4:
        return restored

    groups = []
    for i in range(0, len(restored), 4):
        groups.append(' '.join(restored[i:i+4]))

    if len(restored) % 4 != 0:
        tail = groups.pop()
        groups[-1] += ' ' + tail

    return groups

# =========================
# Streaming TTS Pipeline
# =========================

stop_event = threading.Event()

def generate_worker(text_groups, audio_q):
    try:
        for sentence in text_groups:
            if stop_event.is_set():
                break

            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device=="cuda" else torch.no_grad()
            with ctx:
                wav = model.generate(
                    text=sentence,
                    audio_prompt_path=str(AUDIO_PROMPT_PATH) if AUDIO_PROMPT_PATH else None,
                    # exaggeration=0.8,
                    length_scale= 1.2,
                    noise_scale=0.7
                )

            wav = wav.squeeze()
            if wav.abs().max() > 1:
                wav = wav / wav.abs().max()

            audio_q.put((sentence, wav.cpu().float().numpy()))

    finally:
        audio_q.put(None)

def speak_sentences_grouped(text: str):
    groups = split_sentences(text)
    audio_q = queue.Queue(maxsize=2)

    producer = threading.Thread(
        target=generate_worker,
        args=(groups, audio_q),
        daemon=True
    )
    producer.start()

    try:
        while True:
            item = audio_q.get()
            if item is None or stop_event.is_set():
                break

            sentence, wav_np = item
            print(f"→ {sentence}")

            sd.play(wav_np, samplerate=int(model.sr))
            sd.wait()

            time.sleep(0.2)

    except KeyboardInterrupt:
        stop_event.set()
        sd.stop()
        print("\nCtrl+C → exiting.\n")
        sys.exit(0)

# =========================
# Main loop
# =========================

if __name__ == "__main__":
    print("Chatterbox-Turbo – Offline TTS")
    print("  Enter text → Enter")
    print("  Ctrl+C → exit immediately\n")

    try:
        while True:
            stop_event.clear()
            text = input("You: ").strip()
            if not text:
                continue

            speak_sentences_grouped(text)

    except KeyboardInterrupt:
        stop_event.set()
        sd.stop()
        print("\nExited.\n")
        sys.exit(0)