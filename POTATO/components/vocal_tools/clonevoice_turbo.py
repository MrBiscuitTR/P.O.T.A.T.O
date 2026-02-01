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

# Force CUDA GPU (not iGPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    print("Package 'chatterbox-tts' not installed.")
    print("Run:  pip install chatterbox-tts --no-deps")
    exit(1)

# Folders & paths
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR   = SCRIPT_DIR / ".temp"
print("Using temp dir:", TEMP_DIR)
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

# Model loaded on-demand, not at import time
_model = None
_model_lock = threading.Lock()

def _get_model():
    """Lazy-load the model only when needed"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # Double-check locking
                print("Loading Chatterbox Turbo model...")
                _model = ChatterboxTurboTTS.from_local(ckpt_dir=str(ckpt_path), device=device)
                print("Chatterbox Turbo model loaded.")
    return _model

# Abbreviation-safe sentence splitter
ABBREVIATIONS = [
    r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bDr\.', r'\bProf\.', r'\bSr\.', r'\bJr\.',
    r'\bSt\.', r'\bAve\.', r'\bRd\.', r'\bBlvd\.', r'\bInc\.', r'\bLtd\.', r'\bCo\.',
    r'\bU\.S\.', r'\bU\.K\.', r'\bF\.B\.I\.', r'\bC\.I\.A\.', r'\betc\.', r'\bvs\.'
]

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, group 3 by 3, overflow goes into previous group."""
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

    # Restore abbreviations
    restored = []
    for s in sentences:
        for ph, orig in protected.items():
            s = s.replace(ph, orig)
        s = s.strip()
        if s:
            restored.append(s)

    # If 3 or fewer sentences, return as is
    if len(restored) <= 3:
        return [' '.join(restored)]

    # Group sentences in chunks of 3
    chunk_size = 3
    groups = [restored[i:i+chunk_size] for i in range(0, len(restored), chunk_size)]

    # Merge the last group into the previous if it has fewer than chunk_size sentences
    if len(groups[-1]) < chunk_size and len(groups) > 1:
        groups[-2].extend(groups[-1])
        groups.pop()

    # Join sentences back into strings
    return [' '.join(g) for g in groups]

# =========================
# Streaming TTS Pipeline
# =========================

stop_event = threading.Event()
shutdown_event = threading.Event()
audio_q = queue.Queue()
playback_q = queue.Queue()
producer_thread = None
playback_thread = None
current_stream = None

def generate_worker(text_groups, audio_q):
    """Generate TTS audio chunks and put them in queue"""
    try:
        model = _get_model()  # Get model instance
        for sentence in text_groups:
            if stop_event.is_set() or shutdown_event.is_set():
                break

            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device=="cuda" else torch.no_grad()
            with ctx:
                wav = model.generate(
                    text=sentence,
                    audio_prompt_path=str(AUDIO_PROMPT_PATH) if AUDIO_PROMPT_PATH else None,
                    exaggeration=0.8, 
                    cfg_weight=0.5
                )

            wav = wav.squeeze()
            if wav.abs().max() > 1:
                wav = wav / wav.abs().max()

            audio_q.put((sentence, wav.cpu().float().numpy()))
    finally:
        audio_q.put(None)

def playback_worker(audio_q, sample_rate):
    """Continuously play audio chunks from queue without gaps"""
    global current_stream
    try:
        while not shutdown_event.is_set():
            if stop_event.is_set():
                # Clear queue when stopped
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except:
                        break
                stop_event.clear()
                continue
            
            item = audio_q.get()
            if item is None:
                break
            
            sentence, wav_np = item
            print(f"→ {sentence}")
            
            if stop_event.is_set():
                continue
            
            # Play audio and wait for completion
            sd.play(wav_np, samplerate=sample_rate, blocking=True)
            
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        current_stream = None

def speak_sentences_grouped(text: str):
    """Stream TTS in chunks with seamless playback queue. Safe for very long text."""
    global producer_thread, playback_thread
    
    # Don't clear stop_event here - let it be managed by stop_current_tts()
    # Clear audio queue
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except:
            break

    groups = split_sentences(text)
    model = _get_model()  # Get model instance for sample rate

    # Start producer thread
    producer_thread = threading.Thread(
        target=generate_worker,
        args=(groups, audio_q),
        daemon=True
    )
    producer_thread.start()

    # Start playback thread if not running
    if playback_thread is None or not playback_thread.is_alive():
        playback_thread = threading.Thread(
            target=playback_worker,
            args=(audio_q, int(model.sr)),
            daemon=True
        )
        playback_thread.start()
    
    # Wait for producer to finish
    producer_thread.join()
# =========================
# Control functions
# =========================

def stop_current_tts():
    """Stop the current generation & playback, keep model loaded. Does NOT terminate app."""
    stop_event.set()
    sd.stop()
    
    # Clear audio queue
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except:
            break
    
    # Don't join threads - let them handle stop_event gracefully
    print("→ Current TTS stopped, model still loaded.")
    
    # Reset stop event after a brief moment
    time.sleep(0.1)
    stop_event.clear()

def shutdown_tts():
    """Stop everything and unload the model to free VRAM. Safe, won't crash app."""
    global _model, producer_thread, playback_thread
    
    # Signal stop
    stop_event.set()
    shutdown_event.set()
    sd.stop()
    
    # Clear queues
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except:
            break
    
    # Put None to signal threads to exit
    try:
        audio_q.put(None, timeout=0.1)
    except:
        pass
    
    # Wait for threads
    if producer_thread is not None and producer_thread.is_alive():
        producer_thread.join(timeout=1.0)
    if playback_thread is not None and playback_thread.is_alive():
        playback_thread.join(timeout=1.0)
    
    # Unload model
    with _model_lock:
        if _model is not None:
            del _model
            _model = None
            torch.cuda.empty_cache()
    
    # Reset events
    stop_event.clear()
    shutdown_event.clear()
    
    producer_thread = None
    playback_thread = None
    
    print("→ TTS shutdown complete, model unloaded, VRAM freed.")

def unload_model():
    """Unload the model from memory"""
    global _model
    with _model_lock:
        if _model is not None:
            del _model
            _model = None
            torch.cuda.empty_cache()
            print("Chatterbox Turbo model unloaded from VRAM")

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
        shutdown_event.set()
        sd.stop()
        print("\nExited.\n")
        sys.exit(0)
