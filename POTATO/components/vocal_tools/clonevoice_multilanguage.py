"""
Chatterbox TTS (English + Multilingual) - Offline real-time sentence-by-sentence
"""

import os
import re
import torch
import threading
import torchaudio as ta
import sounddevice as sd
from pathlib import Path
import numpy as np

# Force offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)

try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("Package 'chatterbox-tts' not installed. Run: pip install chatterbox-tts")
    raise RuntimeError("chatterbox-tts package not installed")

# Folders & paths
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR   = SCRIPT_DIR / ".temp"
TEMP_DIR.mkdir(exist_ok=True)

SAMPLES_DIR = SCRIPT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

REFERENCE_PATH = SAMPLES_DIR / "Wheatley.wav"
if not REFERENCE_PATH.is_file():
    print(f"Note: Reference audio missing → cloning disabled.")
    REFERENCE_PATH = None

# Tunable params - lowered to reduce repetition / analyzer triggers
TEMPERATURE    = 0.70
EXAGGERATION   = 0.40
CFG_WEIGHT     = 0.5
MAX_LENGTH     = 384   # prevents very long sampling → less analyzer calls

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "zh": "Chinese (Mandarin)", "ja": "Japanese",
    "ko": "Korean", "ru": "Russian", "tr": "Turkish", "pl": "Polish",
    "nl": "Dutch", "sv": "Swedish", "el": "Greek",
}

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Audio device selection (unchanged)
selected_device = None

def select_output_device():
    global selected_device
    print("\n=== Audio output devices ===")
    devices = sd.query_devices()
    output_devices = [i for i, d in enumerate(devices) if d['max_output_channels'] > 0]
    if not output_devices:
        print("No output devices!")
        raise RuntimeError("No audio output devices available")

    for i in output_devices:
        d = devices[i]
        print(f"[{i}] {d['name']} ({d['max_output_channels']} ch, {d['default_samplerate']:.0f} Hz)")

    default_idx = sd.default.device[1] if sd.default.device[1] is not None else output_devices[0]
    print(f"\nDefault: [{default_idx}] {devices[default_idx]['name']}")

    choice = input("Enter number (Enter = default): ").strip()
    try:
        if choice:
            idx = int(choice)
            if idx in output_devices:
                selected_device = idx
            else:
                raise ValueError
        else:
            selected_device = default_idx
    except:
        print("Invalid → using default")
        selected_device = default_idx

    sd.default.device = (None, selected_device)
    print(f"Using: [{selected_device}] {devices[selected_device]['name']}\n")

select_output_device()

# Models loaded on-demand, not at import time
_english_model = None
_multilingual_model = None
_model_lock = threading.Lock()

def _get_english_model():
    """Lazy-load English model"""
    global _english_model
    if _english_model is None:
        with _model_lock:
            if _english_model is None:
                print("Loading Chatterbox English model...")
                _english_model = ChatterboxTTS.from_pretrained(device=device)
                print("English model loaded.")
    return _english_model

def _get_multilingual_model():
    """Lazy-load Multilingual model"""
    global _multilingual_model
    if _multilingual_model is None:
        with _model_lock:
            if _multilingual_model is None:
                print("Loading Chatterbox Multilingual model...")
                _multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                print("Multilingual model loaded.")
    return _multilingual_model

def unload_models():
    """Unload both models from memory"""
    global _english_model, _multilingual_model
    with _model_lock:
        if _english_model is not None:
            del _english_model
            _english_model = None
        if _multilingual_model is not None:
            del _multilingual_model
            _multilingual_model = None
        torch.cuda.empty_cache()
        print("Chatterbox models unloaded from VRAM")

# Sentence splitter (unchanged)
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

    return restored if restored else [text]

# Main speak function
def speak(text: str, language: str = "en"):
    text = text.strip()
    if not text:
        return

    lang = language.lower().strip()
    if lang not in SUPPORTED_LANGUAGES and lang != "en":
        print(f"Warning: Unsupported '{lang}' → fallback to English")
        lang = "en"

    model = _get_english_model() if lang == "en" else _get_multilingual_model()
    lang_id = None if lang == "en" else lang

    sentences = split_sentences(text)

    for sentence in sentences:
        print(f"→ ({lang}) {sentence}")

        try:
            with torch.no_grad(), \
                 torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():

                wav = model.generate(
                    text=sentence,
                    language_id=lang_id,
                    audio_prompt_path=str(REFERENCE_PATH) if REFERENCE_PATH else None,
                    temperature=TEMPERATURE,
                    exaggeration=EXAGGERATION,            # key addition: limit tokens
                    # repetition_penalty=1.2,           # uncomment if Chatterbox accepts it
                )

            # Safe numpy handling (no .abs() on ndarray)
            wav_np = wav.squeeze().cpu().float().numpy()
            max_abs_val = np.max(np.abs(wav_np))   # safe & equivalent
            if max_abs_val > 1.0:
                wav_np /= max_abs_val

            sd.play(wav_np, samplerate=int(model.sr), device=selected_device)
            sd.wait()

            # Save (unchanged)
            idx = len(list(TEMP_DIR.glob("out_*.wav")))
            safe_text = sentence[:35].replace(" ", "_").replace(".", "").replace("!", "").replace("?", "")
            out_path = TEMP_DIR / f"out_{idx:03d}_{lang}_{safe_text}.wav"
            ta.save(out_path, torch.from_numpy(wav_np), model.sr, backend="soundfile")
            print(f"  Saved: {out_path.name}\n")

        except Exception as e:
            print(f"Error in generation/playback: {type(e).__name__}: {str(e)}")

# Interactive mode
if __name__ == "__main__":
    print("Chatterbox TTS – Offline real-time (English + Multilingual)")
    print("  Text → Enter    (or text | lang e.g. Merhaba | tr)")
    print("  Empty / Ctrl+C → quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if "|" in user_input:
                parts = user_input.rsplit("|", 1)
                text_part = parts[0].strip()
                lang_part = parts[1].strip().lower()
            else:
                text_part = user_input
                lang_part = "fr"

            speak(text_part, language=lang_part)

        except KeyboardInterrupt:
            print("\nExited.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}\n")