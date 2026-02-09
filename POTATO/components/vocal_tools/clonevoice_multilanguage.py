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

REFERENCE_PATH = SAMPLES_DIR / "dilara-tr.wav"
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

# Audio device selection (lazy - only when needed)
selected_device = None
_device_initialized = False

def select_output_device():
    """Select output device - only call in interactive mode (__main__)"""
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

def _ensure_device_initialized():
    """Ensure device is set to default when used as library (non-interactive)"""
    global selected_device, _device_initialized
    if not _device_initialized:
        # Use system default device when imported as library
        default_output = sd.default.device[1]
        if default_output is not None:
            selected_device = default_output
        else:
            # Fallback: find first output device
            devices = sd.query_devices()
            output_devices = [i for i, d in enumerate(devices) if d['max_output_channels'] > 0]
            selected_device = output_devices[0] if output_devices else None
        _device_initialized = True
        if selected_device is not None:
            print(f"[TTS] Using default audio device: {sd.query_devices(selected_device)['name']}")

# DO NOT call select_output_device() at import time - only in __main__ block

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

def preload_model(language: str = "en"):
    """Preload TTS model to VRAM. Called when speech detected."""
    if language.lower() == "en":
        _get_english_model()
    else:
        _get_multilingual_model()
    print(f"[TTS] {language} model preloaded to VRAM")

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
    # Ensure device is initialized (uses default when imported as library)
    _ensure_device_initialized()

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

def generate_tts_wav_multilingual(text: str, language: str = "en") -> bytes:
    """Generate TTS audio and return as WAV bytes for browser playback."""
    import io
    from scipy.io import wavfile

    text = text.strip()
    if not text:
        return b""

    lang = language.lower().strip()
    if lang not in SUPPORTED_LANGUAGES and lang != "en":
        lang = "en"

    model = _get_english_model() if lang == "en" else _get_multilingual_model()
    lang_id = None if lang == "en" else lang
    sample_rate = int(model.sr)
    sentences = split_sentences(text)

    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            with torch.no_grad(), \
                 torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                wav = model.generate(
                    text=sentence,
                    language_id=lang_id,
                    audio_prompt_path=str(REFERENCE_PATH) if REFERENCE_PATH else None,
                    temperature=TEMPERATURE,
                    exaggeration=EXAGGERATION,
                )
            wav_np = wav.squeeze().cpu().float().numpy()
            max_abs_val = np.max(np.abs(wav_np))
            if max_abs_val > 1.0:
                wav_np /= max_abs_val
            chunks.append(wav_np)
        except Exception as e:
            print(f"[TTS] Error generating sentence: {e}")
            continue

    if not chunks:
        return b""

    audio = np.concatenate(chunks)
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()


# Stop event imported from turbo module so both share the same flag.
# This lets the frontend's stop button cancel multilingual generation too.
def _get_stop_event():
    """Import the stop_event from clonevoice_turbo so we share one flag."""
    try:
        from POTATO.components.vocal_tools.clonevoice_turbo import stop_event
        return stop_event
    except ImportError:
        return None


def generate_tts_wav_streamed_multilingual(text: str, language: str = "en"):
    """Yield individual WAV-chunk bytes for each sentence group (multilingual).

    Mirrors the streaming interface of clonevoice_turbo.generate_tts_wav_streamed()
    so the SSE endpoint can treat both identically.

    Yields:
        tuple(int, int, bytes):  (chunk_index, total_chunks, wav_bytes)
    """
    import io
    from scipy.io import wavfile

    text = text.strip()
    if not text:
        return

    lang = language.lower().strip()
    if lang not in SUPPORTED_LANGUAGES and lang != "en":
        lang = "en"

    model = _get_english_model() if lang == "en" else _get_multilingual_model()
    lang_id = None if lang == "en" else lang
    sample_rate = int(model.sr)
    sentences = split_sentences(text)

    # Group sentences into chunks of 2-3 for balanced sizes (same strategy as turbo)
    total = len(sentences)
    if total <= 3:
        groups = [' '.join(sentences)] if sentences else []
    else:
        # Calculate balanced distribution to avoid one huge chunk
        remainder = total % 3

        if remainder == 0:
            # Perfect division by 3: all chunks get 3 sentences
            chunk_size = 3
            raw_groups = [sentences[i:i+chunk_size] for i in range(0, total, chunk_size)]
        elif remainder == 1:
            # Total = 3n+1: split as 2,3,3,3,... (first chunk has 2, rest have 3)
            raw_groups = [sentences[0:2]]
            for i in range(2, total, 3):
                raw_groups.append(sentences[i:i+3])
        else:  # remainder == 2
            # Total = 3n+2: split as 3,3,3,...,2 (all have 3 except last has 2)
            chunk_size = 3
            raw_groups = [sentences[i:i+chunk_size] for i in range(0, total, chunk_size)]

        groups = [' '.join(g) for g in raw_groups]

    total = len(groups)
    if total == 0:
        return

    stop_evt = _get_stop_event()

    for idx, sentence in enumerate(groups):
        # Check cancellation between chunks
        if stop_evt and stop_evt.is_set():
            print(f"[TTS-ML-STREAM] Cancelled before chunk {idx}/{total}")
            return

        if not sentence.strip():
            continue

        try:
            with torch.no_grad(), \
                 torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                wav = model.generate(
                    text=sentence,
                    language_id=lang_id,
                    audio_prompt_path=str(REFERENCE_PATH) if REFERENCE_PATH else None,
                    temperature=TEMPERATURE,
                    exaggeration=EXAGGERATION,
                )

            wav_np = wav.squeeze().cpu().float().numpy()
            max_abs_val = np.max(np.abs(wav_np))
            if max_abs_val > 1.0:
                wav_np /= max_abs_val

            audio_int16 = (wav_np * 32767).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, audio_int16)
            wav_bytes = buf.getvalue()

            print(f"[TTS-ML-STREAM] Chunk {idx+1}/{total} ready "
                  f"({len(sentence)} chars, {len(wav_bytes)} bytes)")

            yield (idx, total, wav_bytes)

        except Exception as e:
            print(f"[TTS-ML-STREAM] Error generating chunk {idx}: {e}")
            continue


# Interactive mode
if __name__ == "__main__":
    print("Chatterbox TTS – Offline real-time (English + Multilingual)")
    print("  Text → Enter    (or text | lang e.g. Merhaba | tr)")
    print("  Empty / Ctrl+C → quit\n")

    # Only prompt for device selection in interactive mode
    select_output_device()

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