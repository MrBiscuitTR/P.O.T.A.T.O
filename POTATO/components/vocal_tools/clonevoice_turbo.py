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
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)

# Force CUDA GPU (not iGPU) and maximize performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA for speed
torch.cuda.set_device(0)

# Set to high performance mode (no power saving)
if torch.cuda.is_available():
    print("[TTS - Chatterbox] CUDA is available. High performance mode enabled.")
    # torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
    # torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere+
    # torch.backends.cudnn.allow_tf32 = True

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    print("Package 'chatterbox-tts' not installed.")
    print("Run:  pip install chatterbox-tts --no-deps")
    raise RuntimeError("chatterbox-tts package not installed")

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
    raise RuntimeError(f"Missing TTS checkpoint: {CKPT_DIR}")

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
                print(f"Loading Chatterbox Turbo model to {device}...")

                # Load model (temporarily uses CPU RAM before moving to CUDA)
                _model = ChatterboxTurboTTS.from_local(
                    ckpt_dir=str(ckpt_path),
                    device=device
                )

                # NOTE: Chatterbox Turbo model structure (confirmed via introspection):
                # - Does NOT have: sampler, scheduler, noise_scheduler attributes
                # - Does NOT support: num_steps, cfg_weight, exaggeration parameters
                # - Components: s3gen (S3Token2Wav), t3 (T3), ve (VoiceEncoder)
                # - s3gen only has 'resamplers' attribute, no step counters
                # - Progress bar x/1000 is internal to model.generate() - cannot be configured
                # - Model is NOT a PyTorch nn.Module, so no .parameters() or .eval() methods

                print(f"[TTS - Chatterbox] Chatterbox Turbo model loaded to {device}")

                # NOTE: Chatterbox Turbo is NOT a standard PyTorch nn.Module
                # - No .parameters() method (not trainable via standard PyTorch)
                # - No .eval() method (doesn't have train/eval modes)
                # - Cannot use torch.compile() on it
                # - Model handles its own CUDA device placement

                if device == "cuda":
                    # Free CPU RAM after model moved to CUDA
                    import gc
                    gc.collect()

                    try:
                        vram_used = torch.cuda.memory_allocated(0) / 1e9
                        print(f"[TTS - Chatterbox] Using {vram_used:.2f}GB VRAM")
                    except:
                        pass

                print("[TTS - Chatterbox] Ready for inference (Turbo mode: fast 1-step generation)")

    return _model

def preload_model():
    """Preload TTS model to VRAM without generating audio. Called when speech detected."""
    _get_model()  # This loads the model if not already loaded
    print("[TTS - Chatterbox] Model preloaded to VRAM")

# Abbreviation-safe sentence splitter
ABBREVIATIONS = [
    r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bDr\.', r'\bProf\.', r'\bSr\.', r'\bJr\.',
    r'\bSt\.', r'\bAve\.', r'\bRd\.', r'\bBlvd\.', r'\bInc\.', r'\bLtd\.', r'\bCo\.',
    r'\bU\.S\.', r'\bU\.K\.', r'\bF\.B\.I\.', r'\bC\.I\.A\.', r'\betc\.', r'\bvs\.'
]

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, group 2-3 sentences per chunk for balanced sizes.

    Uses dynamic chunking to avoid one huge chunk:
    - Calculates optimal chunk size based on total sentence count
    - Distributes remainder sentences across all chunks evenly
    - Ensures all chunks are 2-3 sentences (never 1, never >3)
    """
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

    total = len(restored)

    # If 3 or fewer sentences, return as single chunk
    if total <= 3:
        return [' '.join(restored)]

    # Calculate how many chunks we need for balanced distribution
    # Goal: all chunks 2-3 sentences, prefer 3 when possible
    # If total % 3 == 0: all chunks get 3 sentences
    # If total % 3 == 1: make 4 chunks of size 3,3,3,...,3,4 -> BAD!
    #                    Better: split into chunks of size 2,3,3,3... or 3,3,3,...,2,2
    # If total % 3 == 2: make chunks of size 3,3,3,...,3,2 -> last chunk has 2

    # Strategy: Use chunks of size 3, but if remainder is 1, use one chunk of size 2
    # to avoid a tiny 1-sentence chunk or a huge merged chunk

    remainder = total % 3

    if remainder == 0:
        # Perfect division by 3
        chunk_size = 3
        groups = [restored[i:i+chunk_size] for i in range(0, total, chunk_size)]
    elif remainder == 1:
        # Total = 3n+1 (e.g., 4, 7, 10, 13, ...)
        # Split as: 2, 3, 3, 3, ... (first chunk has 2, rest have 3)
        groups = [restored[0:2]]  # First chunk: 2 sentences
        for i in range(2, total, 3):
            groups.append(restored[i:i+3])
    else:  # remainder == 2
        # Total = 3n+2 (e.g., 5, 8, 11, 14, ...)
        # Split as: 3, 3, 3, ..., 2 (all have 3 except last has 2)
        chunk_size = 3
        groups = [restored[i:i+chunk_size] for i in range(0, total, chunk_size)]
        # Last group will naturally have 2 sentences

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

# Global flag to track if TTS is currently speaking (for VOX interrupt detection)
_is_speaking = False

# Cache audio prompt path string to avoid repeated conversions
_audio_prompt_str = str(AUDIO_PROMPT_PATH) if AUDIO_PROMPT_PATH else None

def _get_cached_audio_prompt_path() -> str | None:
    """Return the audio prompt path string, caching the file read on first call.

    Chatterbox Turbo accepts audio_prompt_path as a string and loads it internally.
    We cache the path string here.  The model itself caches the loaded waveform
    after the first generate() call, so subsequent calls with the same path are fast.
    """
    return _audio_prompt_str

def generate_worker(text_groups, audio_q):
    """Generate TTS audio chunks and put them in queue"""
    try:
        model = _get_model()  # Get model instance
        for sentence in text_groups:
            # Check stop flags before processing each sentence
            if stop_event.is_set() or shutdown_event.is_set():
                print("[TTS - Chatterbox] Generation stopped by flag")
                break

            try:
                # NOTE: Only text and audio_prompt_path are supported
                # No num_steps, cfg_weight, exaggeration, etc.
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device=="cuda" else torch.no_grad()
                with ctx:
                    wav = model.generate(
                        text=sentence,
                        audio_prompt_path=_audio_prompt_str
                    )

                wav = wav.squeeze()
                if wav.abs().max() > 1:
                    wav = wav / wav.abs().max()

                # Use timeout to prevent blocking if queue is being cleared
                try:
                    audio_q.put((sentence, wav.cpu().float().numpy()), timeout=0.5)
                except queue.Full:
                    print("[TTS - Chatterbox] Queue full, skipping chunk")
                    break
            except Exception as e:
                print(f"[TTS - Chatterbox] Error generating sentence: {e}")
                # Continue to next sentence instead of crashing
                continue
                
    except Exception as e:
        print(f"[TTS - Chatterbox] Fatal error in generate_worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal end of generation
        try:
            audio_q.put(None, timeout=0.5)
        except:
            pass

def playback_worker(audio_q, sample_rate):
    """Continuously play audio chunks from queue without gaps"""
    global current_stream
    try:
        while True:
            # Check shutdown first - exit immediately if set
            if shutdown_event.is_set():
                print("[TTS - Chatterbox] Playback worker shutting down...")
                break

            if stop_event.is_set():
                # Stop current audio
                try:
                    sd.stop()
                except:
                    pass

                # Clear queue completely
                cleared_count = 0
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                        cleared_count += 1
                    except queue.Empty:
                        break

                if cleared_count > 0:
                    print(f"[TTS - Chatterbox] Cleared {cleared_count} queued chunks")

                # DON'T clear stop_event here - let stop_current_tts() handle it
                # Wait a moment before checking again
                time.sleep(0.1)
                continue

            # Get next audio with shorter timeout to respond faster to shutdown
            try:
                item = audio_q.get(timeout=0.1)
            except queue.Empty:
                # No audio available, check flags and continue
                continue

            # None signals end of generation
            if item is None:
                print("[TTS - Chatterbox] Received end signal (None)")
                break

            # Check stop again before playing
            if stop_event.is_set():
                continue

            sentence, wav_np = item
            print(f"→ {sentence}")

            try:
                # Play audio and wait for completion
                sd.play(wav_np, samplerate=sample_rate, blocking=True)
            except Exception as e:
                print(f"[TTS - Chatterbox] Playback error: {e}")
                # Continue to next chunk instead of crashing

    except Exception as e:
        print(f"[TTS - Chatterbox] Playback error: {e}")
    finally:
        current_stream = None
        print("[TTS - Chatterbox] Playback worker exited")

def speak_sentences_grouped(text: str):
    """Stream TTS in chunks with seamless playback queue. Safe for very long text."""
    global producer_thread, playback_thread, _is_speaking
    
    try:
        # Set speaking flag
        _is_speaking = True
        
        # Clear stop event if it's set from a previous stop
        if stop_event.is_set():
            stop_event.clear()
        
        # Clear audio queue before starting new generation
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

        groups = split_sentences(text)
        model = _get_model()  # Get model instance for sample rate

        # Start producer thread for this generation
        producer_thread = threading.Thread(
            target=generate_worker,
            args=(groups, audio_q),
            daemon=True
        )
        producer_thread.start()

        # Start playback thread if not running (single persistent thread)
        if playback_thread is None or not playback_thread.is_alive():
            playback_thread = threading.Thread(
                target=playback_worker,
                args=(audio_q, int(model.sr)),
                daemon=True
            )
            playback_thread.start()
        
        # Wait for producer to finish generating audio
        producer_thread.join(timeout=300)  # 5 minute timeout
        
    except Exception as e:
        print(f"[TTS - Chatterbox] Error in speak_sentences_grouped: {e}")
        import traceback
        traceback.print_exc()
        # Don't crash - let the error be handled gracefully
    finally:
        # Clear speaking flag when done (even if error)
        _is_speaking = False

def generate_tts_wav(text: str) -> bytes:
    """Generate TTS audio and return as WAV bytes (for browser playback). Respects stop flags."""
    import io
    import numpy as np
    from scipy.io import wavfile

    model = _get_model()
    sample_rate = int(model.sr)
    groups = split_sentences(text)

    # NOTE: Progress bar showing x/1000 during generation is internal to the model
    # The x value can vary (e.g., 50/1000, 659/1000) based on text length
    # This is normal behavior and doesn't indicate progressive slowdown

    # Clear CUDA cache before generation
    # if device == "cuda":
    #     torch.cuda.empty_cache()

    chunks = []
    with torch.inference_mode():  # More efficient than autocast for inference
        for sentence in groups:
            # Check stop flags before each sentence
            if stop_event.is_set() or shutdown_event.is_set():
                print("[TTS - Chatterbox] Generation cancelled by stop flag")
                break

            if not sentence.strip():
                continue

            try:
                # NOTE: Chatterbox Turbo only supports these parameters:
                # - text (required): The text to synthesize
                # - audio_prompt_path (optional): Voice reference for cloning
                # Does NOT support: num_steps, cfg_weight, exaggeration, min_p, etc.

                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        wav = model.generate(
                            text=sentence,
                            audio_prompt_path=_audio_prompt_str
                        )
                else:
                    wav = model.generate(
                        text=sentence,
                        audio_prompt_path=_audio_prompt_str
                    )

                wav = wav.squeeze()
                if wav.abs().max() > 1:
                    wav = wav / wav.abs().max()

                # Single CPU transfer at the end (much faster than per-sentence)
                chunks.append(wav)

            except Exception as e:
                print(f"[TTS - Chatterbox] Error generating sentence: {e}")
                continue

    if not chunks:
        return b""

    # Concatenate on GPU first, then single transfer to CPU (MUCH faster)
    audio_gpu = torch.cat(chunks, dim=0)
    audio = audio_gpu.cpu().float().numpy()

    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)

    # NOTE: s3gen (S3Token2Wav) component doesn't have resettable state
    # The model handles its own internal state management
    # Clear CUDA cache after generation to free up temp memory
    # if device == "cuda":
    #     torch.cuda.empty_cache()

    return buf.getvalue()


def generate_tts_wav_streamed(text: str):
    """Yield individual WAV-chunk bytes as each sentence group is generated.

    This is the core of the streaming TTS pipeline:
    - Splits text into 2-3 sentence groups (abbreviation-safe)
    - Generates TTS for each group one at a time
    - Yields (chunk_index, total_chunks, wav_bytes) after each group finishes
    - Checks stop_event between every group so the caller can cancel instantly
    - The browser can start playing chunk 0 while chunk 1 is being generated

    Yields:
        tuple(int, int, bytes):  (chunk_index, total_chunks, wav_bytes_for_this_chunk)
    """
    import io
    import numpy as np
    from scipy.io import wavfile

    global _is_speaking
    _is_speaking = True

    try:
        model = _get_model()
        sample_rate = int(model.sr)
        groups = split_sentences(text)
        total = len(groups)

        if total == 0:
            return

        # Clear stop flag from any previous run so we start clean
        if stop_event.is_set():
            stop_event.clear()

        # Clear CUDA cache before starting a batch of generations
        # if device == "cuda":
        #     torch.cuda.empty_cache()

        prompt_path = _get_cached_audio_prompt_path()

        for idx, sentence in enumerate(groups):
            # --- Check cancellation before generating each chunk ---
            if stop_event.is_set() or shutdown_event.is_set():
                print(f"[TTS-STREAM] Generation cancelled before chunk {idx}/{total}")
                return

            if not sentence.strip():
                continue

            try:
                # Generate this chunk's audio
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        wav = model.generate(
                            text=sentence,
                            audio_prompt_path=prompt_path
                        )
                else:
                    wav = model.generate(
                        text=sentence,
                        audio_prompt_path=prompt_path
                    )

                wav = wav.squeeze()
                if wav.abs().max() > 1:
                    wav = wav / wav.abs().max()

                # Transfer to CPU and convert to int16 WAV bytes
                audio_np = wav.cpu().float().numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)

                buf = io.BytesIO()
                wavfile.write(buf, sample_rate, audio_int16)
                wav_bytes = buf.getvalue()

                print(f"[TTS-STREAM] Chunk {idx+1}/{total} ready "
                      f"({len(sentence)} chars, {len(wav_bytes)} bytes)")

                yield (idx, total, wav_bytes)

            except Exception as e:
                print(f"[TTS-STREAM] Error generating chunk {idx}: {e}")
                # Skip this chunk, continue with next
                continue

        # Free CUDA temp memory after all chunks done
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[TTS-STREAM] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _is_speaking = False


# =========================
# Control functions
# =========================

def stop_current_tts():
    """Stop the current generation & playback immediately, keep model loaded. Crash-proof."""
    global producer_thread, current_stream, _is_speaking
    
    try:
        # Clear speaking flag
        _is_speaking = False
        
        # Set stop flag FIRST
        stop_event.set()
        
        # Stop audio playback immediately
        try:
            sd.stop()
        except Exception as e:
            print(f"[TTS - Chatterbox] Error stopping audio: {e}")
        
        # Clear audio queue completely
        cleared_count = 0
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        if cleared_count > 0:
            print(f"→ Stopped TTS and cleared {cleared_count} queued chunks")
        else:
            print("→ TTS stopped")
        
        # Wait briefly for threads to notice stop flag
        time.sleep(0.2)
        
        # Clear stop event so playback can resume on next speak call
        stop_event.clear()
        
    except Exception as e:
        print(f"[TTS - Chatterbox] Error in stop_current_tts (non-fatal): {e}")
        import traceback
        traceback.print_exc()
        # Clear the flag anyway
        stop_event.clear()

def is_speaking():
    """Check if TTS is currently speaking (for VOX interrupt detection)"""
    return _is_speaking

def shutdown_tts():
    """Stop everything and unload the model to free VRAM. Safe, won't crash app."""
    global _model, producer_thread, playback_thread

    print("[TTS - Chatterbox] Starting shutdown sequence...")

    # Signal stop
    stop_event.set()
    shutdown_event.set()

    # Stop audio immediately
    try:
        sd.stop()
    except:
        pass

    # Clear queues
    cleared = 0
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
            cleared += 1
        except:
            break
    if cleared > 0:
        print(f"[TTS - Chatterbox] Cleared {cleared} items from queue")

    # Put None to signal threads to exit
    try:
        audio_q.put(None, timeout=0.1)
    except:
        pass

    # Wait for threads with logging
    if producer_thread is not None and producer_thread.is_alive():
        print("[TTS - Chatterbox] Waiting for producer thread...")
        producer_thread.join(timeout=0.5)
        if producer_thread.is_alive():
            print("[TTS - Chatterbox] Producer thread still alive (daemon will exit with app)")

    if playback_thread is not None and playback_thread.is_alive():
        print("[TTS - Chatterbox] Waiting for playback thread...")
        playback_thread.join(timeout=0.5)
        if playback_thread.is_alive():
            print("[TTS - Chatterbox] Playback thread still alive (daemon will exit with app)")

    # Unload model
    with _model_lock:
        if _model is not None:
            print("[TTS - Chatterbox] Unloading model from VRAM...")
            del _model
            _model = None
            torch.cuda.empty_cache()
            print("[TTS - Chatterbox] Model unloaded, VRAM freed")

    # Reset events
    stop_event.clear()
    shutdown_event.clear()

    producer_thread = None
    playback_thread = None

    print("→ TTS shutdown complete.")

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
        pass  # Exit main loop gracefully
