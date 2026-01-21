"""Supports: 

[clear throat][sigh][shush][cough][groan][sniff][gasp][chuckle][laugh]"""

"""
Chatterbox-Turbo - Fully offline, real-time sentence-by-sentence TTS
- Loads via .from_local() → no Hugging Face token/network ever
- Speaks input text sentence by sentence (preserves F.B.I., Mr. etc.)
- Uses your sample1cagan.m4a for voice cloning if present
- GPU-accelerated with bfloat16 + autocast (RTX 5090 friendly)
- Interactive terminal loop + audio device selection
"""

import os
import re
import torch
import torchaudio as ta
import sounddevice as sd
from pathlib import Path

# Force offline mode right at the start
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
except ImportError:
    print("Package 'chatterbox-tts' not installed.")
    print("Run:  pip install chatterbox-tts --no-deps")
    exit(1)

# ────────────────────────────────────────────────
# Folders & paths
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR   = SCRIPT_DIR / ".temp"
TEMP_DIR.mkdir(exist_ok=True)

SAMPLES_DIR = SCRIPT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

AUDIO_PROMPT_PATH = SAMPLES_DIR / "sample1cagan.wav"

if not AUDIO_PROMPT_PATH.is_file():
    print(f"Note: No reference audio at {AUDIO_PROMPT_PATH}")
    print("→ Voice cloning disabled (using default voice).")
    print("→ Place a 5–15s clean audio file there for cloning.\n")
    AUDIO_PROMPT_PATH = None

# ────────────────────────────────────────────────
# LOCAL SNAPSHOT PATH
# ────────────────────────────────────────────────
CKPT_DIR = r"C:\Users\cagan\.cache\huggingface\hub\models--ResembleAI--chatterbox-turbo\snapshots\749d1c1a46eb10492095d68fbcf55691ccf137cd"

ckpt_path = Path(CKPT_DIR)
if not ckpt_path.is_dir():
    print(f"ERROR: Snapshot directory not found: {CKPT_DIR}")
    print("→ Download once (needs internet):")
    print('   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\'ResembleAI/chatterbox-turbo\')"')
    exit(1)

required_files = ["t3_turbo_v1.safetensors"]
missing = [f for f in required_files if not (ckpt_path / f).exists()]
if missing:
    print(f"ERROR: Missing files in {CKPT_DIR}: {missing}")
    exit(1)

print(f"Using local checkpoint: {CKPT_DIR}")

# ────────────────────────────────────────────────
# Device & dtype
# ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Device: {device} | dtype: {dtype}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ────────────────────────────────────────────────
# Audio output device selection
# ────────────────────────────────────────────────
selected_device = None

def select_output_device():
    global selected_device

    print("\n=== Available audio output devices ===")
    devices = sd.query_devices()
    output_devices = []

    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            output_devices.append(i)
            print(f"[{i}] {dev['name']}")
            print(f"    Channels: {dev['max_output_channels']} | Default samplerate: {dev['default_samplerate']:.0f} Hz")
            print(f"    Latency: {dev['default_low_output_latency']:.3f} – {dev['default_high_output_latency']:.3f} s")
            print()

    if not output_devices:
        print("No output devices found! Check Windows sound settings.")
        exit(1)

    default_idx = sd.default.device[1] if sd.default.device[1] is not None else output_devices[0]
    print(f"Default device appears to be: [{default_idx}] {devices[default_idx]['name']}")

    try:
        choice = input("\nEnter the number of the device to use (press Enter for default): ").strip()
        if choice:
            selected_device = int(choice)
            if selected_device not in output_devices:
                raise ValueError
        else:
            selected_device = default_idx

        sd.default.device = (None, selected_device)  # output device only
        print(f"→ Using: [{selected_device}] {devices[selected_device]['name']}\n")
    except:
        print("Invalid selection → falling back to default device.\n")
        selected_device = default_idx
        sd.default.device = (None, selected_device)

# Run selection once at startup
select_output_device()

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
print("\nLoading Chatterbox-Turbo from local files...")
model = ChatterboxTurboTTS.from_local(
    ckpt_dir=str(ckpt_path),
    device=device
)
print("Model loaded successfully – fully offline.\n")

# ────────────────────────────────────────────────
# Sentence splitter (abbreviation-safe)
# ────────────────────────────────────────────────
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
    for i, abbr_pattern in enumerate(ABBREVIATIONS):
        for match in re.finditer(abbr_pattern, text, re.IGNORECASE):
            placeholder = f"__ABBR_{i}_{len(protected)}__"
            text = text[:match.start()] + placeholder + text[match.end():]
            protected[placeholder] = match.group(0)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    restored = []
    for s in sentences:
        for ph, orig in protected.items():
            s = s.replace(ph, orig)
        cleaned = s.strip()
        if cleaned:
            restored.append(cleaned)

    return restored if restored else [text]

# ────────────────────────────────────────────────
# Speak one sentence
# ────────────────────────────────────────────────
def speak_sentence(sentence: str):
    print(f"→ {sentence}")

    try:
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wav = model.generate(
                        text=sentence,
                        audio_prompt_path=str(AUDIO_PROMPT_PATH) if AUDIO_PROMPT_PATH else None,
                    )
                    wav = wav.squeeze()                     # remove any extra dims (e.g. [1, T] → [T])
        
                    # Normalize to [-1, 1] range if needed
                    if wav.abs().max() > 1.0:
                        wav = wav / wav.abs().max()
                    
                    # Always convert to float32 numpy
                    wav_np = wav.cpu().float().numpy()
                    
                    # Debug: print stats to check if audio has content
                    print(f"    Audio stats: shape={wav_np.shape}, max={wav_np.max():.4f}, min={wav_np.min():.4f}, mean={wav_np.mean():.4f}")
                    
                    try:
                        sd.play(wav_np, samplerate=int(model.sr), device=selected_device, blocking=False)
                        sd.wait()
                    except Exception as play_err:
                        print(f"    Playback exception: {type(play_err).__name__}: {str(play_err)}")
                        # Fallback: try forcing 44100 Hz resampling
                        import torchaudio.transforms as T
                        resampler = T.Resample(orig_freq=model.sr, new_freq=44100)
                        wav_resampled = resampler(wav.cpu())
                        wav_res_np = wav_resampled.float().numpy()
                        print("    Trying resampled at 44100 Hz...")
                        sd.play(wav_res_np, samplerate=44100, device=selected_device)
                        sd.wait()
            else:
                wav = model.generate(
                    text=sentence,
                    audio_prompt_path=str(AUDIO_PROMPT_PATH) if AUDIO_PROMPT_PATH else None,
                )

        wav_cpu = wav.cpu().numpy()  # sounddevice prefers numpy array
        sd.play(wav_cpu, samplerate=model.sr, device=selected_device)
        sd.wait()

        # Save
        idx = len(list(TEMP_DIR.glob("out_*.wav")))
        safe_text = sentence[:35].replace(" ", "_").replace(".", "").replace("!", "").replace("?", "")
        out_path = TEMP_DIR / f"out_{idx:03d}_{safe_text}.wav"
        ta.save(out_path, torch.from_numpy(wav_cpu), model.sr)
        print(f"  Saved → {out_path.name}\n")

    except Exception as e:
        print(f"Playback / generation failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()  # ← this prints the full stack trace
        if "memory" in str(e).lower():
            print("→ Try shorter text or close other GPU apps.")

# ────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chatterbox-Turbo – Offline real-time TTS")
    print("  Enter text (multiple sentences OK) → Enter")
    print("  Empty line or Ctrl+C → quit\n")

    while True:
        try:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            sentences = split_sentences(user_text)

            for sent in sentences:
                speak_sentence(sent)

        except KeyboardInterrupt:
            print("\nExited.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")