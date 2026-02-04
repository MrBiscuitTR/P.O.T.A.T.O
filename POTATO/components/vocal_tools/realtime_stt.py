"""
realtime_stt.py - Real-time Speech-to-Text without ffmpeg
Uses sounddevice for audio capture and Whisper for transcription
"""
import sounddevice as sd
import numpy as np
import torch
import queue
import threading
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional, Callable
import json
from pathlib import Path

# Global state
_whisper_pipeline = None
_audio_queue = queue.Queue()
_is_listening = False
_listen_thread = None

# Preferences file
PREFERENCES_FILE = Path(__file__).parent.parent.parent / ".data" / "preferences.json"


def load_preferences():
    """Load audio device preferences"""
    try:
        if PREFERENCES_FILE.exists():
            with open(PREFERENCES_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading preferences: {e}")
    return {}


def save_preferences(prefs):
    """Save audio device preferences"""
    try:
        PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREFERENCES_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"Error saving preferences: {e}")


def list_audio_devices():
    """
    List all available audio input devices.
    Returns list of dicts with 'index', 'name', 'channels', 'sample_rate'
    """
    devices = sd.query_devices()
    input_devices = []
    
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'index': idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': int(device['default_samplerate'])
            })
    
    return input_devices


def detect_working_device():
    """
    Test each input device to find one that works.
    Returns dict with device info or None if none work.
    Filters out output-only devices like speakers.
    """
    devices = list_audio_devices()
    
    # Filter out known output device keywords
    output_keywords = ['speaker', 'output', 'headphone', 'hdmi', 'display']
    
    for device in devices:
        # Skip if device name suggests it's an output device
        device_name_lower = device['name'].lower()
        if any(keyword in device_name_lower for keyword in output_keywords):
            print(f"⊗ Skipping output device: {device['name']}")
            continue
        
        try:
            # Try to record 0.5 seconds from this device
            test_audio = sd.rec(
                int(0.5 * device['sample_rate']),
                samplerate=device['sample_rate'],
                channels=1,
                device=device['index'],
                dtype='float32'
            )
            sd.wait()
            
            # Check if we got actual audio data (not all zeros)
            if np.abs(test_audio).max() > 0.001:
                print(f"✓ Working device found: {device['name']}")
                return device
        except Exception as e:
            print(f"✗ Device {device['name']} failed: {e}")
            continue
    
    return None


def get_whisper_pipeline():
    """Get or create Whisper pipeline"""
    global _whisper_pipeline
    
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    
    print(f"Loading Whisper model on {device}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    
    _whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("Whisper model loaded successfully")
    return _whisper_pipeline


def transcribe_audio_chunk(audio_data: np.ndarray, sample_rate: int, language: str = "en") -> str:
    """
    Transcribe a chunk of audio data.
    
    Args:
        audio_data: numpy array of audio samples (float32, range -1 to 1)
        sample_rate: sample rate of the audio
        language: language code
    
    Returns:
        Transcribed text
    """
    try:
        pipe = get_whisper_pipeline()
        
        # Whisper expects 16kHz audio
        if sample_rate != 16000:
            # Resample (simple decimation - for production use scipy.signal.resample)
            audio_data = audio_data[::int(sample_rate / 16000)]
        
        # Ensure audio is 1D
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Transcribe
        result = pipe(
            audio_data.astype(np.float32),
            generate_kwargs={"language": language}
        )
        
        return result['text'].strip()
    
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def start_realtime_listening(
    callback: Callable[[str], None],
    device_index: Optional[int] = None,
    language: str = "en",
    chunk_duration: float = 3.0,
    silence_threshold: float = 0.01
):
    """
    Start real-time audio listening and transcription.
    
    Args:
        callback: Function to call with transcribed text
        device_index: Audio device index (None = auto-detect)
        language: Language code for transcription
        chunk_duration: Duration of audio chunks to process (seconds)
        silence_threshold: RMS threshold below which audio is considered silence
    """
    global _is_listening, _listen_thread, _audio_queue
    
    if _is_listening:
        print("Already listening")
        return
    
    # Get or detect device
    if device_index is None:
        prefs = load_preferences()
        device_index = prefs.get('audio_device_index')
        
        if device_index is None:
            print("Auto-detecting working audio device...")
            device_info = detect_working_device()
            if device_info:
                device_index = device_info['index']
                # Save preference
                prefs['audio_device_index'] = device_index
                prefs['audio_device_name'] = device_info['name']
                save_preferences(prefs)
                print(f"Saved device preference: {device_info['name']}")
            else:
                raise RuntimeError("No working audio input device found")
    
    # Get device info
    device_info = sd.query_devices(device_index)
    sample_rate = int(device_info['default_samplerate'])
    
    print(f"Starting real-time STT with device: {device_info['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    
    _is_listening = True
    _audio_queue = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}")
        _audio_queue.put(indata.copy())
    
    def process_audio_loop():
        """Background thread that processes audio chunks"""
        buffer = np.array([], dtype='float32')
        chunk_samples = int(chunk_duration * sample_rate)
        
        while _is_listening:
            try:
                # Get audio data (with timeout to allow checking _is_listening)
                audio_chunk = _audio_queue.get(timeout=0.1)
                
                # Flatten if stereo
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.mean(axis=1)
                
                # Add to buffer
                buffer = np.concatenate([buffer, audio_chunk])
                
                # Process when we have enough samples
                if len(buffer) >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
                    
                    # Check for silence
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms < silence_threshold:
                        continue
                    
                    # Transcribe
                    text = transcribe_audio_chunk(chunk, sample_rate, language)
                    if text:
                        callback(text)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    # Start audio stream
    stream = sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=sample_rate,
        callback=audio_callback,
        blocksize=int(sample_rate * 0.1)  # 100ms blocks
    )
    
    # Start processing thread
    _listen_thread = threading.Thread(target=process_audio_loop, daemon=True)
    _listen_thread.start()
    
    stream.start()
    
    print("Real-time STT started")
    
    # Keep reference to stream
    globals()['_audio_stream'] = stream


def stop_realtime_listening():
    """Stop real-time audio listening"""
    global _is_listening
    
    _is_listening = False
    
    if '_audio_stream' in globals():
        try:
            globals()['_audio_stream'].stop()
            globals()['_audio_stream'].close()
        except:
            pass
    
    print("Real-time STT stopped")


if __name__ == "__main__":
    # Test the module
    print("Testing real-time STT...")
    print("\nAvailable audio devices:")
    for device in list_audio_devices():
        print(f"  [{device['index']}] {device['name']} - {device['channels']} channels @ {device['sample_rate']} Hz")
    
    print("\nDetecting working device...")
    device = detect_working_device()
    if device:
        print(f"Using device: {device['name']}")
        
        def on_transcription(text):
            print(f"Transcribed: {text}")
        
        start_realtime_listening(callback=on_transcription, device_index=device['index'])
        
        input("\nPress Enter to stop...\n")
        stop_realtime_listening()
    else:
        print("No working device found!")
