"""
audioflow.py - Audio I/O module for P.O.T.A.T.O
Uses Whisper-large-v3 for STT and Chatterbox for TTS
All offline, using local models
"""
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional, Union

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force CUDA GPU (not iGPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# Global variables for model caching
_whisper_model = None
_whisper_processor = None
_whisper_pipeline = None

# TTS speaking state (shared with VOX to prevent echo)
_tts_is_speaking = False

def _get_whisper_pipeline():
    """
    Lazy-load Whisper model and pipeline (cached).
    Uses local model from Hugging Face cache.
    """
    global _whisper_model, _whisper_processor, _whisper_pipeline
    
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    
    print(f"Loading Whisper model from cache on {device}...")
    
    _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True  # Ensures offline operation
    )
    _whisper_model.to(device)
    
    _whisper_processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True
    )
    
    _whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=_whisper_model,
        tokenizer=_whisper_processor.tokenizer,
        feature_extractor=_whisper_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("Whisper model loaded successfully.\n")
    return _whisper_pipeline


def unload_whisper():
    """Unload Whisper model from VRAM to free memory"""
    global _whisper_model, _whisper_processor, _whisper_pipeline
    
    try:
        # Delete pipeline first
        if _whisper_pipeline is not None:
            del _whisper_pipeline
            _whisper_pipeline = None
            print("[Whisper] Pipeline unloaded")
        
        # Delete processor
        if _whisper_processor is not None:
            del _whisper_processor
            _whisper_processor = None
            print("[Whisper] Processor unloaded")
        
        # Delete model and move to CPU first to release CUDA memory
        if _whisper_model is not None:
            try:
                _whisper_model.to('cpu')
            except:
                pass
            del _whisper_model
            _whisper_model = None
            print("[Whisper] Model unloaded")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[Whisper] CUDA cache cleared")
        
        print("→ Whisper completely unloaded from VRAM")
        return True
        
    except Exception as e:
        print(f"[Whisper] Error during unload: {e}")
        import traceback
        traceback.print_exc()
        return False


def listen_stt(audio_input: Union[str, Path, np.ndarray], language: str = "en") -> dict:
    """
    Transcribe audio to text using Whisper-large-3.
    
    Args:
        audio_input: Either file path (str/Path) or numpy array of audio data
        language: Language code (default: "en" for English)
    
    Returns:
        dict with keys:
            - 'text': transcribed text
            - 'language': detected/specified language
            - 'success': bool indicating success
    """
    try:
        pipe = _get_whisper_pipeline()
        
        # If it's a file path, load it
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                return {
                    'text': '',
                    'language': language,
                    'success': False,
                    'error': f"File not found: {audio_path}"
                }
            # Pipeline can handle file paths directly
            result = pipe(str(audio_path), generate_kwargs={"language": language})
        else:
            # Assume numpy array
            result = pipe(audio_input, generate_kwargs={"language": language})
        
        return {
            'text': result['text'].strip(),
            'language': language,
            'success': True
        }
    
    except Exception as e:
        return {
            'text': '',
            'language': language,
            'success': False,
            'error': str(e)
        }


def detect_language(audio_input: Union[str, Path, np.ndarray]) -> str:
    """
    Detect language from audio input.
    
    Args:
        audio_input: Either file path or numpy array
    
    Returns:
        Language code (e.g., 'en', 'es', 'fr')
    """
    try:
        pipe = _get_whisper_pipeline()
        
        # Transcribe without language specification to detect
        if isinstance(audio_input, (str, Path)):
            result = pipe(str(audio_input))
        else:
            result = pipe(audio_input)
        
        # Whisper doesn't return language in basic pipeline
        # For now, default to 'en', but you can enhance this
        return "en"
    
    except Exception as e:
        print(f"Language detection error: {e}")
        return "en"


# Import TTS functions from existing modules
try:
    from POTATO.components.vocal_tools.clonevoice_turbo import speak_sentences_grouped, stop_current_tts, shutdown_tts
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: TTS modules not available")
    TTS_AVAILABLE = False
    
    def speak_sentences_grouped(text, **kwargs):
        print(f"[TTS FALLBACK] Would speak: {text}")
    
    def stop_current_tts():
        print("[TTS FALLBACK] Would stop TTS")
    
    def shutdown_tts():
        print("[TTS FALLBACK] Would shutdown TTS")


def speak(text: str, language: str = "en"):
    """
    Text-to-speech wrapper.
    Uses English TTS by default, multilingual if language is not English.
    
    Args:
        text: Text to speak
        language: Language code (default: "en")
    """
    if not TTS_AVAILABLE:
        print(f"[TTS] {text}")
        return
    
    if language == "en":
        # Use clonevoice_turbo for English
        speak_sentences_grouped(text)
    else:
        # Use multilingual TTS
        try:
            from POTATO.components.vocal_tools.clonevoice_multilanguage import speak_multilingual
            speak_multilingual(text, language=language)
        except ImportError:
            print(f"[TTS] Multilingual TTS not available, using English TTS")
            speak_sentences_grouped(text)


if __name__ == "__main__":
    # Test STT
    print("Testing audioflow module...")
    audio_file = input("Enter path to audio file (or press Enter to skip): ").strip()
    
    if audio_file:
        result = listen_stt(audio_file)
        if result['success']:
            print(f"\nTranscription: {result['text']}")
            print(f"Language: {result['language']}")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
    
    # Test TTS
    test_text = "Hello, this is a test of the audio flow module."
    print(f"\nTesting TTS with: {test_text}")
    speak(test_text)