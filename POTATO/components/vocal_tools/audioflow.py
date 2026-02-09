"""
audioflow.py - Audio I/O module for P.O.T.A.T.O
Uses Whisper-large-v3 for STT and Chatterbox for TTS
All offline, using local models

NOTE: This module now uses realtime_stt's Whisper pipeline to avoid
loading duplicate models (saves 3GB VRAM).
"""
import os
import torch
import numpy as np
from pathlib import Path
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

# TTS speaking state (shared with VOX to prevent echo)
_tts_is_speaking = False

def _get_whisper_pipeline():
    """
    Get Whisper pipeline from realtime_stt (shared model).
    This avoids loading duplicate Whisper models, saving 3GB VRAM.
    """
    from POTATO.components.vocal_tools.realtime_stt import get_whisper_pipeline
    return get_whisper_pipeline()
    return _whisper_pipeline


def unload_whisper():
    """Unload Whisper model from VRAM (delegates to realtime_stt)"""
    try:
        from POTATO.components.vocal_tools.realtime_stt import unload_whisper_model
        return unload_whisper_model()
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
            with torch.no_grad():
                result = pipe(str(audio_path), generate_kwargs={"language": language})
        else:
            # Assume numpy array
            with torch.no_grad():
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
        with torch.no_grad():
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