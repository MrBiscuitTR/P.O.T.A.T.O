#!/usr/bin/env python3
"""
Test script to force TTS model reload and see debug output
"""
import sys
sys.path.insert(0, ".")

print("="*70)
print("TTS MODEL RELOAD TEST")
print("="*70)

print("\n[1] Unloading any existing TTS model...")
from POTATO.components.vocal_tools import clonevoice_turbo
clonevoice_turbo.shutdown_tts()

print("\n[2] Loading TTS model (will show comprehensive debug output)...")
model = clonevoice_turbo._get_model()

print("\n[3] Model loaded successfully!")
print(f"    Model type: {type(model)}")

print("\n[4] Testing short generation...")
text = "Hello, this is a test."
try:
    wav_bytes = clonevoice_turbo.generate_tts_wav(text)
    print(f"    ✓ Generated {len(wav_bytes)} bytes of audio")
except Exception as e:
    print(f"    ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete. Check the debug output above for model structure.")
print("="*70)
