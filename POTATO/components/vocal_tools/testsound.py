# test_playback.py
import sounddevice as sd
import numpy as np

# Simple 440 Hz beep for 1 second
fs = 44100
t = np.linspace(0, 1, fs, False)
tone = 0.5 * np.sin(2 * np.pi * 440 * t)

print("Trying to play a test tone...")
sd.play(tone, fs)   # or sd.play(tone, fs, device=YOUR_INDEX)
sd.wait()
print("Playback done. Did you hear a beep?")