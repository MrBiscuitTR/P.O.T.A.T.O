#!/usr/bin/env python3
"""
Memory diagnostic script for P.O.T.A.T.O
Shows what's using RAM vs VRAM
"""
import sys
import gc
import psutil
import torch

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()

    print("\n" + "="*60)
    print("MEMORY USAGE REPORT")
    print("="*60)

    # System RAM
    ram_gb = mem_info.rss / 1e9
    print(f"\nSystem RAM used by app.py: {ram_gb:.2f} GB")

    # VRAM
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"\nCUDA Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved:  {reserved:.2f} GB")
            print(f"  - Total:     {total:.2f} GB")
    else:
        print("\nNo CUDA devices available")

    print("\n" + "="*60)

def test_whisper_loading():
    """Test Whisper model loading"""
    print("\n[TEST] Loading Whisper model...")
    get_memory_usage()

    from POTATO.components.vocal_tools.realtime_stt import get_whisper_pipeline
    pipe = get_whisper_pipeline()

    print("\n[TEST] Whisper model loaded")
    get_memory_usage()

    return pipe

def test_chatterbox_loading():
    """Test Chatterbox model loading"""
    print("\n[TEST] Loading Chatterbox model...")
    get_memory_usage()

    from POTATO.components.vocal_tools.clonevoice_turbo import _get_model
    model = _get_model()

    print("\n[TEST] Chatterbox model loaded")
    get_memory_usage()

    return model

if __name__ == "__main__":
    print("P.O.T.A.T.O Memory Diagnostic")
    print("Starting diagnosis...")

    # Initial state
    get_memory_usage()

    # Test Whisper
    try:
        whisper = test_whisper_loading()
    except Exception as e:
        print(f"\n[ERROR] Whisper loading failed: {e}")

    # Test Chatterbox
    try:
        chatterbox = test_chatterbox_loading()
    except Exception as e:
        print(f"\n[ERROR] Chatterbox loading failed: {e}")

    print("\nDiagnosis complete!")
