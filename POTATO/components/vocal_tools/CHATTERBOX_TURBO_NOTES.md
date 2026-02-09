# Chatterbox Turbo TTS - Technical Notes

## Model Architecture (Confirmed via Introspection)

### What the Model HAS:
- **Main components:**
  - `s3gen`: S3Token2Wav (converts tokens to waveform)
  - `t3`: T3 (text/token processing)
  - `ve`: VoiceEncoder (voice cloning)
- **Device**: Handles its own CUDA placement
- **Sample rate**: `model.sr` attribute

### What the Model DOES NOT HAVE:
- ❌ No `sampler` attribute
- ❌ No `scheduler` attribute
- ❌ No `noise_scheduler` attribute
- ❌ NOT a PyTorch `nn.Module` (no `.parameters()` or `.eval()`)
- ❌ Cannot use `torch.compile()` on it
- ❌ s3gen has no configurable step counters (only `resamplers` attribute)

## Supported Parameters

### model.generate() - ONLY supports:
```python
model.generate(
    text="Hello, world!",           # Required: Text to synthesize
    audio_prompt_path="/path/to/voice.wav"  # Optional: Voice reference for cloning
)
```

### NOT Supported (will cause errors):
- ❌ `num_steps` - Model has fixed internal steps
- ❌ `cfg_weight` - Turbo version doesn't support CFG
- ❌ `exaggeration` - Turbo version doesn't support this
- ❌ `min_p` - Not a parameter
- ❌ Any scheduler/sampler configuration

## Progress Bar Behavior

When generating TTS, you'll see progress like:
```
5%|████ | 50/1000 [00:00<00:14, 64.02it/s]
```

**What this means:**
- The `/1000` is an internal maximum limit (NOT actual steps taken)
- The first number (e.g., `50`) varies based on **text length**
- Longer text = higher first number (e.g., `659/1000` for long responses)
- This is **normal** and doesn't indicate slowdown
- The number increasing across responses is expected for varying text lengths

## Performance Optimizations Applied

### ✅ What WORKS:
1. **CUDA Performance Settings:**
   ```python
   torch.backends.cudnn.benchmark = True
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

2. **Autocast for Inference:**
   ```python
   with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
       wav = model.generate(...)
   ```

3. **Inference Mode:**
   ```python
   with torch.inference_mode():
       # Faster than torch.no_grad()
   ```

4. **Batch GPU→CPU Transfers:**
   - Keep all chunks on GPU during generation
   - Concatenate on GPU: `torch.cat(chunks, dim=0)`
   - Single transfer to CPU at the end

5. **CUDA Cache Management:**
   ```python
   torch.cuda.empty_cache()  # After generation
   ```

### ❌ What DOESN'T WORK:
1. Setting `model.sampler.num_steps = 1` (no sampler attribute)
2. Calling `model.eval()` (not an nn.Module)
3. Disabling gradients via `param.requires_grad = False` (no .parameters())
4. Using `torch.compile()` on the model
5. Trying to configure internal step counts

## Memory Usage

### Expected Behavior:
- **VRAM**: ~2-3GB for model (on CUDA GPU 0)
- **RAM**: ~2-3GB during model loading (PyTorch loads to CPU first, then moves to CUDA)
- **After loading**: RAM usage drops, VRAM stays constant

### Optimization:
```python
# After model loads to CUDA, free CPU RAM
import gc
gc.collect()
```

## Speed Expectations

- **Original speed**: 1.5-2 seconds for 250+ characters
- **Current speed**: Should be similar (fast Turbo mode)
- **Slowdown causes**: Usually NOT the model itself, but:
  - Multiple sequential CPU↔GPU transfers
  - Not using autocast/inference_mode
  - CUDA cache accumulation

## VOX Integration Fixed Issues

### ✅ Fixes Applied:
1. **Duplicate Transcription** - Added `onstopProcessed` flag in vox.js
2. **Multiple Audio Playback** - Track all active audio elements in array
3. **Stop All Button** - Now stops ALL active audio, not just one
4. **Stop Generation** - Added stop flag checking in generate_tts_wav()
5. **Removed Invalid Code** - Cleaned up sampler/scheduler code that doesn't exist

## Testing

To test TTS generation:
```bash
python test_tts_reload.py
```

This will:
- Unload any existing model
- Load model with debug output
- Generate test audio
- Show VRAM usage

## Summary

Chatterbox Turbo is a **self-contained, optimized TTS model** that:
- Manages its own internal state
- Doesn't expose typical PyTorch model attributes
- Only needs `text` and optional `audio_prompt_path` parameters
- Progress bar is informational only (not configurable)
- Works best with CUDA autocast and inference mode enabled
