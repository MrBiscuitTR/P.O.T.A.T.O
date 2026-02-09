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
- No `sampler` attribute
- No `scheduler` attribute
- No `noise_scheduler` attribute
- NOT a PyTorch `nn.Module` (no `.parameters()` or `.eval()`)
- Cannot use `torch.compile()` on it
- s3gen has no configurable step counters (only `resamplers` attribute)

## Supported Parameters

### model.generate() - ONLY supports:
```python
model.generate(
    text="Hello, world!",           # Required: Text to synthesize
    audio_prompt_path="/path/to/voice.wav"  # Optional: Voice reference for cloning
)
```

### NOT Supported (will cause errors):
- `num_steps` - Model has fixed internal steps
- `cfg_weight` - Turbo version doesn't support CFG
- `exaggeration` - Turbo version doesn't support this
- `min_p` - Not a parameter
- Any scheduler/sampler configuration

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

## TTS Pipeline Architecture

### Streaming Pipeline (current implementation)

```
LLM Response (complete text)
    |
    v
split_sentences(text)  -- groups 2-3 sentences per chunk, abbreviation-safe
    |
    v
generate_tts_wav_streamed(text)  -- Python generator, yields one WAV chunk at a time
    |                                checks stop_event between every chunk
    v
/api/vox_speak_wav_stream  -- SSE endpoint, streams base64-encoded WAV chunks
    |
    v
vox.js speakText()  -- reads SSE events, decodes base64 -> Audio elements
    |
    v
voxAudioQueue[]  -- browser-side FIFO queue of Audio elements
    |
    v
_drainAudioQueue()  -- plays chunks back-to-back with no gap
                       starts as soon as the FIRST chunk arrives
```

**Key behavior:**
- Audio playback starts as soon as the first chunk is generated (not after all chunks)
- While the browser plays chunk N, the server generates chunk N+1
- This gives near-instant perceived playback for long responses
- Stop buttons flush the queue, abort the SSE fetch, AND set the backend stop flag

### Stop Mechanism

```
User presses "Stop Speaking" or "Stop All"
    |
    v
_flushAudioQueue()  -- (browser) pauses all Audio elements, clears queue,
    |                   aborts AbortController on the SSE fetch
    v
fetch('/api/vox_stop')  -- (backend) sets stop_event flag
    |
    v
stop_current_tts()  -- sets stop_event, calls sd.stop(), clears audio_q
    |
    v
generate_tts_wav_streamed() checks stop_event before each chunk -> returns early
```

Both browser-side and Python-side cancellation happen. The SSE fetch abort
causes the Flask generator to stop being consumed. The stop_event flag
causes the Python generate loop to exit before starting the next chunk.

## API Endpoints

### TTS Generation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vox_speak_wav` | POST | Generate complete WAV (legacy, still works) |
| `/api/vox_speak_wav_stream` | POST | **Stream WAV chunks via SSE (preferred)** |
| `/api/vox_speak` | POST | Synchronous speaker playback (desktop, not browser) |

### TTS Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vox_stop` | POST | Stop current TTS generation + playback |
| `/api/vox_stop_speak` | POST | Alias for vox_stop |
| `/api/stop_all_vox` | POST | Emergency stop: TTS + LLM + recording |
| `/api/tts_is_speaking` | GET | Check if TTS is currently active |

### TTS Model Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts_load_turbo` | POST | Load English TTS model into VRAM |
| `/api/tts_load_multilingual` | POST | Load multilingual TTS model |
| `/api/unload_tts` | POST | Unload TTS models from VRAM |
| `/api/tts_unload` | POST | Alternative unload with thread cleanup |

## Key Functions (clonevoice_turbo.py)

| Function | Description |
|----------|-------------|
| `_get_model()` | Lazy-load model on first use (thread-safe) |
| `split_sentences(text)` | Split text into 2-3 sentence groups, abbreviation-safe |
| `generate_tts_wav(text)` | Generate complete WAV bytes (legacy) |
| `generate_tts_wav_streamed(text)` | **Generator: yields (idx, total, wav_bytes) per chunk** |
| `generate_worker(groups, q)` | Producer thread for desktop playback |
| `playback_worker(q, sr)` | Consumer thread for desktop playback |
| `speak_sentences_grouped(text)` | Desktop speaker playback (not used by browser) |
| `stop_current_tts()` | Set stop flag, clear queues, stop audio |
| `shutdown_tts()` | Full shutdown: stop + unload model |
| `is_speaking()` | Check `_is_speaking` flag |

## Key Functions (vox.js - browser)

| Function | Description |
|----------|-------------|
| `speakText(text, lang)` | Streams TTS via SSE, queues + plays audio chunks |
| `_drainAudioQueue()` | Plays queued Audio elements back-to-back |
| `_flushAudioQueue()` | Stops playback, clears queue, aborts SSE fetch |
| `stopVOXSpeaking()` | Flush queue + tell backend to cancel |
| `stopAllVOX()` | Flush queue + stop recording + stop all backend |

## Performance Optimizations Applied

### What WORKS:
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

3. **Inference Mode (legacy endpoint):**
   ```python
   with torch.inference_mode():
       # Faster than torch.no_grad()
   ```

4. **Batch GPU->CPU Transfers (legacy endpoint):**
   - Keep all chunks on GPU during generation
   - Concatenate on GPU: `torch.cat(chunks, dim=0)`
   - Single transfer to CPU at the end

5. **Streaming Pipeline (new):**
   - Each chunk transferred to CPU immediately after generation
   - Browser starts playing while next chunk generates
   - Perceived latency = time to generate first chunk only

6. **Audio Prompt Path Caching:**
   - Path string cached at module load time
   - Model internally caches the loaded waveform after first call

7. **CUDA Cache Management:**
   ```python
   torch.cuda.empty_cache()  # Before and after generation batches
   ```

### What DOESN'T WORK:
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

- **Original speed**: 1.5-2 seconds for 250+ characters per chunk
- **Perceived speed with streaming**: Audio starts playing after first chunk (~1.5-2s)
  instead of waiting for ALL chunks to finish
- **Slowdown causes**: Usually NOT the model itself, but:
  - Multiple sequential CPU<->GPU transfers (fixed in legacy endpoint)
  - Not using autocast/inference_mode
  - CUDA cache accumulation

## Sentence Chunking Strategy

The `split_sentences()` function:
1. Protects abbreviations (Mr., Dr., U.S., F.B.I., etc.) from being split
2. Splits on sentence-ending punctuation followed by whitespace: `.` `!` `?`
3. Groups sentences into chunks of 3
4. If the last group has fewer than 3 sentences, merges it into the previous group
5. If total sentences <= 3, returns everything as a single chunk

This ensures each TTS chunk is 2-3 sentences long - enough for natural-sounding
speech but short enough for fast generation.
