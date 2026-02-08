# P.O.T.A.T.O Comprehensive Fixes - Implementation Guide

## CRITICAL FIXES COMPLETED ✅

### 1. TTS Queuing & Playback (clonevoice_turbo.py)
- ✅ Separated generation and playback into two threads
- ✅ Removed `maxsize=2` limit on audio queue
- ✅ Playback worker continuously processes queue without gaps
- ✅ `blocking=True` in sd.play() for seamless audio
- ✅ Stop function clears queue without crashing

### 2. CUDA GPU Allocation (clonevoice_turbo.py)
- ✅ Added `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- ✅ Added `torch.cuda.set_device(0)`
- ✅ Forces all models to NVIDIA GPU #0

### 3. Safe Stop Functions (clonevoice_turbo.py)
- ✅ `stop_current_tts()` - Sets event, stops audio, clears queue, resets event
- ✅ `shutdown_tts()` - Stops threads safely, unloads model, frees VRAM
- ✅ NO app.py termination

## INSTRUCTIONS FOR REMAINING FIXES

### Fix #1: Stop Generation Button (Chat Page)
**Location**: `POTATO/webui/app.py` + `POTATO/webui/static/script.js`

**Step 1**: Add endpoint in app.py:
```python
@app.route('/api/stop_model', methods=['POST'])
def stop_model():
    """Stop currently running Ollama model"""
    try:
        data = request.json
        model_name = data.get('model') or currentModel
        import ollama
        ollama.stop(model_name)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

**Step 2**: Update script.js stopGeneration():
```javascript
async function stopGeneration() {
    isGenerating = false;
    if (currentAbortController) {
        currentAbortController.abort();
    }
    // Also stop the model
    try {
        await fetch('/api/stop_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: currentModel})
        });
    } catch(e) {
        console.error('Error stopping model:', e);
    }
}
```

### Fix #2: Auto-Save New Chats
**Location**: `POTATO/webui/app.py` in `chat_stream()` function

**Current flow**: User message → AI generates → Save with AI-generated title
**New flow**: User message → **Save immediately** → AI generates → Update title

**Implementation**: In chat_stream(), right after receiving user message:
```python
# Around line 350 in chat_stream()
history.append({"role": "user", "content": user_message})

# IMMEDIATE SAVE with first 30 chars
if not existing_chat:
    temp_title = user_message[:30] + ("..." if len(user_message) > 30 else "")
    save_chat_session(session_id, history, title=temp_title)
    
# Continue with AI generation...
```

### Fix #3: Multi-Tool Execution
**Location**: `POTATO/main.py` in tool execution loop

**Current issue**: Loop breaks after first tool
**Fix**: Remove early `break` statements, allow multiple tool calls

Find this section (around line 450-500):
```python
for tool_call in tool_calls:
    # Execute tool
    result = execute_tool(tool_call)
    # DON'T BREAK HERE - continue loop
```

### Fix #4: Search Pagination
**Location**: `POTATO/MCP/searx_mcp.py`

**Already provided fix above** - Add `page` parameter and `'pageno': page` to params dict.

### Fix #5: Interrupt Words
**Location**: `POTATO/webui/app.py` in `vox_stream()`

**Current**: Checks words but doesn't stop TTS properly
**Fix**: Import and call stop_current_tts():
```python
# At top of file
from POTATO.components.vocal_tools.clonevoice_turbo import stop_current_tts

# In vox_stream() around line 910:
if any(word in user_text.lower() for word in interrupt_words):
    stop_current_tts()  # Add this line
    yield f"data: {json.dumps({'status': 'interrupted', ...})}\n\n"
    return
```

### Fix #6: Continuous Mode Not Transcribing
**Possible causes**:
1. MediaRecorder not stopping
2. onstop callback not firing
3. voxAudioChunks empty

**Debug**: Add logging in stopVOXRecording():
```javascript
console.log('[VOX] stopVOXRecording called, MediaRecorder state:', voxMediaRecorder?.state);
console.log('[VOX] Audio chunks collected:', voxAudioChunks.length);
```

**If MediaRecorder.state is 'inactive'**: Already stopped, onstop won't fire again
**Solution**: Check state before calling stop():
```javascript
if (voxMediaRecorder && voxMediaRecorder.state === 'recording') {
    voxMediaRecorder.stop();
} else {
    console.warn('[VOX] MediaRecorder not recording, state:', voxMediaRecorder?.state);
}
```

## RESTART APP AFTER FIXES
```bash
# Kill current instance (Ctrl+C in terminal)
python -m POTATO.webui.app
```

## VERIFICATION TESTS
1. **TTS Queue**: Say long paragraph → All chunks play smoothly
2. **Stop Speaking**: Click mid-sentence → Stops immediately, no crash
3. **Continuous Mode**: Say something → Auto-stops after silence → Transcribes → Responds → Repeats
4. **GPU Usage**: Run `nvidia-smi` → All VRAM on GPU #0, 0MB on iGPU
5. **Stop Generation**: Start chat response → Click stop → Model stops
6. **Auto-Save**: New chat → Type message → Refresh page → Chat exists with first 30 chars
7. **Multi-Tool**: Ask "search X and Y" → Both searches execute
8. **Pagination**: Ask "search page 2 of X" → Returns page 2 results

## FILES MODIFIED
- ✅ POTATO/components/vocal_tools/clonevoice_turbo.py (DONE)
- TODO: POTATO/webui/app.py (stop endpoint, auto-save, interrupt words)
- TODO: POTATO/webui/static/script.js (stop button)
- TODO: POTATO/webui/static/vox.js (continuous mode debug)
- TODO: POTATO/MCP/searx_mcp.py (pagination)
- TODO: POTATO/main.py (multi-tool execution)
