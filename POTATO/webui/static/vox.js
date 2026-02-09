// VOX - Voice Interaction Module
// Note: voxLanguage is defined in script.js and shared globally (for TTS)
let voxSessionId = null;
let voxIsListening = false;
let voxIsSpeaking = false;
let voxAudioContext = null;
let voxMediaRecorder = null;
let voxAudioChunks = [];
let voxSelectedDevice = null;
let voxSelectedOutputDevice = null;
let voxCurrentAudio = null;
let voxActiveAudioElements = [];  // Track ALL active audio elements
let whisperLoaded = false;
let whisperLoading = false;
let voxSilenceTimeout = null;
let voxLastAudioLevel = 0;

// Transcription language (separate from TTS language)
let voxTranscriptionLanguage = 'auto';  // Default to auto-detect

// CRITICAL: Global stop flag to prevent continuous mode from restarting
let voxShouldStop = false;

// Track manual unloads (different from graceful exit unloads)
window.sttManuallyUnloaded = false;
window.ttsManuallyUnloaded = false;
window.whisperModelLoaded = false;

// Create loading overlay
function createVOXLoadingOverlay() {
    const existingOverlay = document.getElementById('vox-loading-overlay');
    if (existingOverlay) return existingOverlay;
    
    const overlay = document.createElement('div');
    overlay.id = 'vox-loading-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(5px);
    `;
    
    overlay.innerHTML = `
        <div style="text-align: center; color: var(--text-color);">
            <div style="font-size: 24px; margin-bottom: 20px;">
                <i class="fas fa-microphone" style="color: var(--accent-color); animation: pulse 1.5s infinite;"></i>
            </div>
            <div id="vox-loading-status" style="font-size: 18px; margin-bottom: 10px;">Loading Whisper STT...</div>
            <div style="font-size: 14px; opacity: 0.7;">This may take a moment on first load</div>
            <div style="margin-top: 20px; width: 300px; height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px; overflow: hidden;">
                <div id="vox-loading-progress" style="height: 100%; width: 0%; background: var(--accent-color); transition: width 0.3s;"></div>
            </div>
        </div>
    `;
    
    document.body.appendChild(overlay);
    return overlay;
}

function updateVOXLoadingProgress(percent) {
    const progressBar = document.getElementById('vox-loading-progress');
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
    }
}

function removeVOXLoadingOverlay() {
    const overlay = document.getElementById('vox-loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// Load Whisper model
async function loadWhisperModel() {
    if (whisperLoaded || whisperLoading) return true;
    
    whisperLoading = true;
    const overlay = createVOXLoadingOverlay();
    
    try {
        updateVOXLoadingProgress(20);
        const statusEl = document.getElementById('vox-loading-status');
        if (statusEl) statusEl.textContent = 'Loading Whisper STT...';
        
        // Ping the transcribe endpoint to trigger model load
        const response = await fetch('/api/whisper_status');
        updateVOXLoadingProgress(50);
        
        if (!response.ok) {
            throw new Error('Failed to load Whisper');
        }
        
        updateVOXLoadingProgress(60);
        whisperLoaded = true;
        
        return true;
    } catch (error) {
        console.error('Error loading Whisper:', error);
        removeVOXLoadingOverlay();
        alert('Failed to load Whisper model. Please try again.');
        return false;
    } finally {
        whisperLoading = false;
    }
}

// Load TTS models
async function loadTTSModels(language = 'en') {
    try {
        updateVOXLoadingProgress(70);
        const statusEl = document.getElementById('vox-loading-status');
        if (statusEl) statusEl.textContent = 'Loading TTS models...';
        
        // Load appropriate TTS model based on language
        const endpoint = language === 'en' ? '/api/tts_load_turbo' : '/api/tts_load_multilingual';
        const response = await fetch(endpoint, { method: 'POST' });
        
        updateVOXLoadingProgress(90);
        
        if (!response.ok) {
            throw new Error('Failed to load TTS');
        }
        
        updateVOXLoadingProgress(100);
        
        setTimeout(() => {
            removeVOXLoadingOverlay();
        }, 300);
        
        return true;
    } catch (error) {
        console.error('Error loading TTS:', error);
        removeVOXLoadingOverlay();
        alert('Failed to load TTS model. Continuing with STT only.');
        return false;
    }
}

// Initialize VOX
async function initVOX() {
    try {
        // Load browser-level devices FIRST (immediate, no server dependency)
        await loadAudioDevices();
        await loadOutputDevices();

        // Load VOX model picker (fast API call)
        await loadVOXModel();

        // Load voice chat history (fast)
        await loadVoiceChatList();

        // Initialize audio context
        voxAudioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Load Whisper model with progress (slow - loads model into VRAM)
        const whisperOk = await loadWhisperModel();
        if (!whisperOk) return;

        // Load TTS models based on selected language (slow - loads model into VRAM)
        const language = document.getElementById('vox-tts-language')?.value || 'en';
        await loadTTSModels(language);

        // Start interrupt monitoring
        if (window.VOXInterrupt) {
            window.VOXInterrupt.start();
            console.log('[VOX] Interrupt monitoring started');
        }

        console.log('VOX initialized');
    } catch (error) {
        console.error('Error initializing VOX:', error);
    }
}

// Load available audio devices using browser API
async function loadAudioDevices() {
    try {
        const deviceSelect = document.getElementById('vox-device-select');
        if (!deviceSelect) return;

        // Request permission first so labels are populated
        try {
            const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            tempStream.getTracks().forEach(t => t.stop());
        } catch (e) {
            console.warn('[VOX] Mic permission denied, device labels may be empty');
        }

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = allDevices.filter(d => d.kind === 'audioinput');

        deviceSelect.innerHTML = '';

        // Default option
        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.textContent = 'Default microphone';
        if (!voxSelectedDevice) defaultOpt.selected = true;
        deviceSelect.appendChild(defaultOpt);

        audioInputs.forEach((device, idx) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Microphone ${idx + 1}`;
            if (device.deviceId === voxSelectedDevice) {
                option.selected = true;
            }
            deviceSelect.appendChild(option);
        });

        console.log(`[VOX] Found ${audioInputs.length} audio input(s)`);
    } catch (error) {
        console.error('Error loading audio devices:', error);
    }
}

// Auto-detect working audio device (browser-level - just use default)
function autoDetectDevice() {
    updateVOXStatus('Using default microphone');
    voxSelectedDevice = '';
}

// Set audio device (stores browser deviceId)
function setAudioDevice(deviceId, deviceName) {
    voxSelectedDevice = deviceId;
    console.log(`[VOX] Input device set to: ${deviceName || deviceId || 'default'}`);
}

// Load output devices using browser API
async function loadOutputDevices() {
    try {
        const deviceSelect = document.getElementById('vox-output-device-select');
        if (!deviceSelect) return;

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const audioOutputs = allDevices.filter(d => d.kind === 'audiooutput');

        deviceSelect.innerHTML = '';

        const defaultOpt = document.createElement('option');
        defaultOpt.value = '';
        defaultOpt.textContent = 'Default speaker';
        if (!voxSelectedOutputDevice) defaultOpt.selected = true;
        deviceSelect.appendChild(defaultOpt);

        audioOutputs.forEach((device, idx) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Speaker ${idx + 1}`;
            if (device.deviceId === voxSelectedOutputDevice) {
                option.selected = true;
            }
            deviceSelect.appendChild(option);
        });

        console.log(`[VOX] Found ${audioOutputs.length} audio output(s)`);
    } catch (error) {
        console.error('Error loading output devices:', error);
    }
}

// Set output device (stores browser deviceId)
function setOutputDevice(deviceId, deviceName) {
    voxSelectedOutputDevice = deviceId;
    console.log(`[VOX] Output device set to: ${deviceName || deviceId || 'default'}`);
}

// Load VOX model list and current selection
async function loadVOXModel() {
    try {
        const response = await fetch('/api/vox_model');
        const data = await response.json();
        const select = document.getElementById('vox-model-select');
        if (!select) return;
        select.innerHTML = '';
        (data.models || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === data.model) opt.selected = true;
            select.appendChild(opt);
        });
        if (data.models && data.models.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'No models found';
            select.appendChild(opt);
        }
    } catch (error) {
        console.error('[VOX] Error loading model list:', error);
    }
}

// Set VOX model
async function setVOXModel(model) {
    if (!model) return;
    try {
        await fetch('/api/vox_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model })
        });
        console.log(`[VOX] Model set to: ${model}`);
    } catch (error) {
        console.error('[VOX] Error setting model:', error);
    }
}

// Start voice recording
async function startVOXRecording() {
    console.log('[VOX] startVOXRecording called, voxIsListening:', voxIsListening);
    
    // Reset stop flag when user manually starts recording
    voxShouldStop = false;

    // Reload STT (Whisper) if it was unloaded
    if (!whisperLoaded) {
        window.sttManuallyUnloaded = false;
        window.whisperModelLoaded = false;
        updateVOXStatus('Loading STT...');
        const ok = await loadWhisperModel();
        if (!ok) { updateVOXStatus('STT load failed'); return; }
    }

    // Reload TTS if it was unloaded
    if (window.ttsManuallyUnloaded) {
        window.ttsManuallyUnloaded = false;
        const language = document.getElementById('vox-language')?.value || 'en';
        updateVOXStatus('Loading TTS...');
        await loadTTSModels(language);
    }
    
    if (voxIsListening) {
        console.log('[VOX] Already listening, stopping instead');
        stopVOXRecording();
        return;
    }
    
    // PRIORITY: Stop TTS if speaking (user speech takes priority)
    // Flush the entire audio queue so queued chunks don't play later
    if (voxCurrentAudio || voxAudioQueue.length > 0 || voxIsSpeaking) {
        console.log('[VOX] Stopping TTS - user is speaking');
        _flushAudioQueue();
        // Also stop backend TTS generation
        try {
            await fetch('/api/vox_stop_speak', { method: 'POST' });
        } catch (e) {
            console.log('[VOX] Failed to stop backend TTS:', e);
        }
    }
    
    try {
        console.log('[VOX] Starting recording...');
        updateVOXStatus('Listening...');
        voxIsListening = true;
        
        // Request microphone access
        console.log('[VOX] Requesting microphone access...');
        const audioConstraints = {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 16000
        };
        if (voxSelectedDevice) {
            audioConstraints.deviceId = { exact: voxSelectedDevice };
        }
        const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
        
        // Set up audio context for volume detection
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        const analyzer = audioContext.createAnalyser();
        analyzer.fftSize = 512;
        source.connect(analyzer);
        
        const dataArray = new Uint8Array(analyzer.frequencyBinCount);
        let hasSignificantAudio = false;
        let significantAudioCount = 0;
        const REQUIRED_SIGNIFICANT_CHECKS = 3; // Must have 3+ checks above threshold
        const SILENCE_THRESHOLD = 20; // Audio level below this is considered silence
        const SILENCE_DURATION = 1000; // Stop recording after 1.0s of silence
        let consecutiveSilenceChecks = 0;
        
        // Check audio level periodically - filters garbage/background noise AND detects silence
        const volumeCheckInterval = setInterval(() => {
            analyzer.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            voxLastAudioLevel = average;
            
            // Update waveform visualizer
            const waveform = document.getElementById('waveform');
            if (waveform) {
                const waves = waveform.querySelectorAll('.wave');
                const intensity = Math.min(average / 100, 1); // Normalize to 0-1
                waves.forEach((wave, i) => {
                    const height = 2 + (intensity * 30); // 2px to 32px height
                    wave.style.height = `${height}px`;
                });
            }
            
            // Check for significant audio (above 20 = actual speech)
            if (average > 22) {
                significantAudioCount++;
                consecutiveSilenceChecks = 0; // Reset silence counter
                if (significantAudioCount >= REQUIRED_SIGNIFICANT_CHECKS) {
                    hasSignificantAudio = true;
                }
            } 
            // If we've detected significant audio before, now check for silence
            else if (hasSignificantAudio && average < SILENCE_THRESHOLD) {
                // Only auto-stop if auto-silence detection is enabled
                const autoSilenceEnabled = document.getElementById('vox-auto-silence')?.checked;
                
                if (autoSilenceEnabled) {
                    consecutiveSilenceChecks++;
                    const silenceDurationMs = consecutiveSilenceChecks * 100;
                    const remainingMs = SILENCE_DURATION - silenceDurationMs;
                    
                    // Show countdown during silence
                    if (remainingMs > 0) {
                        updateVOXStatus(`Recording... (stopping in ${(remainingMs/1000).toFixed(1)}s)`);
                    }
                    
                    // Auto-stop after SILENCE_DURATION ms of silence
                    if (silenceDurationMs >= SILENCE_DURATION) {
                        console.log('[VOX] Silence detected for', silenceDurationMs, 'ms - auto-stopping recording');
                        if (voxMediaRecorder && voxMediaRecorder.state === 'recording') {
                            stopVOXRecording();
                        }
                    }
                }
            }
            // Reset silence counter if we get audio again
            else if (hasSignificantAudio && average >= SILENCE_THRESHOLD) {
                if (consecutiveSilenceChecks > 0) {
                    console.log('[VOX] Audio resumed, resetting silence counter');
                    consecutiveSilenceChecks = 0;
                    updateVOXStatus('Recording...');
                }
            }
        }, 100);
        
        console.log('[VOX] Microphone access granted, creating MediaRecorder...');
        
        // Use WAV format which soundfile/scipy can handle without ffmpeg
        const options = { mimeType: 'audio/wav' };
        
        // Fallback to webm if wav not supported, but try wav first
        if (!MediaRecorder.isTypeSupported('audio/wav')) {
            console.warn('[VOX] WAV not supported, trying webm');
            options.mimeType = 'audio/webm';
        }
        
        voxMediaRecorder = new MediaRecorder(stream, options);
        console.log('[VOX] MediaRecorder format:', options.mimeType);
        
        voxAudioChunks = [];
        
        console.log('[VOX] MediaRecorder created, setting up event handlers...');
        
        voxMediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                voxAudioChunks.push(event.data);
                
                // CRITICAL: Process chunk for interrupt detection
                if (window.VOXInterrupt) {
                    window.VOXInterrupt.processChunk(event.data);
                }
            }
        };
        
        // Use a flag to prevent duplicate onstop processing
        let onstopProcessed = false;

        voxMediaRecorder.onstop = async () => {
            // CRITICAL: Prevent duplicate processing
            if (onstopProcessed) {
                console.log('[VOX] onstop already processed, ignoring duplicate call');
                return;
            }
            onstopProcessed = true;

            clearInterval(volumeCheckInterval);
            audioContext.close();
            stream.getTracks().forEach(track => track.stop());

            // Only transcribe if significant audio was detected
            if (!hasSignificantAudio) {
                updateVOXStatus('No significant audio detected - Ready');
                voxIsListening = false;
                const recordBtn = document.getElementById('vox-record-btn');
                if (recordBtn) {
                    recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
                    recordBtn.classList.remove('recording');
                }
                return;
            }

            // Convert WebM to WAV using Web Audio API (NO FFMPEG)
            try {
                const audioBlob = new Blob(voxAudioChunks, { type: 'audio/webm' });
                const arrayBuffer = await audioBlob.arrayBuffer();

                // Decode audio
                const newAudioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await newAudioContext.decodeAudioData(arrayBuffer);

                // Convert to WAV
                const wavBlob = await audioBufferToWav(audioBuffer);
                await transcribeAndRespond(wavBlob);

                newAudioContext.close();
            } catch (error) {
                console.error('[VOX] Audio conversion error:', error);
                await transcribeAndRespond(new Blob(voxAudioChunks, { type: 'audio/webm' }));
            }
        };
        
        voxMediaRecorder.start();
        console.log('[VOX] MediaRecorder started');
        
        // Start continuous transcription for interrupt detection
        startContinuousTranscription();
        
        // Update UI
        const recordBtn = document.getElementById('vox-record-btn');
        if (recordBtn) {
            recordBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            recordBtn.classList.add('recording');
            console.log('[VOX] UI updated to show recording state');
        }
        
    } catch (error) {
        console.error('[VOX] Error starting recording:', error);
        updateVOXStatus('Error accessing microphone');
        voxIsListening = false;
    }
}

// Stop voice recording
function stopVOXRecording() {
    if (!voxIsListening || !voxMediaRecorder) return;
    
    // Check MediaRecorder state before stopping to prevent errors
    if (voxMediaRecorder.state === 'inactive') {
        console.log('[VOX] MediaRecorder already inactive');
        voxIsListening = false;
        return;
    }
    
    // Stop continuous transcription
    stopContinuousTranscription();
    
    voxIsListening = false;
    voxMediaRecorder.stop();
    
    // Update UI
    const recordBtn = document.getElementById('vox-record-btn');
    if (recordBtn) {
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordBtn.classList.remove('recording');
    }
    
    updateVOXStatus('Processing...');
}

// Transcribe audio and get AI response
async function transcribeAndRespond(audioBlob) {
    console.log('[VOX] transcribeAndRespond called, audioBlob size:', audioBlob.size);
    
    try {
        // Check if we should only process post-TTS audio
        if (window.VOXInterrupt) {
            const audioData = window.VOXInterrupt.getAudio();
            
            // If TTS is still speaking, check for interrupts only
            if (audioData.interruptOnly && audioData.audio) {
                console.log('[VOX] TTS speaking - checking for interrupt words only');
                
                // Quick transcription check for interrupt
                const formData = new FormData();
                const ext = audioData.audio.type.includes('wav') ? 'wav' : 'webm';
                formData.append('audio', audioData.audio, `interrupt-check.${ext}`);
                formData.append('language_mode', voxTranscriptionLanguage || 'auto');
                
                const response = await fetch('/api/transcribe_realtime', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.text) {
                        const isInterrupt = await window.VOXInterrupt.handleInterrupt(data.text);
                        if (isInterrupt) {
                            console.log('[VOX] Interrupt detected, TTS stopped');
                            updateVOXStatus('Interrupted - Ready');
                        }
                    }
                }
                
                // Don't process as user prompt - TTS still speaking
                return;
            }
            
            // Use post-TTS audio if available
            if (audioData.audio && !audioData.interruptOnly) {
                console.log('[VOX] Using post-TTS audio for transcription');
                audioBlob = audioData.audio;
            } else if (!audioData.audio) {
                console.log('[VOX] No post-TTS audio to process');
                updateVOXStatus('Ready');
                return;
            }
        }
        
        // Show loading overlay on first use
        const loadingOverlay = document.getElementById('vox-loading-overlay');
        if (loadingOverlay && !window.whisperModelLoaded) {
            loadingOverlay.style.display = 'flex';
            console.log('[VOX] Showing Whisper loading overlay');
        }
        
        // Create FormData for audio upload
        const formData = new FormData();
        const ext = audioBlob.type.includes('wav') ? 'wav' : 'webm';
        formData.append('audio', audioBlob, `recording.${ext}`);
        formData.append('language_mode', voxTranscriptionLanguage || 'auto');
        
        console.log('[VOX] Sending audio to transcription endpoint...');
        
        // Transcribe using realtime endpoint (no ffmpeg)
        updateVOXStatus('Transcribing...');
        const transcribeResponse = await fetch('/api/transcribe_realtime', {
            method: 'POST',
            body: formData
        });
        
        // Hide loading overlay after first successful transcription
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
            window.whisperModelLoaded = true;
        }
        
        if (!transcribeResponse.ok) {
            throw new Error(`Transcription failed: ${transcribeResponse.status}`);
        }
        
        const transcribeData = await transcribeResponse.json();
        
        if (!transcribeData.text || transcribeData.text.trim() === '') {
            updateVOXStatus('No speech detected - Ready');
            // Reset recording button if in one-shot mode
            const continuousMode = document.getElementById('vox-continuous')?.checked;
            if (!continuousMode) {
                voxIsListening = false;
                const recordBtn = document.getElementById('vox-record-btn');
                if (recordBtn) {
                    recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
                    recordBtn.classList.remove('recording');
                }
            }
            return;
        }
        
        const userText = transcribeData.text;
        addVOXMessage(userText, 'user');
        updateVOXStatus('Thinking...');
        
        // Stream AI response
        const response = await fetch('/api/vox_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: userText,
                session_id: voxSessionId,
                language: voxLanguage
            })
        });
        
        if (!response.ok) {
            throw new Error(`Streaming failed: ${response.status} ${response.statusText}`);
        }
        
        console.log('[VOX] Starting SSE stream reading...');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let accumulatedResponse = '';
        const botMsgId = addVOXMessage('', 'bot');
        let chunkCount = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                console.log('[VOX] SSE stream completed, total chunks:', chunkCount);
                break;
            }
            
            chunkCount++;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.session_id) {
                            voxSessionId = data.session_id;
                            console.log('[VOX] Received session_id:', data.session_id);
                        }
                        
                        if (data.content) {
                            accumulatedResponse += data.content;
                            updateVOXMessage(botMsgId, accumulatedResponse);
                        }
                        
                        if (data.done && data.speak) {
                            console.log('[VOX] Received done signal, accumulated response length:', accumulatedResponse.length);
                            // Update session_id if returned (with vox_ prefix)
                            if (data.session_id) {
                                voxSessionId = data.session_id;
                                console.log('[VOX] Final session_id:', voxSessionId);
                            }
                            
                            // Speak the response
                            updateVOXStatus('Speaking...');
                            console.log('[VOX] Starting TTS...');
                            await speakText(accumulatedResponse, data.language || voxLanguage);
                            console.log('[VOX] TTS completed');
                            updateVOXStatus('Ready');
                            
                            // Ensure button is in correct state
                            voxIsListening = false;
                            const recordBtn = document.getElementById('vox-record-btn');
                            if (recordBtn) {
                                recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
                                recordBtn.classList.remove('recording');
                            }
                            
                            // Auto-restart listening if continuous mode enabled AND not explicitly stopped
                            if (document.getElementById('vox-continuous')?.checked && !voxShouldStop) {
                                console.log('[VOX] Continuous mode enabled, restarting in 500ms...');
                                setTimeout(() => {
                                    // Double-check the flag before actually restarting
                                    if (!voxShouldStop) {
                                        startVOXRecording();
                                    } else {
                                        console.log('[VOX] Stop flag set, aborting continuous restart');
                                    }
                                }, 500);
                            }
                        }
                        
                        if (data.error) {
                            console.error('[VOX] Server error:', data.error);
                            updateVOXStatus(`Error: ${data.error}`);
                            throw new Error(data.error);
                        }
                        
                    } catch (e) {
                        console.error('[VOX] Error parsing SSE data:', e, 'Line:', line);
                    }
                }
            }
        }
        
        // If we got here with no accumulated response, something went wrong
        if (accumulatedResponse.length === 0) {
            console.warn('[VOX] No response received from stream');
            updateVOXStatus('No response - Ready');
        }
        
    } catch (error) {
        console.error('[VOX] Error in voice interaction:', error);
        updateVOXStatus('Error - Ready');
        
        // Reset recording button state
        voxIsListening = false;
        const recordBtn = document.getElementById('vox-record-btn');
        if (recordBtn) {
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordBtn.classList.remove('recording');
        }
        
        // Hide loading overlay if visible
        const loadingOverlay = document.getElementById('vox-loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
        
        // Do NOT restart listening - user must click button again
    }
}

// Continuous transcription during recording (for interrupt detection)
let continuousTranscriptionInterval = null;

async function startContinuousTranscription() {
    if (!window.VOXInterrupt) {
        console.warn('[VOX] VOXInterrupt module not available, skipping continuous transcription');
        return;
    }
    
    // Stop any existing interval
    if (continuousTranscriptionInterval) {
        clearInterval(continuousTranscriptionInterval);
    }
    
    // Check every 2 seconds during recording
    continuousTranscriptionInterval = setInterval(async () => {
        if (!voxIsListening || !voxMediaRecorder || voxMediaRecorder.state !== 'recording') {
            return;
        }
        
        // Only check for interrupts if TTS is speaking
        if (!window.VOXInterrupt.isSpeaking()) {
            return;
        }
        
        console.log('[VOX] Continuous check - TTS is speaking, checking for interrupts');
        
        // Create a temporary chunk for interrupt detection
        if (voxAudioChunks.length > 0) {
            const recentChunk = new Blob([voxAudioChunks[voxAudioChunks.length - 1]], { type: 'audio/webm' });
            
            try {
                const formData = new FormData();
                formData.append('audio', recentChunk, 'interrupt-check.webm');
                formData.append('language', voxLanguage || 'en');
                
                const response = await fetch('/api/transcribe_realtime', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.text) {
                        const isInterrupt = await window.VOXInterrupt.handleInterrupt(data.text);
                        if (isInterrupt) {
                            console.log('[VOX] Interrupt detected during continuous monitoring');
                            updateVOXStatus('Interrupted - Listening...');
                        }
                    }
                }
            } catch (error) {
                console.error('[VOX] Error in continuous transcription:', error);
            }
        }
    }, 2000); // Check every 2 seconds
}

function stopContinuousTranscription() {
    if (continuousTranscriptionInterval) {
        clearInterval(continuousTranscriptionInterval);
        continuousTranscriptionInterval = null;
        console.log('[VOX] Stopped continuous transcription');
    }
}

// ============================================================
// Streaming TTS Audio Queue
// ============================================================
// Audio chunks arrive from the server one-by-one via SSE.
// Each chunk is decoded and queued.  A playback loop drains the
// queue, playing chunks back-to-back with no gap.  If the user
// presses "Stop Speaking" or "Stop All", the queue is flushed,
// the current Audio element is stopped, and the SSE fetch is
// aborted so the server stops generating.

let voxTtsAbortController = null;  // AbortController for the SSE fetch
let voxAudioQueue = [];            // Array of {audio: Audio, url: string} waiting to play
let voxQueuePlaying = false;       // True while the queue drain loop is active
let voxTtsCancelled = false;       // Set true by stop functions to break loops

/**
 * Drain the audio queue: plays items one-by-one until the queue is
 * empty or playback is cancelled.  Called once when the first chunk
 * arrives; subsequent chunks are picked up automatically.
 */
async function _drainAudioQueue() {
    if (voxQueuePlaying) return;   // Already draining
    voxQueuePlaying = true;

    try {
        while (voxAudioQueue.length > 0 && !voxTtsCancelled) {
            const item = voxAudioQueue.shift();
            if (!item) continue;

            const { audio, url } = item;
            voxCurrentAudio = audio;
            voxActiveAudioElements.push(audio);

            try {
                // Play this chunk and wait for it to finish
                await new Promise((resolve, reject) => {
                    audio.onended = () => resolve();
                    audio.onerror = (e) => reject(e);
                    audio.play().catch(reject);
                });
            } catch (e) {
                // Playback error or abort - keep going to next chunk unless cancelled
                if (voxTtsCancelled) break;
                console.warn('[VOX] Chunk playback error:', e);
            } finally {
                // Clean up this chunk
                URL.revokeObjectURL(url);
                const idx = voxActiveAudioElements.indexOf(audio);
                if (idx > -1) voxActiveAudioElements.splice(idx, 1);
                if (voxCurrentAudio === audio) voxCurrentAudio = null;
            }
        }
    } finally {
        voxQueuePlaying = false;
    }
}

/**
 * Flush the audio queue and stop any currently-playing chunk.
 * Called by stopVOXSpeaking() and stopAllVOX().
 */
function _flushAudioQueue() {
    // Mark cancelled so drain loop exits
    voxTtsCancelled = true;

    // Abort the SSE fetch so the server stops generating
    if (voxTtsAbortController) {
        try { voxTtsAbortController.abort(); } catch (e) {}
        voxTtsAbortController = null;
    }

    // Stop currently playing audio
    if (voxCurrentAudio) {
        try { voxCurrentAudio.pause(); voxCurrentAudio.currentTime = 0; } catch (e) {}
        voxCurrentAudio = null;
    }

    // Stop ALL tracked audio elements (safety net)
    voxActiveAudioElements.forEach(a => {
        try { a.pause(); a.currentTime = 0; } catch (e) {}
    });
    voxActiveAudioElements = [];

    // Revoke URLs and clear the queue
    voxAudioQueue.forEach(item => {
        try { URL.revokeObjectURL(item.url); } catch (e) {}
    });
    voxAudioQueue = [];
    voxQueuePlaying = false;
}

// Speak text using TTS - streams audio chunks from the server and plays them
// back through a queue for near-instant perceived playback.
async function speakText(text, language = 'en') {
    if (!text) return;

    // Reload TTS if it was manually unloaded
    if (window.ttsManuallyUnloaded) {
        console.log('[VOX] TTS was unloaded, reloading...');
        window.ttsManuallyUnloaded = false;
        await loadTTSModels(language);
    }

    // CRITICAL: If TTS is already speaking, stop it first to prevent overlapping audio
    // This ensures only ONE TTS stream is active at a time
    if (voxIsSpeaking || voxTtsAbortController || voxQueuePlaying) {
        console.log('[VOX] Stopping previous TTS before starting new one');
        _flushAudioQueue();
        // Wait a bit for the previous stream to fully abort
        await new Promise(r => setTimeout(r, 50));
    }

    voxIsSpeaking = true;
    voxTtsCancelled = false;

    // CRITICAL: Pause recording while speaking to prevent echo/feedback
    const wasRecording = voxIsListening;
    if (wasRecording && voxMediaRecorder && voxMediaRecorder.state === 'recording') {
        console.log('[VOX] Pausing recording during TTS to prevent echo');
        voxMediaRecorder.pause();
    }

    try {
        // Create an AbortController so stop buttons can kill the fetch
        voxTtsAbortController = new AbortController();

        const response = await fetch('/api/vox_speak_wav_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language }),
            signal: voxTtsAbortController.signal
        });

        if (!response.ok) throw new Error(`TTS stream request failed: ${response.status}`);

        // Read SSE events from the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            if (voxTtsCancelled) break;

            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // Keep the last (possibly incomplete) line in the buffer
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.error) {
                        console.error('[VOX] TTS stream error:', data.error);
                        continue;
                    }

                    if (data.done) {
                        console.log('[VOX] TTS stream complete');
                        continue;
                    }

                    if (data.audio_b64) {
                        // Decode base64 WAV into a Blob and create an Audio element
                        const binaryStr = atob(data.audio_b64);
                        const bytes = new Uint8Array(binaryStr.length);
                        for (let i = 0; i < binaryStr.length; i++) {
                            bytes[i] = binaryStr.charCodeAt(i);
                        }
                        const blob = new Blob([bytes], { type: 'audio/wav' });
                        const url = URL.createObjectURL(blob);
                        const audio = new Audio(url);

                        // Route to selected output device if supported
                        if (voxSelectedOutputDevice && typeof audio.setSinkId === 'function') {
                            try { await audio.setSinkId(voxSelectedOutputDevice); } catch (e) {}
                        }

                        // Push to queue
                        voxAudioQueue.push({ audio, url });
                        console.log(`[VOX] Queued TTS chunk ${data.chunk + 1}/${data.total}`);

                        // Start draining the queue as soon as the first chunk arrives
                        // (_drainAudioQueue is a no-op if already running)
                        _drainAudioQueue();
                    }
                } catch (parseErr) {
                    // Ignore unparseable lines (SSE comments, blank lines)
                }
            }
        }

        // All chunks received - wait for the queue to finish playing
        // Poll until drain loop finishes (or cancelled)
        while (voxQueuePlaying && !voxTtsCancelled) {
            await new Promise(r => setTimeout(r, 100));
        }

        console.log('[VOX] TTS playback finished');
    } catch (error) {
        // AbortError is expected when the user presses stop
        if (error.name !== 'AbortError') {
            console.error('[VOX] Error in streaming TTS:', error);
        }
    } finally {
        voxIsSpeaking = false;
        voxTtsAbortController = null;

        // Resume recording after TTS finishes
        if (wasRecording && voxMediaRecorder && voxMediaRecorder.state === 'paused') {
            console.log('[VOX] Resuming recording after TTS');
            voxMediaRecorder.resume();
        }
    }
}

// Stop TTS - flushes the audio queue, aborts the SSE stream, and tells the
// backend to cancel any in-progress generation so the GPU stops immediately.
async function stopVOXSpeaking() {
    try {
        // Flush the browser-side audio queue and abort the fetch
        _flushAudioQueue();

        // Tell backend to set its stop_event flag (cancels generate loop)
        await fetch('/api/vox_stop', { method: 'POST' });

        voxIsSpeaking = false;
        updateVOXStatus('Stopped');
        console.log('[VOX] TTS stopped (queue flushed + backend cancelled)');
    } catch (error) {
        console.error('Error stopping TTS:', error);
    }
}

// Add message to VOX chat
function addVOXMessage(text, type) {
    const chatWindow = document.getElementById('vox-chat-window');
    if (!chatWindow) return null;
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `vox-message vox-${type}`;
    const msgId = `vox-msg-${Date.now()}`;
    msgDiv.id = msgId;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'vox-message-content';
    contentDiv.textContent = text;
    
    msgDiv.appendChild(contentDiv);
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return msgId;
}

// Update VOX message
function updateVOXMessage(msgId, text) {
    const msgDiv = document.getElementById(msgId);
    if (!msgDiv) return;
    
    const contentDiv = msgDiv.querySelector('.vox-message-content');
    if (contentDiv) {
        contentDiv.textContent = text;
    }
    
    const chatWindow = document.getElementById('vox-chat-window');
    if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

// Update VOX status
function updateVOXStatus(status) {
    const statusEl = document.getElementById('vox-status');
    if (statusEl) {
        statusEl.textContent = status;
    }
}

// Clear VOX chat - clears GUI and backend message history
async function clearVOXChat() {
    const chatWindow = document.getElementById('vox-chat-window');
    if (chatWindow) chatWindow.innerHTML = '';

    if (voxSessionId) {
        try {
            await fetch(`/api/chats/${voxSessionId}/messages`, { method: 'DELETE' });
        } catch(e) {}
    }
    voxSessionId = null;
    updateVOXStatus('Ready');
}

// Convert AudioBuffer to WAV Blob (NO FFMPEG)
function audioBufferToWav(audioBuffer) {
    const numberOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    // Interleave channels
    let length = audioBuffer.length * numberOfChannels * 2;
    let buffer = new ArrayBuffer(44 + length);
    let view = new DataView(buffer);
    
    // Write WAV header
    let offset = 0;
    const writeString = (str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset++, str.charCodeAt(i));
        }
    };
    
    writeString('RIFF');
    view.setUint32(offset, 36 + length, true); offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true); offset += 4;
    view.setUint16(offset, format, true); offset += 2;
    view.setUint16(offset, numberOfChannels, true); offset += 2;
    view.setUint32(offset, sampleRate, true); offset += 4;
    view.setUint32(offset, sampleRate * numberOfChannels * bitDepth / 8, true); offset += 4;
    view.setUint16(offset, numberOfChannels * bitDepth / 8, true); offset += 2;
    view.setUint16(offset, bitDepth, true); offset += 2;
    writeString('data');
    view.setUint32(offset, length, true); offset += 4;
    
    // Write interleaved PCM samples
    for (let i = 0; i < audioBuffer.length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            let sample = audioBuffer.getChannelData(channel)[i];
            sample = Math.max(-1, Math.min(1, sample));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, sample, true);
            offset += 2;
        }
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

// Voice Chat History Management
let allVoiceChats = [];

async function loadVoiceChatList() {
    try {
        const response = await fetch('/api/voice_chats');
        allVoiceChats = await response.json();
        
        const select = document.getElementById('vox-chat-list');
        if (!select) return;
        
        select.innerHTML = '<option value="">New Voice Chat</option>';
        
        allVoiceChats.forEach(chat => {
            const option = document.createElement('option');
            option.value = chat.id;
            option.textContent = chat.title || 'Untitled Voice Chat';
            option.dataset.messages = JSON.stringify(chat.messages || []);
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading voice chat list:', error);
    }
}

function filterVoiceChats(searchQuery) {
    const select = document.getElementById('vox-chat-list');
    if (!select) return;
    
    const query = searchQuery.toLowerCase().trim();
    select.innerHTML = '<option value="">New Voice Chat</option>';
    
    if (!query) {
        // Show all chats if no search query
        allVoiceChats.forEach(chat => {
            const option = document.createElement('option');
            option.value = chat.id;
            option.textContent = chat.title || 'Untitled Voice Chat';
            option.dataset.messages = JSON.stringify(chat.messages || []);
            select.appendChild(option);
        });
        return;
    }
    
    // Split search query into keywords
    const keywords = query.split(/\s+/).filter(k => k.length > 0);
    
    // Filter chats by keywords (fuzzy matching)
    const filteredChats = allVoiceChats.filter(chat => {
        const title = (chat.title || '').toLowerCase();
        const messages = chat.messages || [];
        
        // Check if any keyword matches title or message content
        return keywords.some(keyword => {
            // Title match
            if (title.includes(keyword)) return true;
            
            // Message content match
            return messages.some(msg => {
                const content = (msg.content || '').toLowerCase();
                return content.includes(keyword);
            });
        });
    });
    
    // Add filtered chats to dropdown
    filteredChats.forEach(chat => {
        const option = document.createElement('option');
        option.value = chat.id;
        option.textContent = chat.title || 'Untitled Voice Chat';
        option.dataset.messages = JSON.stringify(chat.messages || []);
        select.appendChild(option);
    });
    
    // Show message if no results
    if (filteredChats.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No matching chats found';
        option.disabled = true;
        select.appendChild(option);
    }
}

async function loadVoiceChat(chatId) {
    if (!chatId) {
        // New chat - clear everything
        voxSessionId = null;
        clearVOXChat();
        return;
    }
    
    try {
        const response = await fetch(`/api/chats/${chatId}`);
        const chat = await response.json();
        
        voxSessionId = chatId;
        clearVOXChat();
        
        // Load messages into chat window
        chat.messages.forEach(msg => {
            if (msg.role === 'user') {
                addVOXMessage(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                addVOXMessage(msg.content, 'bot');
            }
        });
        
        console.log('[VOX] Loaded voice chat:', chat.title);
    } catch (error) {
        console.error('Error loading voice chat:', error);
        alert('Failed to load voice chat');
    }
}

async function deleteCurrentVoiceChat() {
    const select = document.getElementById('vox-chat-list');
    const chatId = select.value;
    
    if (!chatId) {
        alert('No voice chat selected');
        return;
    }
    
    const selectedOption = select.options[select.selectedIndex];
    const chatTitle = selectedOption.textContent || 'this voice chat';
    
    // Show confirmation modal
    if (typeof showConfirmModal !== 'undefined') {
        showConfirmModal(
            'Delete Voice Chat',
            `Are you sure you want to delete "${chatTitle}"? This cannot be undone.`,
            async () => {
                try {
                    const response = await fetch(`/api/chats/${chatId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        await loadVoiceChatList();
                        if (chatId === voxSessionId) {
                            clearVOXChat();
                            voxSessionId = null;
                        }
                        console.log('[VOX] Deleted voice chat:', chatId);
                    } else {
                        alert('Failed to delete voice chat');
                    }
                } catch (error) {
                    console.error('Error deleting voice chat:', error);
                    alert('Error deleting voice chat');
                }
            }
        );
    } else {
        console.error('showConfirmModal is not defined');
        alert('Cannot delete: confirmation modal not available');
    }
}

// Stop all active VOX operations without unloading models.
// Flushes the audio queue, aborts any in-flight TTS SSE stream,
// stops recording, and tells the backend to cancel generation.
async function stopAllVOX() {
    console.log('[VOX] Stopping all operations...');

    // Prevent continuous mode from restarting
    voxShouldStop = true;

    // Stop recording
    if (voxMediaRecorder && voxMediaRecorder.state !== 'inactive') {
        try { voxMediaRecorder.stop(); } catch(e) {}
    }
    voxIsListening = false;

    // Flush the streaming TTS audio queue + abort SSE fetch
    _flushAudioQueue();

    voxIsSpeaking = false;

    // Stop continuous transcription interval
    stopContinuousTranscription();

    // Clear silence timeout
    if (voxSilenceTimeout) {
        clearTimeout(voxSilenceTimeout);
        voxSilenceTimeout = null;
    }

    // Tell backend to stop ALL inference (LLM + TTS)
    try {
        await fetch('/api/stop_all_vox', { method: 'POST' });
    } catch(e) {}

    // Reset record button UI
    const recordBtn = document.getElementById('vox-record-btn');
    if (recordBtn) {
        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordBtn.classList.remove('recording');
    }

    updateVOXStatus('Stopped');
    console.log('[VOX] All operations stopped (queue flushed + backend cancelled)');
}

// Full teardown: stop everything AND unload STT/TTS/speech LLM from VRAM
async function teardownVOX() {
    console.log('[VOX] Tearing down VOX...');

    await stopAllVOX();

    // Unload STT with proper await and logging
    try {
        console.log('[VOX] Unloading STT...');
        const sttResponse = await fetch('/api/unload_stt', { method: 'POST' });
        const sttResult = await sttResponse.json();
        console.log('[VOX] STT unload result:', sttResult);
        window.sttManuallyUnloaded = true;
        window.whisperModelLoaded = false;
        whisperLoaded = false;
    } catch(e) {
        console.error('[VOX] STT unload error:', e);
    }

    // Unload TTS with proper await and logging
    try {
        console.log('[VOX] Unloading TTS...');
        const ttsResponse = await fetch('/api/unload_tts', { method: 'POST' });
        const ttsResult = await ttsResponse.json();
        console.log('[VOX] TTS unload result:', ttsResult);
        window.ttsManuallyUnloaded = true;
    } catch(e) {
        console.error('[VOX] TTS unload error:', e);
    }

    // Unload VOX Ollama chat model from VRAM
    try {
        console.log('[VOX] Unloading VOX Ollama model...');
        const settings = await fetch('/api/vox_model').then(r => r.json());
        const voxModel = settings.model;
        if (voxModel) {
            const modelResponse = await fetch('/api/unload_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: voxModel })
            });
            const modelResult = await modelResponse.json();
            console.log(`[VOX] Unloaded VOX model ${voxModel}:`, modelResult);
        }
    } catch(e) {
        console.error('[VOX] Could not unload VOX model:', e);
    }

    console.log('[VOX] Teardown complete');
}

// Unload STT (Whisper) - stops listening first
async function unloadSTT() {
    try {
        // Stop any active recording
        if (voxMediaRecorder && voxMediaRecorder.state !== 'inactive') {
            try { voxMediaRecorder.stop(); } catch(e) {}
        }
        voxIsListening = false;
        voxShouldStop = true;
        const recordBtn = document.getElementById('vox-record-btn');
        if (recordBtn) {
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordBtn.classList.remove('recording');
        }

        updateVOXStatus('Unloading STT...');
        const response = await fetch('/api/unload_stt', { method: 'POST' });
        const result = await response.json();
        if (result.success) {
            window.sttManuallyUnloaded = true;
            window.whisperModelLoaded = false;
            whisperLoaded = false;
            updateVOXStatus('STT unloaded');
        } else {
            updateVOXStatus('STT unload error');
        }
    } catch (error) {
        console.error('[STT] Unload error:', error);
        updateVOXStatus('STT unload failed');
    }
}

// Unload TTS - stops/cancels any queued or playing audio first
async function unloadTTS() {
    try {
        // Stop any playing audio immediately
        if (voxCurrentAudio) {
            try { voxCurrentAudio.pause(); } catch(e) {}
            voxCurrentAudio = null;
        }
        voxIsSpeaking = false;

        updateVOXStatus('Unloading TTS...');
        console.log('[TTS] Sending unload request...');

        const response = await fetch('/api/unload_tts', { method: 'POST' });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[TTS] Unload response:', result);

        if (result.success) {
            window.ttsManuallyUnloaded = true;
            updateVOXStatus('TTS unloaded - Ready');
            console.log('[TTS] Unload successful');
        } else {
            updateVOXStatus(`TTS unload error: ${result.error || 'Unknown'}`);
            console.error('[TTS] Unload failed:', result);
        }
    } catch (error) {
        console.error('[TTS] Unload error:', error);
        updateVOXStatus(`TTS unload failed: ${error.message}`);
    }
}

// Export functions for global use IMMEDIATELY (before DOMContentLoaded)
if (typeof window.VOX === 'undefined') {
    window.VOX = {
        init: initVOX,
        startRecording: startVOXRecording,
        stopRecording: stopVOXRecording,
        stopSpeaking: stopVOXSpeaking,
        stopAll: stopAllVOX,
        clearChat: clearVOXChat,
        setLanguage: (lang) => { voxLanguage = lang; },
        setTranscriptionLanguage: (lang) => {
            voxTranscriptionLanguage = lang;
            console.log(`[VOX] Transcription language set to: ${lang}`);
        },
        setAudioDevice: setAudioDevice,
        loadTTSModels: loadTTSModels  // Expose for tab switching preload
    };
    console.log('VOX module loaded and exported to window.VOX');
} else {
    console.warn('VOX already defined - skipping export');
}

// Export voice chat functions globally
window.loadVoiceChatList = loadVoiceChatList;
window.filterVoiceChats = filterVoiceChats;
window.loadVoiceChat = loadVoiceChat;
window.deleteCurrentVoiceChat = deleteCurrentVoiceChat;
window.unloadSTT = unloadSTT;
window.unloadTTS = unloadTTS;
window.stopAllVOX = stopAllVOX;
window.teardownVOX = teardownVOX;

// Export device functions globally for HTML onchange handlers
window.setAudioDevice = setAudioDevice;
window.setOutputDevice = setOutputDevice;
window.setVOXModel = setVOXModel;
