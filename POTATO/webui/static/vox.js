// VOX - Voice Interaction Module
// Note: voxLanguage is defined in script.js and shared globally
let voxSessionId = null;
let voxIsListening = false;
let voxIsSpeaking = false;
let voxAudioContext = null;
let voxMediaRecorder = null;
let voxAudioChunks = [];
let voxSelectedDevice = null;
let voxCurrentAudio = null;
let whisperLoaded = false;
let whisperLoading = false;
let voxSilenceTimeout = null;
let voxLastAudioLevel = 0;

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
        // Load Whisper model first with progress
        const whisperOk = await loadWhisperModel();
        if (!whisperOk) return;
        
        // Load TTS models based on selected language
        const language = document.getElementById('vox-language')?.value || 'en';
        await loadTTSModels(language);
        
        // Load audio devices
        await loadAudioDevices();
        
        // Load voice chat history
        await loadVoiceChatList();
        
        // Initialize audio context
        voxAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        console.log('VOX initialized');
    } catch (error) {
        console.error('Error initializing VOX:', error);
    }
}

// Load available audio devices
async function loadAudioDevices() {
    try {
        const response = await fetch('/api/audio_devices');
        const data = await response.json();
        
        const deviceSelect = document.getElementById('vox-device-select');
        if (deviceSelect) {
            deviceSelect.innerHTML = '';
            
            data.devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.index;
                option.textContent = device.name;
                if (device.index === data.selected_device) {
                    option.selected = true;
                    voxSelectedDevice = device.index;
                }
                deviceSelect.appendChild(option);
            });
            
            // Auto-detect if no device selected
            if (!data.selected_device) {
                await autoDetectDevice();
            }
        }
    } catch (error) {
        console.error('Error loading audio devices:', error);
    }
}

// Auto-detect working audio device
async function autoDetectDevice() {
    try {
        updateVOXStatus('Detecting audio device...');
        const response = await fetch('/api/detect_audio_device', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            voxSelectedDevice = data.device.index;
            updateVOXStatus(`Device detected: ${data.device.name}`);
            await loadAudioDevices();
        } else {
            updateVOXStatus('No working device found');
        }
    } catch (error) {
        console.error('Error detecting device:', error);
        updateVOXStatus('Error detecting device');
    }
}

// Set audio device
async function setAudioDevice(deviceIndex, deviceName) {
    try {
        await fetch('/api/set_audio_device', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                device_index: deviceIndex,
                device_name: deviceName
            })
        });
        voxSelectedDevice = deviceIndex;
    } catch (error) {
        console.error('Error setting device:', error);
    }
}

// Start voice recording
async function startVOXRecording() {
    console.log('[VOX] startVOXRecording called, voxIsListening:', voxIsListening);
    
    if (voxIsListening) {
        console.log('[VOX] Already listening, stopping instead');
        stopVOXRecording();
        return;
    }
    
    // PRIORITY: Stop TTS if speaking (user speech takes priority)
    if (voxCurrentAudio) {
        console.log('[VOX] Stopping TTS - user is speaking');
        voxCurrentAudio.pause();
        voxCurrentAudio.currentTime = 0;
        voxCurrentAudio = null;
        // Also stop backend TTS
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
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });
        
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
        const SILENCE_THRESHOLD = 15; // Audio level below this is considered silence
        const SILENCE_DURATION = 1500; // Stop recording after 1.5s of silence
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
            if (average > 20) {
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
            }
        };
        
        voxMediaRecorder.onstop = async () => {
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
        formData.append('language', voxLanguage || 'en');
        
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
                            
                            // Auto-restart listening if continuous mode enabled
                            if (document.getElementById('vox-continuous')?.checked) {
                                console.log('[VOX] Continuous mode enabled, restarting in 500ms...');
                                setTimeout(() => startVOXRecording(), 500);
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

// Speak text using TTS
async function speakText(text, language = 'en') {
    if (!text) return;
    
    voxIsSpeaking = true;
    
    try {
        await fetch('/api/vox_speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });
    } catch (error) {
        console.error('Error speaking:', error);
    } finally {
        voxIsSpeaking = false;
    }
}

// Stop TTS
async function stopVOXSpeaking() {
    try {
        await fetch('/api/vox_stop', { method: 'POST' });
        voxIsSpeaking = false;
        updateVOXStatus('Stopped');
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

// Clear VOX chat
function clearVOXChat() {
    const chatWindow = document.getElementById('vox-chat-window');
    if (chatWindow) {
        chatWindow.innerHTML = '';
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

// Export functions for global use IMMEDIATELY (before DOMContentLoaded)
if (typeof window.VOX === 'undefined') {
    window.VOX = {
        init: initVOX,
        startRecording: startVOXRecording,
        stopRecording: stopVOXRecording,
        stopSpeaking: stopVOXSpeaking,
        clearChat: clearVOXChat,
        setLanguage: (lang) => { voxLanguage = lang; },
        setAudioDevice: setAudioDevice
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

// Also export setAudioDevice globally for HTML onchange handler
window.setAudioDevice = setAudioDevice;
