// P.O.T.A.T.O Web UI JavaScript
// Global State
let currentSessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let currentModel = 'gpt-oss:20b';
let voxLanguage = 'en';
let webSearchEnabled = false;
let stealthMode = false;
let uploadedFiles = [];
let settings = {};
let currentSettingsTab = 'core';
let voxSessionId = null;
let isRecording = false;

// Configure marked.js for markdown rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadPreferences();  // Load saved preferences first
    loadModels();
    loadSettings();
    updateStats();
    loadChatList();
    
    // Update stats every 2 seconds
    setInterval(updateStats, 2000);
    
    // Search chats
    document.querySelector('.search-box input').addEventListener('input', (e) => {
        loadChatList(e.target.value);
    });
    
    // Enter to send message
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Model selector change - unload old model first
    document.getElementById('model-selector').addEventListener('change', async (e) => {
        const newModel = e.target.value;
        if (newModel !== currentModel) {
            await switchModel(newModel);
        }
    });
    
    // VOX language selector change - save preference
    const voxLangSelector = document.getElementById('vox-language');
    if (voxLangSelector) {
        voxLangSelector.addEventListener('change', (e) => {
            voxLanguage = e.target.value;
            savePreferences();
        });
    }
    
    // Hamburger menu for mobile
    document.getElementById('hamburger-left').addEventListener('click', () => {
        document.getElementById('left-sidebar').classList.toggle('mobile-open');
    });
});

// --- PREFERENCES ---
async function loadPreferences() {
    try {
        const response = await fetch('/api/preferences');
        const prefs = await response.json();
        
        if (prefs.selected_model) {
            currentModel = prefs.selected_model;
        }
        if (prefs.vox_language) {
            voxLanguage = prefs.vox_language;
            const langSelector = document.getElementById('vox-language');
            if (langSelector) {
                langSelector.value = voxLanguage;
            }
        }
    } catch (error) {
        console.error('Error loading preferences:', error);
    }
}

async function savePreferences() {
    try {
        await fetch('/api/preferences', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                selected_model: currentModel,
                vox_language: voxLanguage
            })
        });
    } catch (error) {
        console.error('Error saving preferences:', error);
    }
}

// --- MODEL MANAGEMENT ---
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const selector = document.getElementById('model-selector');
        selector.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                selector.appendChild(option);
            });
            
            // Set from preferences if available, otherwise use first model
            if (!currentModel || !data.models.includes(currentModel)) {
                currentModel = data.models[0];
            }
            selector.value = currentModel;
        } else {
            selector.innerHTML = '<option value="gpt-oss:20b">gpt-oss:20b (default)</option>';
        }
        
        updateModelDisplay();
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function updateModelDisplay() {
    document.getElementById('current-model-display').textContent = `Model: ${currentModel}`;
}

async function switchModel(newModel) {
    try {
        updateModelDisplay();
        document.getElementById('current-model-display').textContent = 'Switching models...';
        
        // Stop old model to unload from VRAM
        await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        
        // Wait a moment for unload
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Update current model
        currentModel = newModel;
        updateModelDisplay();
        savePreferences();  // Save the new model choice
        
        console.log(`Switched to model: ${currentModel}`);
    } catch (error) {
        console.error('Error switching model:', error);
        currentModel = newModel; // Still switch even if stop fails
        updateModelDisplay();
    }
}

// --- TAB SWITCHING ---
function switchTab(tabId) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabId + '-view').classList.add('active');
    
    const btnMap = {
        'chat': 0,
        'vox-core': 1,
        'browser': 2,
        'settings': 3
    };
    
    const buttons = document.querySelectorAll('.nav-btn');
    if (btnMap[tabId] !== undefined && buttons[btnMap[tabId]]) {
        buttons[btnMap[tabId]].classList.add('active');
    }
    
    // Auto-start VOX when tab opens
    if (tabId === 'vox-core') {
        preloadWhisper();
        autoStartVoxCore();
    }
    
    // Close mobile sidebar when switching tabs
    closeMobileSidebar();
}

// Mobile menu toggle
function toggleMobileMenu() {
    const sidebar = document.getElementById('left-sidebar');
    sidebar.classList.toggle('mobile-open');
    document.body.classList.toggle('sidebar-open');
}

function closeMobileSidebar() {
    const sidebar = document.getElementById('left-sidebar');
    sidebar.classList.remove('mobile-open');
    document.body.classList.remove('sidebar-open');
}

// --- CHAT FUNCTIONS ---
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    input.value = '';
    input.style.height = 'auto';
    
    // Toggle send/stop buttons
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-inference-btn');
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';
    
    // Add user message to UI
    appendMessage(message, 'user');
    
    // Create bot placeholder
    const botMsgId = 'bot-' + Date.now();
    createBotPlaceholder(botMsgId);
    
    try {
        const response = await fetch('/api/chat_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId,
                model: currentModel,
                web_search: webSearchEnabled,
                rag_enabled: document.getElementById('rag-enable')?.checked || false,
                context_folder: document.getElementById('rag-folder')?.value || '',
                uploaded_files: uploadedFiles
            })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        updateBotMessage(botMsgId, data);
                        
                        if (data.session_id) {
                            currentSessionId = data.session_id;
                        }
                        
                        // Check if stream is done
                        if (data.done) {
                            resetButtonsAfterInference();
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
        
        // Clear uploaded files after successful send
        uploadedFiles = [];
        
        // Ensure buttons reset even if done flag wasn't received
        resetButtonsAfterInference();
        
    } catch (error) {
        console.error('Error sending message:', error);
        updateBotMessage(botMsgId, { error: 'Failed to get response' });
        resetButtonsAfterInference();
    }
}

function resetButtonsAfterInference() {
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-inference-btn');
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
}

// Handle shift+enter for newline, enter alone to send
function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function createBotPlaceholder(id) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg bot';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="model-badge">${currentModel}</div>
        <div class="thinking-section" id="${id}-thinking-section" style="display: none;">
          <div class="thinking-header" onclick="toggleThinking('${id}')">
            <span><i class="fas fa-brain"></i> Thinking & Tools</span>
            <span class="thinking-toggle" id="${id}-thinking-toggle"><i class="fas fa-chevron-down"></i></span>
          </div>
          <div class="thinking-content" id="${id}-thinking-content"></div>
        </div>
        <div id="${id}-content" class="markdown-content">Thinking...</div>
    `;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function toggleThinking(id) {
    const content = document.getElementById(`${id}-thinking-content`);
    const toggle = document.getElementById(`${id}-thinking-toggle`);
    
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

function updateBotMessage(id, data) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    const contentDiv = document.getElementById(`${id}-content`);
    const thinkingSection = document.getElementById(`${id}-thinking-section`);
    const thinkingContent = document.getElementById(`${id}-thinking-content`);
    
    if (data.thinking) {
        // Show thinking section
        thinkingSection.style.display = 'block';
        const thinkingP = document.createElement('p');
        thinkingP.textContent = data.thinking;
        thinkingContent.appendChild(thinkingP);
    }
    
    if (data.tool_status) {
        // Show tool being called
        thinkingSection.style.display = 'block';
        const toolDiv = document.createElement('div');
        toolDiv.className = 'tool-info';
        toolDiv.innerHTML = `<i class="fas fa-tools"></i> ${data.tool_status}`;
        thinkingContent.appendChild(toolDiv);
    }
    
    if (data.tool_result) {
        // Show tool result
        const resultDiv = document.createElement('div');
        resultDiv.className = 'tool-info';
        resultDiv.innerHTML = `<i class="fas fa-check-circle"></i> Result: ${data.tool_result}`;
        thinkingContent.appendChild(resultDiv);
    }
    
    if (data.content) {
        // Auto-collapse thinking section when real content starts
        if (contentDiv.textContent === 'Thinking...') {
            contentDiv.innerHTML = '';
            contentDiv.dataset.rawContent = '';
            
            // Collapse thinking section
            if (thinkingSection.style.display === 'block') {
                const thinkingContentEl = document.getElementById(`${id}-thinking-content`);
                if (thinkingContentEl && !thinkingContentEl.classList.contains('collapsed')) {
                    thinkingContentEl.classList.add('collapsed');
                    const toggle = document.getElementById(`${id}-thinking-toggle`);
                    if (toggle) toggle.classList.add('collapsed');
                }
            }
        }
        
        // Accumulate raw content
        if (!contentDiv.dataset.rawContent) {
            contentDiv.dataset.rawContent = '';
        }
        contentDiv.dataset.rawContent += data.content;
        
        // Render as markdown
        renderMarkdown(contentDiv, contentDiv.dataset.rawContent);
    }
    
    if (data.error) {
        contentDiv.textContent = `Error: ${data.error}`;
        contentDiv.style.color = 'var(--accent-color)';
    }
    
    // Clear placeholder when done
    if (data.done) {
        if (contentDiv.textContent === 'Thinking...') {
            contentDiv.textContent = 'No response received.';
        }
    }
    
    const chatWindow = document.getElementById('chat-window');
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function renderMarkdown(element, text) {
    if (typeof marked === 'undefined') {
        element.textContent = text;
        return;
    }
    
    element.innerHTML = marked.parse(text);
    
    // Add copy buttons to code blocks
    element.querySelectorAll('pre code').forEach((block) => {
        if (!block.parentElement.querySelector('.copy-btn')) {
            const button = document.createElement('button');
            button.className = 'copy-btn';
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.title = 'Copy code';
            button.onclick = () => copyCode(button, block);
            block.parentElement.style.position = 'relative';
            block.parentElement.insertBefore(button, block);
        }
    });
}

function copyCode(button, codeBlock) {
    const code = codeBlock.textContent;
    navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.style.color = 'var(--primary-color)';
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.style.color = '';
        }, 2000);
    });
}

function appendMessage(text, type) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${type}`;
    
    if (type === 'bot' || type === 'assistant') {
        msgDiv.classList.add('markdown-content');
        renderMarkdown(msgDiv, text);
    } else {
        msgDiv.textContent = text;
    }
    
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function stopInference() {
    try {
        await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        appendMessage('Inference stopped', 'system');
    } catch (error) {
        console.error('Error stopping inference:', error);
    }
}

async function unloadChatModel() {
    try {
        document.getElementById('current-model-display').textContent = 'Unloading model...';
        
        const response = await fetch('/api/unload_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        
        const result = await response.json();
        
        if (result.success) {
            appendMessage('Model unloaded from VRAM', 'system');
        } else {
            appendMessage(`Error unloading: ${result.error}`, 'system');
        }
        
        updateModelDisplay();
    } catch (error) {
        console.error('Error unloading model:', error);
        appendMessage('Error unloading model', 'system');
        updateModelDisplay();
    }
}

async function unloadVoxModels() {
    try {
        // Stop any active VOX listening first
        stopVoxListening();
        
        document.getElementById('vox-status').textContent = 'Unloading models...';
        
        const response = await fetch('/api/unload_vox', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('vox-status').textContent = 'Models unloaded';
            console.log('VOX models unloaded successfully');
        } else {
            document.getElementById('vox-status').textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        console.error('Error unloading VOX models:', error);
        document.getElementById('vox-status').textContent = 'Error unloading models';
    }
}

// --- CHAT SESSION MANAGEMENT ---
async function loadChatList(query = '') {
    try {
        const url = query ? `/api/chats?search=${encodeURIComponent(query)}` : '/api/chats';
        const response = await fetch(url);
        const chats = await response.json();
        
        const list = document.getElementById('chat-list');
        list.innerHTML = '';
        
        chats.forEach(chat => {
            const li = document.createElement('li');
            li.textContent = chat.title;
            li.onclick = () => loadSession(chat.id);
            if (chat.id === currentSessionId) {
                li.classList.add('active');
            }
            list.appendChild(li);
        });
    } catch (error) {
        console.error('Error loading chat list:', error);
    }
}

async function loadSession(id) {
    try {
        const response = await fetch(`/api/chats/${id}`);
        const chat = await response.json();
        
        currentSessionId = id;
        
        const chatWindow = document.getElementById('chat-window');
        chatWindow.innerHTML = '';
        
        chat.messages.forEach(msg => {
            if (msg.role !== 'system') {
                appendMessage(msg.content, msg.role === 'user' ? 'user' : 'bot');
            }
        });
        
        loadChatList();
    } catch (error) {
        console.error('Error loading session:', error);
    }
}

function startNewChat() {
    currentSessionId = null;
    document.getElementById('chat-window').innerHTML = '';
    uploadedFiles = [];
    loadChatList();
}

// --- FILE UPLOAD ---
async function uploadFile(input) {
    if (input.files.length === 0) return;
    
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        uploadedFiles.push(data);
        appendMessage(`File uploaded: ${data.filename}`, 'system');
    } catch (error) {
        console.error('Error uploading file:', error);
    }
}

async function uploadFolder(input) {
    if (input.files.length === 0) return;
    
    const files = Array.from(input.files);
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            uploadedFiles.push(data);
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    }
    
    appendMessage(`Folder uploaded: ${files.length} files`, 'system');
}

// --- VOICE RECORDING ---
function toggleVoiceRecording() {
    if (isRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

async function startVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await transcribeAudio(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('voice-overlay').classList.remove('hidden');
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Could not access microphone');
    }
}

function stopVoiceRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        document.getElementById('voice-overlay').classList.add('hidden');
    }
}

async function transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.text) {
            document.getElementById('chat-input').value = data.text;
        }
    } catch (error) {
        console.error('Error transcribing audio:', error);
    }
}

// --- VOX CORE ---
let voxIsListening = false;
let voxListenInterval = null;
let whisperPreloaded = false;

async function preloadWhisper() {
    if (whisperPreloaded) return;
    
    document.getElementById('vox-status').textContent = 'Loading Whisper model...';
    
    try {
        const response = await fetch('/api/transcribe_preload', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            whisperPreloaded = true;
            console.log('Whisper model preloaded successfully');
        }
    } catch (error) {
        console.error('Error preloading Whisper:', error);
    }
}

async function autoStartVoxCore() {
    // Wait a bit for Whisper to load
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    document.getElementById('vox-status').textContent = 'Ready';
    
    // Start listening loop
    if (!voxIsListening) {
        startVoxListening();
    }
}

async function startVoxListening() {
    voxIsListening = true;
    listenCycle();
}

async function listenCycle() {
    if (!voxIsListening) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        document.getElementById('vox-status').textContent = 'LISTENING...';
        document.getElementById('waveform').classList.add('active');
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await processVoxAudio(audioBlob);
        };
        
        mediaRecorder.start();
        
        // Auto-stop after 5 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 5000);
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        document.getElementById('vox-status').textContent = 'Microphone error - check permissions';
        voxIsListening = false;
    }
}

function stopVoxListening() {
    voxIsListening = false;
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    document.getElementById('vox-status').textContent = 'Stopped';
    document.getElementById('waveform').classList.remove('active');
}

// Alias for compatibility (old onclick handler)
function stopVoxCore() {
    stopVoxListening();
}

async function processVoxAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'vox_recording.wav');
    
    const language = document.getElementById('vox-language')?.value || 'en';
    
    try {
        document.getElementById('vox-status').textContent = 'TRANSCRIBING...';
        
        // Transcribe
        const transcribeResponse = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        if (!transcribeResponse.ok) {
            throw new Error(`Transcription failed: ${transcribeResponse.status}`);
        }
        
        const transcribeData = await transcribeResponse.json();
        
        if (!transcribeData.text || transcribeData.text.trim() === '') {
            // No speech detected, restart listening
            if (voxIsListening) {
                listenCycle();
            }
            return;
        }
        
        // Add to transcript with fade effect
        addVoxMessage(transcribeData.text, 'user');
        
        // Get response from vox_core endpoint
        document.getElementById('vox-status').textContent = 'THINKING...';
        
        const voxResponse = await fetch('/api/vox_core', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_message: transcribeData.text,
                session_id: voxSessionId,
                model: currentModel,
                language: language
            })
        });
        
        if (!voxResponse.ok) {
            throw new Error(`VOX Core failed: ${voxResponse.status}`);
        }
        
        const voxData = await voxResponse.json();
        
        if (voxData.session_id) {
            voxSessionId = voxData.session_id;
        }
        
        // Add assistant response
        addVoxMessage(voxData.response, 'assistant');
        
        document.getElementById('vox-status').textContent = 'SPEAKING...';
        
        // Wait for TTS to finish (estimate based on text length)
        const estimatedDuration = Math.max(2000, voxData.response.length * 50);
        await new Promise(resolve => setTimeout(resolve, estimatedDuration));
        
        // Continue listening
        if (voxIsListening) {
            listenCycle();
        } else {
            document.getElementById('vox-status').textContent = 'Ready';
        }
        
    } catch (error) {
        console.error('Error processing vox audio:', error);
        document.getElementById('vox-status').textContent = 'Error - Ready';
        
        // Retry listening
        if (voxIsListening) {
            setTimeout(() => listenCycle(), 2000);
        }
    }
}

function addVoxMessage(text, role) {
    const messagesDiv = document.getElementById('vox-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `vox-msg ${role}`;
    msgDiv.textContent = text;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    // Add fade effect after 3 seconds
    setTimeout(() => {
        msgDiv.classList.add('fading');
    }, 3000);
}

// --- SETTINGS ---
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        settings = await response.json();
        
        // Update UI elements
        if (settings.bools) {
            stealthMode = settings.bools.STEALTH_MODE || false;
            updateToggleState('stealth-toggle', stealthMode);
        }
        
        renderSettingsTab(currentSettingsTab);
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

function switchSettingsTab(tab) {
    currentSettingsTab = tab;
    
    // Update tab buttons
    document.querySelectorAll('.settings-tab').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    renderSettingsTab(tab);
}

function renderSettingsTab(tab) {
    const container = document.getElementById('settings-content');
    container.innerHTML = '';
    
    let configSection = {};
    
    switch (tab) {
        case 'core':
            configSection = settings.core_llm_config || {};
            break;
        case 'online':
            configSection = settings.online_search_config || {};
            break;
        case 'rag':
            configSection = settings.rag_config || {};
            break;
        case 'voice':
            configSection = settings.voice_config || {};
            break;
        case 'bools':
            configSection = settings.bools || {};
            break;
    }
    
    Object.keys(configSection).forEach(key => {
        const card = document.createElement('div');
        card.className = 'setting-card';
        
        const value = configSection[key];
        const inputType = typeof value === 'boolean' ? 'checkbox' : 
                         typeof value === 'number' ? 'number' : 'text';
        
        let inputHtml = '';
        if (inputType === 'checkbox') {
            inputHtml = `<input type="checkbox" id="setting-${key}" ${value ? 'checked' : ''} />`;
        } else {
            inputHtml = `<input type="${inputType}" id="setting-${key}" value="${value}" />`;
        }
        
        card.innerHTML = `
            <label>${key.replace(/_/g, ' ')}</label>
            ${inputHtml}
            <div class="setting-description">${getSettingDescription(key)}</div>
        `;
        
        container.appendChild(card);
    });
}

function getSettingDescription(key) {
    const descriptions = {
        'MAIN_MODEL': 'Primary LLM model for reasoning and responses',
        'TEMPERATURE': 'Creativity level (0.0 = deterministic, 1.0 = creative)',
        'MAX_TOKENS': 'Maximum response length',
        'ENABLE_WEB_SEARCH': 'Allow AI to search the web for current information',
        'STEALTH_MODE': 'Fully offline mode - no internet access',
        'USE_RAG': 'Enable Retrieval Augmented Generation',
        'TTS_ENABLE': 'Enable text-to-speech output',
        'STT_ENABLE': 'Enable speech-to-text input'
    };
    
    return descriptions[key] || 'No description available';
}

async function saveSettings() {
    const newSettings = { ...settings };
    
    // Gather all inputs
    const inputs = document.querySelectorAll('[id^="setting-"]');
    inputs.forEach(input => {
        const key = input.id.replace('setting-', '');
        let value = input.type === 'checkbox' ? input.checked : input.value;
        
        if (input.type === 'number') {
            value = parseFloat(value);
        }
        
        // Find which section this belongs to
        if (newSettings.core_llm_config && key in newSettings.core_llm_config) {
            newSettings.core_llm_config[key] = value;
        } else if (newSettings.online_search_config && key in newSettings.online_search_config) {
            newSettings.online_search_config[key] = value;
        } else if (newSettings.rag_config && key in newSettings.rag_config) {
            newSettings.rag_config[key] = value;
        } else if (newSettings.voice_config && key in newSettings.voice_config) {
            newSettings.voice_config[key] = value;
        } else if (newSettings.bools && key in newSettings.bools) {
            newSettings.bools[key] = value;
        }
    });
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newSettings)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            alert('Settings saved successfully');
            settings = newSettings;
        } else {
            alert('Failed to save settings');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        alert('Error saving settings');
    }
}

async function resetSettings() {
    if (!confirm('Reset all settings to defaults?')) return;
    
    try {
        const response = await fetch('/api/settings/reset', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            settings = result.settings;
            renderSettingsTab(currentSettingsTab);
            alert('Settings reset to defaults');
        }
    } catch (error) {
        console.error('Error resetting settings:', error);
    }
}

// --- SYSTEM STATS ---
async function updateStats() {
    try {
        const response = await fetch('/api/system_stats');
        const stats = await response.json();
        
        setBar('cpu-bar', stats.cpu);
        setBar('ram-bar', stats.ram);
        setBar('gpu-bar', stats.gpu);
        setBar('vram-bar', stats.vram);
        
        document.getElementById('cpu-val').textContent = `${stats.cpu.toFixed(0)}%`;
        document.getElementById('ram-val').textContent = `${stats.ram.toFixed(0)}%`;
        document.getElementById('gpu-val').textContent = `${stats.gpu.toFixed(0)}%`;
        document.getElementById('vram-val').textContent = `${stats.vram.toFixed(0)}%`;
        document.getElementById('temp-display').textContent = `${stats.gpu_temp.toFixed(0)}°C`;
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function setBar(id, value) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = `${value}%`;
    }
}

// --- TOGGLES ---
function updateToggleState(id, active) {
    const element = document.getElementById(id);
    if (element) {
        if (active) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }
    }
}

// Web search toggle
document.addEventListener('DOMContentLoaded', () => {
    const webSearchToggle = document.getElementById('web-search-toggle');
    if (webSearchToggle) {
        webSearchToggle.addEventListener('click', () => {
            webSearchEnabled = !webSearchEnabled;
            updateToggleState('web-search-toggle', webSearchEnabled);
        });
    }
    
    const stealthToggle = document.getElementById('stealth-toggle');
    if (stealthToggle) {
        stealthToggle.addEventListener('click', () => {
            stealthMode = !stealthMode;
            updateToggleState('stealth-toggle', stealthMode);
            
            if (stealthMode) {
                webSearchEnabled = false;
                updateToggleState('web-search-toggle', false);
            }
        });
    }
});

// --- RIGHT SIDEBAR (RAG) ---
function toggleRightSidebar() {
    document.getElementById('right-sidebar').classList.toggle('open');
}

function selectRagFolder() {
    // In a real implementation, this would open a folder picker
    const folder = prompt('Enter folder path:');
    if (folder) {
        document.getElementById('rag-folder').value = folder;
    }
}

async function embedFolderContents() {
    const folder = document.getElementById('rag-folder').value;
    if (!folder) {
        alert('Please select a folder first');
        return;
    }
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = 'Embedding folder contents...\n';
    
    // TODO: Implement actual embedding API call
    setTimeout(() => {
        statusLog.textContent += 'Embedding complete!\n';
    }, 2000);
}

function searchRagIndex() {
    const query = prompt('Enter search query:');
    if (!query) return;
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = `Searching for: ${query}...\n`;
    
    // TODO: Implement actual search API call
    setTimeout(() => {
        statusLog.textContent += 'Search complete!\n';
    }, 1000);
}
