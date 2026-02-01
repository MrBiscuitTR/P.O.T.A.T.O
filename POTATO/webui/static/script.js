// P.O.T.A.T.O Web UI JavaScript
// Global State
let currentSessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let currentModel = 'gpt-oss:20b';
let voxLanguage = 'en';  // Shared with vox.js
let webSearchEnabled = false;
let stealthMode = false;
let uploadedFiles = [];
let settings = {};
let currentSettingsTab = 'core';
let configDescriptions = {}; // Descriptions from config.env.txt
let isGenerating = false; // Track if AI is currently generating

// Load config descriptions on page load
async function loadConfigDescriptions() {
    try {
        const response = await fetch('/api/settings/descriptions');
        configDescriptions = await response.json();
    } catch (error) {
        console.error('Error loading config descriptions:', error);
    }
}
let isRecording = false;
let autoScrollEnabled = true;  // Auto-scroll by default
// Note: voxSessionId and other VOX-specific variables are in vox.js

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
    loadConfigDescriptions(); // Load descriptions from config.env.txt
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
            if (!isGenerating) {
                sendMessage();
            }
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
    
    // Auto-scroll detection for chat window
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow) {
        chatWindow.addEventListener('scroll', () => {
            const isAtBottom = chatWindow.scrollHeight - chatWindow.scrollTop <= chatWindow.clientHeight + 50;
            autoScrollEnabled = isAtBottom;
        });
    }
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
    
    // VOX Core now uses new VOX module (vox.js) - no auto-start needed
    // VOX.init() is called on page load in index.html
    
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
    
    if (!message || isGenerating) return;
    
    isGenerating = true;
    input.value = '';
    input.style.height = 'auto';
    input.disabled = true; // Disable input during generation
    
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
                        console.log('SSE data received:', data);  // DEBUG
                        updateBotMessage(botMsgId, data);
                        
                        if (data.session_id) {
                            const wasNewChat = !currentSessionId;
                            currentSessionId = data.session_id;
                            // Refresh chat list immediately when new chat is created
                            if (wasNewChat) {
                                console.log('[CHAT] New session created, refreshing chat list');
                                loadChatList();
                            }
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
    isGenerating = false;
    const input = document.getElementById('chat-input');
    if (input) input.disabled = false;
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
    scrollToBottom();
}

function toggleThinking(id) {
    const content = document.getElementById(`${id}-thinking-content`);
    const toggle = document.getElementById(`${id}-thinking-toggle`);
    
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

function updateBotMessage(id, data) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) {
        console.error(`Message div not found: ${id}`);
        return;
    }
    
    const contentDiv = document.getElementById(`${id}-content`);
    const thinkingSection = document.getElementById(`${id}-thinking-section`);
    const thinkingContent = document.getElementById(`${id}-thinking-content`);
    
    if (!contentDiv || !thinkingSection || !thinkingContent) {
        console.error(`Missing elements for ${id}`, { contentDiv, thinkingSection, thinkingContent });
        return;
    }
    
    // Handle metadata (model tags, settings)
    if (data.metadata) {
        // Store metadata for this message
        if (!msgDiv.dataset.metadata) {
            msgDiv.dataset.metadata = JSON.stringify(data.metadata);
        }
        // Update model badge with current model
        const modelBadge = msgDiv.querySelector('.model-badge');
        if (modelBadge && data.metadata.model) {
            modelBadge.textContent = data.metadata.model;
        }
        return; // Metadata doesn't need further processing
    }
    
    // Update model badge if model data is sent (for streaming)
    if (data.model) {
        const modelBadge = msgDiv.querySelector('.model-badge');
        if (modelBadge) {
            modelBadge.textContent = data.model;
        }
    }
    
    if (data.thinking) {
        console.log('[THINKING] Received thinking data:', data.thinking.substring(0, 50));
        // Show thinking section
        thinkingSection.style.display = 'block';
        
        // Accumulate thinking in a single element instead of creating many <p> tags
        let thinkingText = thinkingContent.querySelector('.thinking-text');
        if (!thinkingText) {
            thinkingText = document.createElement('div');
            thinkingText.className = 'thinking-text';
            thinkingContent.appendChild(thinkingText);
            
            // Add auto-scroll behavior
            let autoScrollEnabled = true;
            let userScrolling = false;
            let scrollTimeout;
            
            thinkingContent.addEventListener('scroll', () => {
                clearTimeout(scrollTimeout);
                userScrolling = true;
                
                // Check if near bottom (within 50px)
                const isNearBottom = thinkingContent.scrollHeight - thinkingContent.scrollTop - thinkingContent.clientHeight < 50;
                autoScrollEnabled = isNearBottom;
                
                // Re-enable auto-scroll after user stops scrolling
                scrollTimeout = setTimeout(() => {
                    userScrolling = false;
                }, 150);
            });
            
            // Store scroll state on element
            thinkingContent._autoScroll = true;
        }
        
        thinkingText.textContent += data.thinking;
        
        // Auto-scroll if enabled and not collapsed
        if (!thinkingContent.classList.contains('collapsed')) {
            const isNearBottom = thinkingContent.scrollHeight - thinkingContent.scrollTop - thinkingContent.clientHeight < 50;
            if (thinkingContent._autoScroll !== false && isNearBottom) {
                setTimeout(() => {
                    thinkingContent.scrollTop = thinkingContent.scrollHeight;
                }, 0);
            }
        }
    }
    
    if (data.tool_status) {
        // Tool status is backend info - do NOT display to user
        // Tools should be handled by the backend silently
        console.log('[Tool]', data.tool_status);
        // Removed display logic - tools are not user-facing
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
        if (contentDiv && contentDiv.dataset.rawContent) {
            renderMarkdown(contentDiv, contentDiv.dataset.rawContent);
        }
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
    if (autoScrollEnabled) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

function scrollToBottom() {
    const chatWindow = document.getElementById('chat-window');
    if (autoScrollEnabled) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

function renderMarkdown(element, text) {
    if (!element) {
        console.error('renderMarkdown: element is null');
        return;
    }
    
    if (typeof marked === 'undefined') {
        element.textContent = text;
        return;
    }
    
    try {
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
    } catch (e) {
        console.error('Error rendering markdown:', e);
        element.textContent = text;
    }
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

function appendMessage(text, type, modelName = null) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${type}`;
    
    const msgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    msgDiv.id = msgId;
    
    if (type === 'bot' || type === 'assistant') {
        // Use provided model name or fall back to currentModel or Unknown
        const displayModel = modelName || currentModel || 'Unknown Model';
        
        // Create structure with thinking section
        msgDiv.innerHTML = `
            <div class="model-badge">${displayModel}</div>
            <div class="thinking-section" id="${msgId}-thinking-section" style="display: none;">
              <div class="thinking-header" onclick="toggleThinking('${msgId}')">
                <span><i class="fas fa-brain"></i> Thinking & Tools</span>
                <span class="thinking-toggle" id="${msgId}-thinking-toggle"><i class="fas fa-chevron-down"></i></span>
              </div>
              <div class="thinking-content" id="${msgId}-thinking-content"></div>
            </div>
            <div class="markdown-content" id="${msgId}-content"></div>
        `;
        
        // Append to DOM first so we can query the element
        chatWindow.appendChild(msgDiv);
        
        // Now render markdown into the element
        const contentDiv = document.getElementById(`${msgId}-content`);
        if (contentDiv) {
            renderMarkdown(contentDiv, text);
        }
    } else {
        msgDiv.textContent = text;
        chatWindow.appendChild(msgDiv);
    }
    
    scrollToBottom();
    
    return msgId; // Return ID for later reference
}

async function stopInference() {
    try {
        // Mark generation as stopped
        isGenerating = false;
        
        // Save partial response before stopping
        const chatWindow = document.getElementById('chat-window');
        const lastBotMsg = chatWindow.querySelector('.msg.bot:last-child');
        if (lastBotMsg && currentSessionId) {
            const contentDiv = lastBotMsg.querySelector('.markdown-content');
            if (contentDiv && contentDiv.textContent.trim()) {
                // Save current state with [STOPPED] marker
                const partialContent = contentDiv.textContent + ' [STOPPED]';
                await fetch('/api/chats/' + currentSessionId + '/save_partial', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        content: partialContent,
                        model: currentModel 
                    })
                });
            }
        }
        
        await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        appendMessage('Inference stopped', 'system');
    } catch (error) {
        console.error('Error stopping inference:', error);
    } finally {
        resetButtonsAfterInference();
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
        
        console.log('[CHAT LIST] Loaded', chats.length, 'chats');
        
        const list = document.getElementById('chat-list');
        list.innerHTML = '';
        
        // Remove old event listener if exists
        const newList = list.cloneNode(false);
        list.parentNode.replaceChild(newList, list);
        
        // Add event delegation for delete icons
        newList.addEventListener('click', (e) => {
            // Check if clicked element or its parent is the delete icon
            const deleteIcon = e.target.closest('.delete-chat-icon');
            if (deleteIcon) {
                e.stopPropagation();
                const chatId = deleteIcon.dataset.chatId;
                const chatTitle = deleteIcon.dataset.chatTitle;
                console.log('[DELETE ICON] Clicked via delegation! Chat ID:', chatId, 'Title:', chatTitle);
                confirmDeleteChat(chatId, chatTitle, false);
            }
        });
        
        chats.forEach(chat => {
            console.log('[CHAT LIST] Creating item for:', chat.title);
            
            const li = document.createElement('li');
            
            const titleSpan = document.createElement('span');
            titleSpan.textContent = chat.title;
            titleSpan.onclick = () => loadSession(chat.id);
            titleSpan.style.flex = '1';
            titleSpan.style.cursor = 'pointer';
            
            const deleteIcon = document.createElement('i');
            deleteIcon.className = 'fas fa-trash delete-chat-icon';
            deleteIcon.title = 'Delete chat';
            deleteIcon.style.color = '#ff4757';
            deleteIcon.style.cursor = 'pointer';
            deleteIcon.style.marginLeft = 'auto';
            deleteIcon.style.padding = '5px';
            // Store data attributes for event delegation
            deleteIcon.dataset.chatId = chat.id;
            deleteIcon.dataset.chatTitle = chat.title;
            
            console.log('[CHAT LIST] Created delete icon with data:', deleteIcon);
            
            li.appendChild(titleSpan);
            li.appendChild(deleteIcon);
            
            if (chat.id === currentSessionId) {
                li.classList.add('active');
            }
            newList.appendChild(li);
        });
        
        console.log('[CHAT LIST] Done rendering', chats.length, 'items with event delegation');
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
                if (msg.role === 'user') {
                    appendMessage(msg.content, 'user');
                } else if (msg.role === 'assistant') {
                    // Skip ONLY internal tool call messages (no content, just tool_calls)
                    if (msg.tool_calls && msg.tool_calls.length > 0 && (!msg.content || msg.content.trim() === '')) {
                        return; // Don't render empty assistant messages with tool_calls
                    }
                    
                    // Skip assistant messages that are ONLY unexecuted tool call JSON
                    if (msg.content && msg.content.trim().match(/^\{"name":\s*"potatool_\w+"/)) {
                        return; // Don't render unexecuted tool call JSON
                    }
                    
                    // If message has content, render it (even if it also has tool_calls)
                    if (!msg.content || msg.content.trim() === '') {
                        return; // Don't render empty messages
                    }
                    
                    // Bot message - create ONE bubble with correct model label
                    const msgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    const msgDiv = document.createElement('div');
                    msgDiv.className = 'msg bot';
                    msgDiv.id = msgId;
                    
                    // Use model from message JSON, NOT currentModel
                    const displayModel = msg.model || 'Unknown Model';
                    
                    // Check if there's thinking/tools metadata
                    const hasThinking = msg._thinking && msg._thinking.trim();
                    const hasTools = msg._tools && msg._tools.trim();
                    const thinkingToolsContent = (hasThinking ? msg._thinking : '') + (hasTools ? msg._tools : '');
                    
                    msgDiv.innerHTML = `
                        <div class="model-badge">${displayModel}</div>
                        ${thinkingToolsContent ? `
                        <div class="thinking-section" id="${msgId}-thinking-section" style="display: block;">
                          <div class="thinking-header" onclick="toggleThinking('${msgId}')">
                            <span><i class="fas fa-brain"></i> Thinking & Tools</span>
                            <span class="thinking-toggle collapsed" id="${msgId}-thinking-toggle"><i class="fas fa-chevron-down"></i></span>
                          </div>
                          <div class="thinking-content collapsed" id="${msgId}-thinking-content">${thinkingToolsContent}</div>
                        </div>
                        ` : ''}
                        <div class="markdown-content" id="${msgId}-content"></div>
                    `;
                    
                    chatWindow.appendChild(msgDiv);
                    
                    // Render markdown content
                    const contentDiv = document.getElementById(`${msgId}-content`);
                    if (contentDiv) {
                        renderMarkdown(contentDiv, msg.content);
                    }
                }
                // Skip 'tool' role messages - they're internal
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
    
    // Remove active class from all chat items
    const chatItems = document.querySelectorAll('#chat-list li');
    chatItems.forEach(item => item.classList.remove('active'));
    
    // Reload chat list to reflect no active session
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
        // Request microphone access with specific constraints
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
            }
        });
        
        // Use WAV format which scipy can handle without ffmpeg (same as VOX Core)
        const options = { mimeType: 'audio/wav' };
        
        // Fallback to webm if wav not supported
        if (!MediaRecorder.isTypeSupported('audio/wav')) {
            console.warn('WAV not supported by browser, using webm');
            options.mimeType = 'audio/webm';
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        console.log('MediaRecorder format:', options.mimeType);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: options.mimeType });
            
            // Convert WebM to WAV using Web Audio API (same as VOX Core - NO FFMPEG)
            try {
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                const wavBlob = audioBufferToWav(audioBuffer);
                await transcribeAudio(wavBlob);
                audioContext.close();
            } catch (error) {
                console.error('Audio conversion error:', error);
                // Fallback to original blob
                await transcribeAudio(audioBlob);
            }
            
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('voice-overlay').classList.remove('hidden');
        console.log('Voice recording started');
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Could not access microphone. Please check permissions.');
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

// Convert AudioBuffer to WAV Blob (NO FFMPEG - same as VOX Core)
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

async function transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    // Get language mode from voice overlay selector
    const languageMode = document.getElementById('voice-language-mode')?.value || 'translate-en';
    formData.append('language_mode', languageMode);
    
    try {
        // Use audioflow endpoint that handles webm properly
        const response = await fetch('/api/transcribe_audio', {
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
// Note: voxIsListening is managed by vox.js module
let whisperPreloaded = false;

async function preloadWhisper() {
    if (whisperPreloaded) return;
    
    const statusEl = document.getElementById('vox-status');
    if (statusEl) statusEl.textContent = 'Loading Whisper model...';
    
    try {
        const response = await fetch('/api/transcribe_preload', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            whisperPreloaded = true;
            window.whisperPreloaded = true;
            console.log('Whisper model preloaded successfully');
        }
    } catch (error) {
        console.error('Error preloading Whisper:', error);
    }
}

async function unloadWhisperModel() {
    try {
        const response = await fetch('/api/unload_vox', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            whisperPreloaded = false;
            window.whisperPreloaded = false;
            window.whisperModelLoaded = false;
            console.log('Whisper model unloaded');
            alert('Speech-to-text model unloaded from VRAM');
        }
    } catch (error) {
        console.error('Error unloading Whisper:', error);
    }
}

// OLD VOX CODE - DISABLED (Using new VOX module in vox.js instead)
/*
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
    // Legacy function - VOX module handles this now
    // Check if voxIsListening exists globally (it's in vox.js scope)
    const isListening = typeof voxIsListening !== 'undefined' ? voxIsListening : false;
    if (!isListening) return;
    
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
        const statusEl = document.getElementById('vox-status');
        if (statusEl) statusEl.textContent = 'Microphone error - check permissions';
        // voxIsListening is in vox.js scope - can't set it here
    }
}

function stopVoxListening() {
    // Legacy function - VOX module handles this now
    if (typeof voxIsListening !== 'undefined') {
        // Can't modify voxIsListening from here - it's in vox.js scope
    }
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    document.getElementById('vox-status').textContent = 'Stopped';
    document.getElementById('waveform').classList.remove('active');
}
*/
// END OLD VOX CODE

// Alias for compatibility (old onclick handler)
function stopVoxCore() {
    // Use new VOX module
    if (typeof VOX !== 'undefined' && VOX.stopRecording) {
        VOX.stopRecording();
    }
}

// OLD VOX processVoxAudio and addVoxMessage - DISABLED (using VOX module instead)
/*
async function processVoxAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'vox_recording.webm');
    
    const language = document.getElementById('vox-language')?.value || 'en';
    formData.append('language', language);
    
    try {
        document.getElementById('vox-status').textContent = 'TRANSCRIBING...';
        
        // Transcribe using new endpoint that doesn't need ffmpeg
        const transcribeResponse = await fetch('/api/transcribe_realtime', {
            method: 'POST',
            body: formData
        });
        
        if (!transcribeResponse.ok) {
            throw new Error(`Transcription failed: ${transcribeResponse.status}`);
        }
        
        const transcribeData = await transcribeResponse.json();
        
        if (!transcribeData.text || transcribeData.text.trim() === '') {
            // No speech detected - status message
            const statusEl = document.getElementById('vox-status');
            if (statusEl) statusEl.textContent = 'No speech detected - Ready';
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
        
        // Update status
        const statusEl = document.getElementById('vox-status');
        if (statusEl) statusEl.textContent = 'Ready';
        
    } catch (error) {
        console.error('Error processing vox audio:', error);
        const statusEl = document.getElementById('vox-status');
        if (statusEl) statusEl.textContent = 'Error - Ready';
        
        // Don't retry on error
        const retryListening = false;
        if (retryListening) {
            setTimeout(() => listenCycle(), 2000);
        }
    }
}

function addVoxMessage(text, role) {
    const chatWindow = document.getElementById('vox-chat-window');
    if (!chatWindow) {
        console.error('VOX chat window not found');
        return;
    }
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `vox-message vox-${role === 'assistant' ? 'bot' : role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'vox-message-content';
    contentDiv.textContent = text;
    
    msgDiv.appendChild(contentDiv);
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
*/
// END OLD VOX processVoxAudio and addVoxMessage

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
    // First try to get description from config.env.txt
    if (configDescriptions[key]) {
        return configDescriptions[key];
    }
    
    // Fallback descriptions for common settings
    const fallbackDescriptions = {
        'MAIN_MODEL': 'Primary LLM model for reasoning and responses',
        'TEMPERATURE': 'Creativity level (0.0 = deterministic, 1.0 = creative)',
        'MAX_TOKENS': 'Maximum response length',
        'ENABLE_WEB_SEARCH': 'Allow AI to search the web for current information',
        'STEALTH_MODE': 'Fully offline mode - no internet access',
        'USE_RAG': 'Enable Retrieval Augmented Generation',
        'TTS_ENABLE': 'Enable text-to-speech output',
        'STT_ENABLE': 'Enable speech-to-text input'
    };
    
    return fallbackDescriptions[key] || 'No description available';
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
        if (!response.ok) return; // Silently fail if server offline
        
        const stats = await response.json();
        
        // Flash green indicator
        const indicator = document.getElementById('stats-indicator');
        if (indicator) {
            indicator.classList.add('active');
            setTimeout(() => indicator.classList.remove('active'), 300);
        }
        
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

// Web search toggle with persistence
document.addEventListener('DOMContentLoaded', () => {
    // Load web search preference on page load
    fetch('/api/preferences')
        .then(res => res.json())
        .then(prefs => {
            if (prefs.webSearchEnabled !== undefined) {
                webSearchEnabled = prefs.webSearchEnabled;
                updateToggleState('web-search-toggle', webSearchEnabled);
            }
        })
        .catch(err => console.error('Failed to load preferences:', err));
    
    const webSearchToggle = document.getElementById('web-search-toggle');
    if (webSearchToggle) {
        webSearchToggle.addEventListener('click', () => {
            webSearchEnabled = !webSearchEnabled;
            updateToggleState('web-search-toggle', webSearchEnabled);
            
            // Save preference
            fetch('/api/preferences', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ webSearchEnabled: webSearchEnabled })
            }).catch(err => console.error('Failed to save web search preference:', err));
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
                
                // Save stealth mode turning off web search
                fetch('/api/preferences', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ webSearchEnabled: false })
                }).catch(err => console.error('Failed to save preference:', err));
            }
        });
    }
});

// --- RIGHT SIDEBAR (RAG) ---
function toggleRightSidebar() {
    document.getElementById('right-sidebar').classList.toggle('open');
}

// --- RAG FOLDER SELECTION ---
let ragFolderFiles = [];
let ragImageFiles = [];

// Handle folder selection
document.addEventListener('DOMContentLoaded', () => {
    const folderInput = document.getElementById('rag-folder-input');
    const imageInput = document.getElementById('rag-image-input');
    
    if (folderInput) {
        folderInput.addEventListener('change', (e) => {
            ragFolderFiles = Array.from(e.target.files);
            const display = document.getElementById('rag-folder-display');
            if (ragFolderFiles.length > 0) {
                // Get folder name from first file's path
                const firstPath = ragFolderFiles[0].webkitRelativePath || ragFolderFiles[0].name;
                const folderName = firstPath.split('/')[0];
                display.value = `${folderName} (${ragFolderFiles.length} files)`;
                console.log(`Selected folder with ${ragFolderFiles.length} files`);
            } else {
                display.value = 'No folder selected';
            }
        });
    }
    
    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            ragImageFiles = Array.from(e.target.files);
            const countSpan = document.getElementById('rag-image-count');
            if (countSpan) {
                countSpan.textContent = `${ragImageFiles.length} images`;
            }
            console.log(`Selected ${ragImageFiles.length} images for RAG`);
        });
    }
});

function selectRagFolder() {
    // Trigger the hidden file input
    document.getElementById('rag-folder-input').click();
}

async function embedFolderContents() {
    if (ragFolderFiles.length === 0 && ragImageFiles.length === 0) {
        alert('Please select a folder or images first');
        return;
    }
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = 'Embedding contents to Weaviate vector DB...\n';
    
    try {
        const formData = new FormData();
        
        // Add text/document files
        for (const file of ragFolderFiles) {
            formData.append('files', file, file.webkitRelativePath || file.name);
        }
        
        // Add image files
        for (const file of ragImageFiles) {
            formData.append('images', file, file.name);
        }
        
        const embedModel = document.getElementById('rag-embed-model').value;
        formData.append('embed_model', embedModel);
        formData.append('vision_model', currentModel); // Use current chat model for vision
        
        statusLog.textContent += `Uploading ${ragFolderFiles.length} files and ${ragImageFiles.length} images...\n`;
        
        const response = await fetch('/api/embed_to_rag', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            statusLog.textContent += `✓ Embedded ${result.embedded_count} items to vector DB\n`;
            statusLog.textContent += `Collection: ${result.collection_name}\n`;
        } else {
            statusLog.textContent += `✗ Error: ${result.error}\n`;
        }
    } catch (error) {
        statusLog.textContent += `✗ Error: ${error.message}\n`;
        console.error('RAG embedding error:', error);
    }
}

function searchRagIndex() {
    const query = prompt('Enter search query:');
    if (!query) return;
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = `Searching vector DB for: ${query}...\n`;
    
    // TODO: Implement actual search API call
    fetch('/api/search_rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 })
    })
    .then(res => res.json())
    .then(result => {
        if (result.results) {
            statusLog.textContent += `Found ${result.results.length} results:\n`;
            result.results.forEach((r, i) => {
                statusLog.textContent += `${i+1}. ${r.content.substring(0, 100)}... (score: ${r.score.toFixed(3)})\n`;
            });
        }
    })
    .catch(err => {
        statusLog.textContent += `✗ Search error: ${err.message}\n`;
    });
}

// --- CONFIRMATION MODAL ---
function showConfirmModal(title, message, onConfirm) {
    console.log('[Modal] Attempting to show modal:', title);
    const modal = document.getElementById('confirm-modal');
    const modalTitle = document.getElementById('confirm-modal-title');
    const modalMessage = document.getElementById('confirm-modal-message');
    const confirmBtn = document.getElementById('confirm-modal-confirm');
    const cancelBtn = document.getElementById('confirm-modal-cancel');
    
    if (!modal || !modalTitle || !modalMessage || !confirmBtn || !cancelBtn) {
        console.error('[Modal] Missing elements:', { modal, modalTitle, modalMessage, confirmBtn, cancelBtn });
        // Fallback to browser confirm
        if (confirm(message)) {
            onConfirm();
        }
        return;
    }
    
    console.log('[Modal] All elements found, showing modal');
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    
    // Remove old event listeners by cloning buttons
    const newConfirmBtn = confirmBtn.cloneNode(true);
    const newCancelBtn = cancelBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
    cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);
    
    // Add new event listeners
    newCancelBtn.onclick = () => {
        console.log('[Modal] Cancel clicked');
        closeConfirmModal();
    };
    
    newConfirmBtn.onclick = () => {
        console.log('[Modal] Confirm clicked');
        closeConfirmModal();
        if (typeof onConfirm === 'function') {
            onConfirm();
        }
    };
    
    modal.classList.remove('hidden');
    console.log('[Modal] Modal should now be visible');
}

function closeConfirmModal() {
    const modal = document.getElementById('confirm-modal');
    modal.classList.add('hidden');
}

async function confirmDeleteChat(chatId, chatTitle, isVoiceChat) {
    console.log('[DELETE] confirmDeleteChat called with chatId:', chatId, 'title:', chatTitle, 'isVoiceChat:', isVoiceChat);
    showConfirmModal(
        'Delete Chat',
        `Are you sure you want to delete "${chatTitle}"? This cannot be undone.`,
        async () => {
            try {
                const response = await fetch(`/api/chats/${chatId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    // Reload appropriate chat list
                    if (isVoiceChat) {
                        if (typeof loadVoiceChatList !== 'undefined') {
                            await loadVoiceChatList();
                        }
                    } else {
                        await loadChatList();
                    }
                    
                    // Clear current session if it was deleted
                    if (chatId === currentSessionId) {
                        startNewChat();
                    } else if (isVoiceChat && typeof voxSessionId !== 'undefined' && chatId === voxSessionId) {
                        if (typeof clearVOXChat !== 'undefined') {
                            clearVOXChat();
                        }
                        voxSessionId = null;
                    }
                    
                    console.log(`Deleted ${isVoiceChat ? 'voice' : 'text'} chat:`, chatId);
                } else {
                    alert('Failed to delete chat');
                }
            } catch (error) {
                console.error('Error deleting chat:', error);
                alert('Error deleting chat');
            }
        }
    );
}
