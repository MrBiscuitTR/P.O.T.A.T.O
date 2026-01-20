// --- Splitter logic ---
const leftPanel = document.getElementById('left-panel');
const splitter = document.getElementById('splitter');
let isDragging = false;

splitter.addEventListener('mousedown', function(e) {
    isDragging = true;
    document.body.style.cursor = 'ew-resize';
});

document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    let newWidth = e.clientX;
    if (newWidth < 200) newWidth = 200;
    if (newWidth > 600) newWidth = 600;
    leftPanel.style.width = newWidth + 'px';
    splitter.style.left = newWidth + 'px';
});

document.addEventListener('mouseup', function(e) {
    isDragging = false;
    document.body.style.cursor = '';
});

// --- System/model info human readable ---
function progressBar(percent, color, text) {
    // Use flexbox for alignment, and avoid absolute positioning
    return `
    <div class="progress-bar" style="display:flex;align-items:center;position:relative;">
        <div class="progress-bar-inner" style="width:${percent}%;background:${color};"></div>
        <span class="progress-bar-label" style="position:absolute;left:12px;top:0;line-height:18px;">${text}</span>
    </div>`;
}

function tempColor(temp) {
    if (temp < 50) return "#4caf50";
    if (temp < 65) return "#ff9800";
    return "#f44336";
}

async function fetchSystemInfo() {
    const res = await fetch('/api/system_info');
    const data = await res.json();
    let ram = data.ram || {};
    let cpu = data.cpu || {};
    let gpus = data.gpus || [];

    let ramBar = progressBar(
        ram.usage_percent || 0,
        "#2196f3",
        `${ram.available_gb ?? "?"} GB / ${ram.total_gb ?? "?"} GB (${ram.usage_percent ?? "?"}%)`
    );
    let cpuBar = progressBar(
        cpu.usage_percent || 0,
        "#9c27b0",
        `${cpu.usage_percent ?? "?"}%`
    );
    let gpuHtml = "";
    if (gpus.length) {
        gpuHtml = gpus.map(gpu => `
            <div style="margin-bottom:8px;">
                <div><b>${gpu.id_name}</b></div>
                <div>Load: ${progressBar(gpu.load_percent || 0, "#03a9f4", `${gpu.load_percent ?? "?"}%`)}</div>
                <div>VRAM: ${progressBar(
                    gpu.memoryTotal_MB ? ((gpu.memoryTotal_MB - gpu.memoryFree_MB) / gpu.memoryTotal_MB * 100) : 0,
                    "#8bc34a",
                    `${gpu.memoryFree_MB ?? "?"}MB free / ${gpu.memoryTotal_MB ?? "?"}MB`
                )}</div>
                <div>Temp: <span style="color:${tempColor(gpu.temperature_C)}">${gpu.temperature_C ?? "?"}°C</span></div>
            </div>
        `).join("");
    }

    document.getElementById('system-info').innerHTML =
        `<div style="margin-bottom:8px;"><b>RAM:</b>${ramBar}</div>
         <div style="margin-bottom:8px;"><b>CPU Usage:</b>${cpuBar}</div>
         <div><b>GPUs:</b>${gpuHtml || "<div style='color:#888;'>No GPU detected</div>"}</div>`;
}

async function fetchModelInfo() {
    const res = await fetch('/api/model_info');
    const data = await res.json();
    document.getElementById('model-info').innerHTML =
        `<b>Name:</b> ${data.name}<br>
         <b>Version:</b> ${data.version}<br>
         <b>Status:</b> ${data.status}<br>
         <b>Context Length:</b> ${data.context_length}`;
}

fetchSystemInfo();
fetchModelInfo();
setInterval(fetchSystemInfo, 2000);

// --- Chat navbar logic ---
const chatListDiv = document.getElementById('chat-list');
const newChatBtn = document.getElementById('new-chat-btn');
const chatSearch = document.getElementById('chat-search');
let currentChatId = window.current_chat_id || "default";
let chats = window.chats || [];
let chatHistories = {};

function renderChatList(chats, query = "") {
    chatListDiv.innerHTML = '';
    for (const chat of chats) {
        if (query && !chat.title.toLowerCase().includes(query.toLowerCase()) && !chat.id.includes(query)) continue;
        const btn = document.createElement('button');
        btn.textContent = chat.title || chat.id;
        btn.className = 'chat-list-item' + (chat.id === currentChatId ? ' active' : '');
        btn.onclick = () => switchChat(chat.id);
        chatListDiv.appendChild(btn);
    }
}

async function updateChatList() {
    const res = await fetch('/api/chats');
    chats = await res.json();
    renderChatList(chats, chatSearch.value);
}

async function switchChat(chatId) {
    currentChatId = chatId;
    const res = await fetch(`/api/chats/${chatId}`);
    const history = await res.json();
    window.chat_history = history;
    renderChatHistory(window.chat_history);
    renderChatList(chats, chatSearch.value);
}

newChatBtn.addEventListener('click', async () => {
    const res = await fetch('/api/chats/new', {method: 'POST'});
    const data = await res.json();
    currentChatId = data.id;
    window.chat_history = [];
    await updateChatList();
    renderChatHistory(window.chat_history);
    renderChatList(chats, chatSearch.value);
});

chatSearch.addEventListener('input', async () => {
    renderChatList(chats, chatSearch.value);
});

// --- Chat logic ---
const chatHistoryDiv = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const tokensDisplay = document.getElementById('tokens-display');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const voiceBtn = document.getElementById('voice-btn');

let stopRequested = false;

function renderChatHistory(history) {
    chatHistoryDiv.innerHTML = '';
    for (const msg of history) {
        const div = document.createElement('div');
        div.className = 'message ' + msg.role;
        div.textContent = msg.content;
        chatHistoryDiv.appendChild(div);
    }
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

window.addEventListener('DOMContentLoaded', () => {
    renderChatList(chats);
    renderChatHistory(window.chat_history);
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    stopRequested = false;
    sendBtn.disabled = true;
    stopBtn.disabled = false;
    // Add user message to chat
    window.chat_history = window.chat_history || [];
    window.chat_history.push({role: 'user', content: message});
    renderChatHistory(window.chat_history);

    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message, chat_id: currentChatId})
    });
    const data = await res.json();
    if (!stopRequested) {
        window.chat_history = data.chat_history;
        renderChatHistory(window.chat_history);
        tokensDisplay.textContent = `Tokens: ${data.tokens}`;
        await updateChatList(); // update chat list titles if needed
    }
    chatInput.value = '';
    sendBtn.disabled = false;
    stopBtn.disabled = true;
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendMessage();
});
stopBtn.addEventListener('click', () => {
    stopRequested = true;
    sendBtn.disabled = false;
    stopBtn.disabled = true;
});

// File attach
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        alert('File attached: ' + fileInput.files[0].name);
        // Implement file upload logic as needed
    }
});

// Voice mode (stub)
voiceBtn.addEventListener('click', () => {
    alert('Voice mode not implemented yet.');
});
