let currentSessionId = null;
let mediaRecorder;
let audioChunks = [];

document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    loadChatList();
    setInterval(updateStats, 2000);
    document.querySelector('.search-box input').addEventListener('input', (e) => loadChatList(e.target.value));
});

function switchTab(tabId) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId + '-view').classList.add('active');
    const map = {'chat': 0, 'browser': 1, 'settings': 2};
    if(map[tabId] !== undefined) document.querySelectorAll('.nav-btn')[map[tabId]].classList.add('active');
    if(tabId === 'settings') loadSettings();
}

function toggleVoice() { document.getElementById('voice-overlay').classList.toggle('hidden'); }
function toggleRightSidebar() { document.getElementById('right-sidebar').classList.toggle('open'); }

// --- CHAT LOGIC WITH STREAMING ---
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value;
    if(!msg.trim()) return;

    appendMessage(msg, 'user');
    input.value = '';

    // Prepare Streaming Request
    const ragEnabled = document.getElementById('rag-toggle')?.checked || false;
    const ragFolder = document.getElementById('rag-folder')?.value || '';

    // Create Bot Message Placeholder
    const botMsgId = 'msg-' + Date.now();
    createBotPlaceholder(botMsgId);

    try {
        const response = await fetch('/api/chat_stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: msg,
                session_id: currentSessionId,
                rag_enabled: ragEnabled,
                context_folder: ragFolder
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, {stream: true});
            const lines = chunk.split('\n\n');
            
            for (const line of lines) {
                if(line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.replace('data: ', ''));
                        updateBotMessage(botMsgId, data);
                    } catch(e) { console.error(e); }
                }
            }
        }
    } catch (err) {
        console.error(err);
        appendMessage("ERROR: CONNECTION LOST", 'system');
    }
}

function createBotPlaceholder(id) {
    const win = document.getElementById('chat-window');
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.id = id;
    
    // Thinking block (hidden by default)
    const thinking = document.createElement('div');
    thinking.className = 'thinking-block';
    div.appendChild(thinking);
    
    // Content block
    const content = document.createElement('div');
    content.className = 'msg-content';
    div.appendChild(content);

    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
}

function updateBotMessage(id, data) {
    const msgDiv = document.getElementById(id);
    if(!msgDiv) return;
    
    if(data.thinking) {
        const t = msgDiv.querySelector('.thinking-block');
        t.innerText += data.thinking;
        t.classList.add('active');
    }
    if(data.content) {
        msgDiv.querySelector('.msg-content').innerText += data.content;
    }
    if(data.session_id) {
        currentSessionId = data.session_id;
        loadChatList(); // Refresh sidebar order
    }
    
    const win = document.getElementById('chat-window');
    win.scrollTop = win.scrollHeight;
}

function appendMessage(text, type) {
    const win = document.getElementById('chat-window');
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    div.innerText = text; 
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
}

// --- FILE / VOICE HANDLERS ---
function uploadFile(input) {
    const file = input.files[0];
    const fd = new FormData(); fd.append('file', file);
    fetch('/api/upload', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
        document.getElementById('chat-input').value += ` [FILE: ${d.filename}] `;
    });
}

function uploadFolder(input) {
    // Only captures first file path for RAG context mock
    if(input.files.length > 0) {
        const path = input.files[0].webkitRelativePath.split('/')[0];
        document.getElementById('rag-folder').value = path;
        document.getElementById('rag-toggle').checked = true;
        toggleRightSidebar(); // Open sidebar to show settings
    }
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.start();
        mediaRecorder.addEventListener("dataavailable", e => audioChunks.push(e.data));
    });
}

function stopRecording() {
    if(mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.addEventListener("stop", () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });
            const fd = new FormData(); fd.append("audio", blob, "in.wav");
            fetch('/api/transcribe', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
                document.getElementById('chat-input').value += d.text + " ";
            });
        });
    }
}

// --- STANDARD FUNCTIONS (Stats, Load Chats, Settings - Kept from old) ---
function loadChatList(query='') {
    const url = query ? `/api/chats?search=${encodeURIComponent(query)}` : '/api/chats';
    fetch(url).then(r=>r.json()).then(chats=>{
        const list = document.querySelector('.chat-list');
        list.innerHTML = '<li style="color:var(--primary-color)" onclick="startNewChat()"><i class="fa-solid fa-plus"></i> NEW_SESSION</li>';
        chats.forEach(c => {
            const li = document.createElement('li');
            li.innerText = c.title;
            if(c.id === currentSessionId) li.classList.add('active');
            li.onclick = () => loadSession(c.id);
            list.appendChild(li);
        });
    });
}

function loadSession(id) {
    currentSessionId = id;
    fetch(`/api/chats/${id}`).then(r=>r.json()).then(d=>{
        document.getElementById('chat-window').innerHTML = '';
        d.messages.forEach(m => {
            if(m.role === 'user') appendMessage(m.content, 'user');
            else {
                createBotPlaceholder('hist-'+Date.now()); // simplified logic for history
                const els = document.querySelectorAll('.msg.bot');
                els[els.length-1].querySelector('.msg-content').innerText = m.content;
            }
        });
        loadChatList();
    });
}

function startNewChat() {
    currentSessionId = null;
    document.getElementById('chat-window').innerHTML = '<div class="msg system">Initialization Complete. Systems Nominal.</div>';
    loadChatList();
}

function updateStats() {
    fetch('/api/system_stats').then(r=>r.json()).then(d=>{
        setBar('cpu', d.cpu); setBar('ram', d.ram); setBar('gpu', d.gpu); setBar('vram', d.vram)
        document.getElementById('temp-val').innerText = `${Math.round(d.gpu_temp)}°C`;
    });
}

function setBar(id, val) {
    const bar = document.getElementById(`${id}-bar`);
    const txt = document.getElementById(`${id}-val`);
    if(bar) { bar.style.width=`${val}%`; txt.innerText=`${Math.round(val)}%`; }
}

function loadSettings() {
    fetch('/api/settings').then(r=>r.json()).then(s=>{
        const c = document.getElementById('settings-container'); c.innerHTML='';
        for(const [k,v] of Object.entries(s)) {
            const d = document.createElement('div'); d.className='setting-card';
            d.innerHTML = `<label>${k.toUpperCase()}</label>`;
            const i = document.createElement('input'); i.dataset.key=k;
            if(typeof v==='boolean'){ i.type='checkbox'; i.checked=v; }
            else { i.value=v; }
            d.appendChild(i); c.appendChild(d);
        }
    });
}

function saveSettings() {
    const data={};
    document.querySelectorAll('#settings-container input').forEach(i=>{
        data[i.dataset.key] = i.type==='checkbox'?i.checked:i.value;
    });
    fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)})
    .then(()=>alert("SAVED"));
}

document.getElementById('chat-input').addEventListener('keypress', e => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });