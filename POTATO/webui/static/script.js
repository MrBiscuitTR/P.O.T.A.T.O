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
let uploadedImages = [];  // For vision model image uploads
let settings = {};
let currentSettingsTab = 'core';
let configDescriptions = {}; // Descriptions from config.env.txt
let isGenerating = false; // Track if AI is currently generating
let currentAbortController = null; // Controller to cancel ongoing requests

// Upload preview state - unified for files and images
let uploadPreviews = [];  // Array of {id, filename, path, type, isImage, base64, embedded, uploadProgress, abortController}
let pasteUploadsUseRag = false;  // Setting from config
const MAX_PASTE_UPLOADS = 20;

// Vision model state
let currentModelIsVision = false;
let backupVisionModel = 'llava:7b';
let canProcessImages = false;
let availableVisionModels = [];

// Model unload flags (shared with VOX Core)
window.sttManuallyUnloaded = false;
window.ttsManuallyUnloaded = false;

// Load config descriptions on page load
async function loadConfigDescriptions() {
    try {
        const response = await fetch('/api/settings/descriptions');
        configDescriptions = await response.json().catch(error => {
            console.error('Error parsing config descriptions:', error);
        });
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
// On page unload/reload: fire-and-forget unload of VOX models
window.addEventListener('beforeunload', () => {
    navigator.sendBeacon('/api/unload_stt');
    navigator.sendBeacon('/api/unload_tts');
    navigator.sendBeacon('/api/stop_all_vox');
});

document.addEventListener('DOMContentLoaded', () => {
    loadPreferences();  // Load saved preferences first
    loadModels();
    loadConfigDescriptions(); // Load descriptions from config.env.txt
    loadSettings();
    updateStats();
    loadChatList();
    
    // Update stats every 2 seconds
    setInterval(updateStats, 2000);
    
    // Search chats with debouncing (less sensitive)
    let searchTimeout;
    document.querySelector('.search-box input').addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            loadChatList(e.target.value);
        }, 300); // 300ms delay
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
        toggleMobileMenu();
    });
    
    // Close chat list or navbar when clicking on overlay
    document.body.addEventListener('click', (e) => {
        if (e.target === document.body && window.innerWidth <= 768) {
            // Check if overlay is visible by checking the pseudo-element
            const styles = window.getComputedStyle(document.body, '::before');
            const overlayVisible = styles.getPropertyValue('content') !== 'none' && styles.getPropertyValue('content') !== '';
            
            if (overlayVisible) {
                // Close whichever is open
                if (document.body.classList.contains('sidebar-open')) {
                    closeMobileSidebar();
                }
                if (document.body.classList.contains('chat-list-open')) {
                    closeChatList();
                }
            }
        }
    });
    
    // Auto-scroll detection for chat window
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow) {
        chatWindow.addEventListener('scroll', () => {
            const isAtBottom = chatWindow.scrollHeight - chatWindow.scrollTop <= chatWindow.clientHeight + 50;
            autoScrollEnabled = isAtBottom;
        });
    }
    
    // Paste event listener for file/image uploads (only on document to avoid duplicates)
    document.addEventListener('paste', handlePaste);
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
        const data = await response.json().catch(error => {
            console.error('Error loading models:', error);
        });
        
        const selector = document.getElementById('model-selector');
        selector.innerHTML = '';
        
        // Build a map of model vision capabilities
        const visionMap = {};
        if (data.models_info) {
            data.models_info.forEach(info => {
                visionMap[info.name] = info.is_vision;
            });
        }
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                // Add (VL) suffix for vision-capable models
                const isVision = visionMap[model] || false;
                option.textContent = isVision ? `${model} (VL)` : model;
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
        // Check vision capability for current model
        await checkVisionCapability();
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function updateModelDisplay() {
    // Show (VL) suffix if current model is vision-capable
    const displayText = currentModelIsVision ? `AI: ${currentModel} (VL)` : `AI: ${currentModel}`;
    document.getElementById('current-model-display').textContent = displayText;
}

async function checkVisionCapability() {
    // Check if current model supports vision and update UI accordingly
    try {
        const response = await fetch('/api/check_vision_capability', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        const data = await response.json();
        
        currentModelIsVision = data.is_vision;
        backupVisionModel = data.backup_vision_model;
        canProcessImages = data.can_process_images;
        availableVisionModels = data.available_vision_models || [];
        
        // Update model display with vision badge
        updateModelDisplay();
        
        // Update UI to show/hide image upload warnings
        updateVisionUI();
        
        console.log(`[VISION] Model: ${currentModel}, isVision: ${currentModelIsVision}, canProcess: ${canProcessImages}`);
    } catch (error) {
        console.error('Error checking vision capability:', error);
        currentModelIsVision = false;
        canProcessImages = false;
    }
}

function updateVisionUI() {
    // Show/hide warning banner based on vision capability
    let warningBanner = document.getElementById('vision-warning-banner');
    
    if (!canProcessImages) {
        // No vision model available at all
        if (!warningBanner) {
            warningBanner = document.createElement('div');
            warningBanner.id = 'vision-warning-banner';
            warningBanner.className = 'vision-warning-banner';
            const inputContainer = document.querySelector('.input-area-container');
            if (inputContainer) {
                inputContainer.insertBefore(warningBanner, inputContainer.firstChild);
            }
        }
        warningBanner.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>No vision model available. Image uploads disabled. Install a vision model (e.g., llava:7b) via Ollama.</span>
            <button onclick="this.parentElement.style.display='none'" class="close-banner">&times;</button>
        `;
        warningBanner.style.display = 'flex';
        
        // Disable image upload
        const imageUploadIcon = document.querySelector('.widget-icon[title="Upload Image"]');
        if (imageUploadIcon) {
            imageUploadIcon.classList.add('disabled');
            imageUploadIcon.title = 'No vision model available';
        }
    } else if (!currentModelIsVision && availableVisionModels.length > 0) {
        // Backup model available but not current model
        if (!warningBanner) {
            warningBanner = document.createElement('div');
            warningBanner.id = 'vision-warning-banner';
            warningBanner.className = 'vision-info-banner';
            const inputContainer = document.querySelector('.input-area-container');
            if (inputContainer) {
                inputContainer.insertBefore(warningBanner, inputContainer.firstChild);
            }
        }
        warningBanner.innerHTML = `
            <i class="fas fa-info-circle"></i>
            <span>Current model doesn't support images. Backup model (${backupVisionModel}) will be used for image processing.</span>
            <button onclick="this.parentElement.style.display='none'" class="close-banner">&times;</button>
        `;
        warningBanner.className = 'vision-info-banner';
        warningBanner.style.display = 'flex';
        
        // Enable image upload
        const imageUploadIcon = document.querySelector('.widget-icon[title*="Image"]');
        if (imageUploadIcon) {
            imageUploadIcon.classList.remove('disabled');
        }
    } else {
        // Vision model is active
        if (warningBanner) {
            warningBanner.style.display = 'none';
        }
        
        // Enable image upload
        const imageUploadIcon = document.querySelector('.widget-icon[title*="Image"]');
        if (imageUploadIcon) {
            imageUploadIcon.classList.remove('disabled');
        }
    }
    
    // Update uploaded images display
    updateUploadedImagesDisplay();
}

function updateUploadedImagesDisplay() {
    // Clear uploaded images if vision not available
    if (!canProcessImages && uploadedImages.length > 0) {
        uploadedImages = [];
        appendMessage('Images cleared - no vision model available', 'system');
    }
    
    // Update UI to show uploaded images count
    const imageCountSpan = document.getElementById('uploaded-images-count');
    if (imageCountSpan) {
        imageCountSpan.textContent = uploadedImages.length > 0 ? `(${uploadedImages.length} images)` : '';
    }
    
    // Also update the new unified preview
    renderUploadPreviews();
}

// ===============================================
// PASTE UPLOAD HANDLING
// ===============================================

function handlePaste(e) {
    const items = e.clipboardData?.items;
    if (!items || items.length === 0) return;
    
    const files = [];
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file') {
            const file = item.getAsFile();
            if (file) files.push(file);
        }
    }
    
    if (files.length === 0) return;
    
    // Prevent default paste if we have files
    e.preventDefault();
    
    // Check limit
    if (uploadPreviews.length + files.length > MAX_PASTE_UPLOADS) {
        appendMessage(`Maximum ${MAX_PASTE_UPLOADS} files allowed at a time. You have ${uploadPreviews.length} already.`, 'system');
        return;
    }
    
    // Upload each file
    files.forEach(file => {
        uploadPastedFile(file);
    });
}

async function uploadPastedFile(file) {
    const isImage = file.type.startsWith('image/');
    const id = 'upload-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    // Check if image and no vision available
    if (isImage && !canProcessImages) {
        appendMessage(`Cannot upload image "${file.name}" - no vision model available`, 'system');
        return;
    }
    
    // Create preview entry
    const preview = {
        id: id,
        filename: file.name,
        path: null,
        type: getFileType(file.name),
        isImage: isImage,
        base64: null,
        embedded: false,
        uploadProgress: 0,
        abortController: new AbortController(),
        file: file  // Keep reference for potential re-upload
    };
    
    uploadPreviews.push(preview);
    renderUploadPreviews();
    
    // Generate a proper UUID for new chats if no session exists
    // This will be used as the folder name for uploads
    if (!currentSessionId) {
        currentSessionId = crypto.randomUUID();
        console.log('[UPLOAD] Generated new session ID:', currentSessionId);
    }
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chat_id', currentSessionId);
    formData.append('model', currentModel);
    formData.append('use_rag', pasteUploadsUseRag ? 'true' : 'false');
    
    try {
        // Use XMLHttpRequest for progress tracking
        await uploadWithProgress(preview, formData);
    } catch (error) {
        if (error.name === 'AbortError') {
            // Upload was cancelled, remove from previews
            const idx = uploadPreviews.findIndex(p => p.id === id);
            if (idx > -1) uploadPreviews.splice(idx, 1);
            renderUploadPreviews();
        } else {
            console.error('Upload error:', error);
            preview.uploadProgress = -1;  // Mark as failed
            renderUploadPreviews();
            appendMessage(`Failed to upload "${file.name}": ${error.message}`, 'system');
        }
    }
}

function uploadWithProgress(preview, formData) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        // Store xhr for cancellation
        preview.xhr = xhr;
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                preview.uploadProgress = Math.round((e.loaded / e.total) * 100);
                renderUploadPreviews();
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (data.success) {
                        preview.path = data.path;
                        preview.uploadProgress = 100;
                        preview.embedded = data.embedded || false;
                        
                        // Handle different response types based on file and model
                        if (preview.isImage) {
                            // ALWAYS store base64 for images - let send decide based on current model
                            if (data.image_base64) {
                                uploadedImages.push({
                                    filename: data.filename,
                                    base64: data.image_base64,
                                    description: data.description || null  // Fallback description for non-VL
                                });
                                console.log('[UPLOAD] Image stored:', data.filename, 'base64 length:', data.image_base64?.length || 0);
                                
                                // If there's also a description (non-VL fallback), store it too
                                if (data.description) {
                                    console.log('[UPLOAD] Also has description for non-VL fallback:', data.description.length, 'chars');
                                }
                            } else if (data.content) {
                                // Fallback: no base64, just description
                                uploadedFiles.push({
                                    filename: data.filename,
                                    content: data.content,
                                    file_type: 'image_description'
                                });
                                console.log('[UPLOAD] Image as text only:', data.filename);
                            }
                        } else {
                            // PDF or text file handling
                            if (data.file_type === 'pdf' && data.pdf_pages_base64) {
                                // PDF with page images - store for VL models
                                const pdfEntry = {
                                    filename: data.filename,
                                    base64: data.pdf_pages_base64,  // Array of base64 images (may be limited to 8 for large PDFs)
                                    isPdf: true,
                                    pages: data.pages || data.pdf_pages_base64.length
                                };
                                
                                // If PDF was batch OCR-processed (>8 pages), include extracted content
                                if (data.ocr_processed && data.content) {
                                    pdfEntry.ocr_processed = true;
                                    pdfEntry.content = data.content;
                                    pdfEntry.ocr_model = data.ocr_model;
                                    console.log('[UPLOAD] PDF batch OCR processed:', data.filename, 'pages:', data.pages, 'text:', data.content.length, 'chars');
                                } else {
                                    console.log('[UPLOAD] PDF stored as images:', data.filename, 'pages:', data.pdf_pages_base64.length);
                                }
                                
                                uploadedImages.push(pdfEntry);
                                
                                // Also store text content as fallback for non-VL models
                                if (data.content) {
                                    uploadedFiles.push({
                                        filename: data.filename,
                                        content: data.content,
                                        file_type: data.file_type,
                                        pages: data.pages
                                    });
                                    console.log('[UPLOAD] PDF also has text fallback:', data.content?.length, 'chars');
                                }
                            } else {
                                // Regular text file or PDF text extraction
                                uploadedFiles.push({
                                    filename: data.filename,
                                    content: data.content,
                                    file_type: data.file_type,
                                    pages: data.pages
                                });
                                console.log('[UPLOAD] Added file:', data.filename, 'content length:', data.content?.length || 0, 'pages:', data.pages || 'N/A');
                            }
                        }
                        
                        renderUploadPreviews();
                        resolve(data);
                    } else {
                        reject(new Error(data.error || 'Upload failed'));
                    }
                } catch (e) {
                    reject(new Error('Invalid server response'));
                }
            } else {
                reject(new Error(`HTTP ${xhr.status}`));
            }
        });
        
        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        
        xhr.open('POST', '/api/upload_chat_file');
        xhr.send(formData);
    });
}

function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const types = {
        'pdf': 'pdf',
        'doc': 'word', 'docx': 'word',
        'xls': 'excel', 'xlsx': 'excel', 'csv': 'excel',
        'py': 'code', 'js': 'code', 'ts': 'code', 'html': 'code', 'css': 'code', 'json': 'code', 'xml': 'code',
        'zip': 'archive', 'rar': 'archive', '7z': 'archive', 'tar': 'archive', 'gz': 'archive',
        'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'gif': 'image', 'webp': 'image', 'bmp': 'image',
        'txt': 'text', 'md': 'text'
    };
    return types[ext] || 'file';
}

function getFileIcon(type) {
    const icons = {
        'pdf': 'fa-file-pdf',
        'word': 'fa-file-word',
        'excel': 'fa-file-excel',
        'code': 'fa-file-code',
        'archive': 'fa-file-archive',
        'image': 'fa-file-image',
        'text': 'fa-file-alt',
        'file': 'fa-file'
    };
    return icons[type] || 'fa-file';
}

function renderUploadPreviews() {
    const container = document.getElementById('upload-preview-container');
    const scroll = document.getElementById('upload-preview-scroll');
    
    if (!container || !scroll) return;
    
    // Show/hide container
    container.style.display = uploadPreviews.length > 0 ? 'block' : 'none';
    
    // Clear and rebuild
    scroll.innerHTML = '';
    
    uploadPreviews.forEach(preview => {
        const item = document.createElement('div');
        item.className = 'upload-preview-item';
        if (preview.embedded) item.classList.add('embedded');
        if (preview.uploadProgress < 100 && preview.uploadProgress >= 0) item.classList.add('uploading');
        
        // Thumbnail
        const thumb = document.createElement('div');
        thumb.className = 'upload-preview-thumb';
        
        if (preview.isImage && preview.base64) {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + preview.base64;
            img.alt = preview.filename;
            thumb.appendChild(img);
        } else if (preview.isImage && preview.file) {
            // Show preview while uploading
            const img = document.createElement('img');
            img.src = URL.createObjectURL(preview.file);
            img.alt = preview.filename;
            thumb.appendChild(img);
        } else {
            const icon = document.createElement('i');
            icon.className = 'fas ' + getFileIcon(preview.type);
            thumb.appendChild(icon);
        }
        
        item.appendChild(thumb);
        
        // Info
        const info = document.createElement('div');
        info.className = 'upload-preview-info';
        
        const name = document.createElement('div');
        name.className = 'upload-preview-name';
        name.textContent = preview.filename;
        name.title = preview.filename;
        info.appendChild(name);
        
        const status = document.createElement('div');
        status.className = 'upload-preview-status';
        if (preview.uploadProgress < 0) {
            status.textContent = 'Failed';
            status.style.color = 'var(--accent-color)';
        } else if (preview.uploadProgress < 100) {
            status.textContent = preview.uploadProgress + '%';
        } else if (preview.embedded) {
            status.textContent = 'RAG';
            status.classList.add('embedded');
        } else {
            status.textContent = 'Ready';
        }
        info.appendChild(status);
        
        item.appendChild(info);
        
        // Circular progress indicator (while uploading) - positioned in top-left corner
        if (preview.uploadProgress >= 0 && preview.uploadProgress < 100) {
            const circularProgress = document.createElement('div');
            circularProgress.className = 'upload-circular-progress';
            const percent = preview.uploadProgress;
            // SVG circular progress
            circularProgress.innerHTML = `
                <svg viewBox="0 0 36 36" class="circular-chart">
                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    <path class="circle" stroke-dasharray="${percent}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                    <text x="18" y="20.5" class="percentage">${percent}%</text>
                </svg>
            `;
            item.appendChild(circularProgress);
        }
        
        // Embedded badge
        if (preview.embedded) {
            const badge = document.createElement('div');
            badge.className = 'upload-embedded-badge';
            badge.textContent = 'RAG';
            item.appendChild(badge);
        }
        
        // Remove button (only show when not uploading or if upload is done)
        const removeBtn = document.createElement('button');
        removeBtn.className = 'upload-preview-remove';
        removeBtn.innerHTML = '<i class="fas fa-times"></i>';
        removeBtn.title = preview.uploadProgress < 100 ? 'Cancel upload' : (preview.embedded ? 'Remove (will also delete embeddings)' : 'Remove');
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeUploadPreview(preview.id);
        };
        item.appendChild(removeBtn);
        
        scroll.appendChild(item);
    });
}

async function removeUploadPreview(id) {
    const idx = uploadPreviews.findIndex(p => p.id === id);
    if (idx === -1) return;
    
    const preview = uploadPreviews[idx];
    
    // Cancel upload if in progress
    if (preview.xhr && preview.uploadProgress < 100) {
        preview.xhr.abort();
    }
    
    // If embedded, try to delete embeddings
    if (preview.embedded && preview.path) {
        try {
            await fetch('/api/delete_file_embeddings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chat_id: currentSessionId,
                    filename: preview.filename
                })
            });
        } catch (e) {
            console.warn('Failed to delete embeddings:', e);
        }
    }
    
    // Delete file if uploaded
    if (preview.path) {
        try {
            await fetch('/api/delete_uploaded_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: preview.path })
            });
        } catch (e) {
            console.warn('Failed to delete file:', e);
        }
    }
    
    // Remove from legacy arrays
    if (preview.isImage) {
        const imgIdx = uploadedImages.findIndex(img => img.path === preview.path);
        if (imgIdx > -1) uploadedImages.splice(imgIdx, 1);
    } else {
        const fileIdx = uploadedFiles.findIndex(f => f.path === preview.path);
        if (fileIdx > -1) uploadedFiles.splice(fileIdx, 1);
    }
    
    // Remove from previews
    uploadPreviews.splice(idx, 1);
    renderUploadPreviews();
}

function clearAllUploadPreviews() {
    // Cancel all pending uploads
    uploadPreviews.forEach(p => {
        if (p.xhr && p.uploadProgress < 100) {
            p.xhr.abort();
        }
    });
    
    uploadPreviews = [];
    uploadedFiles = [];
    uploadedImages = [];
    renderUploadPreviews();
}

// ===============================================
// END PASTE UPLOAD HANDLING
// ===============================================

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
        
        // Check vision capability for new model
        await checkVisionCapability();
        
        console.log(`Switched to model: ${currentModel}`);
    } catch (error) {
        console.error('Error switching model:', error);
        currentModel = newModel; // Still switch even if stop fails
        updateModelDisplay();
        await checkVisionCapability();
    }
}

// --- TAB SWITCHING ---
async function switchTab(tabId) {
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
    
    // DYNAMIC VRAM MANAGEMENT
    if (tabId === 'vox-core') {
        console.log('[VOX] Entering VOX Core tab - dynamic VRAM management...');
        
        // 1. Stop any active chat model generation
        if (isGenerating) {
            console.log('[VOX] Stopping active chat generation...');
            if (currentAbortController) {
                currentAbortController.abort();
            }
            await fetch('/api/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: currentModel })
            }).catch(e => console.warn('[VOX] Stop failed:', e));
            
            isGenerating = false;
            resetButtonsAfterInference();
        }
        
        // 2. Unload chat model from VRAM to free space for VOX models
        if (currentModel) {
            console.log(`[VOX] Unloading chat model ${currentModel} to free VRAM...`);
            await fetch('/api/unload_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: currentModel })
            }).catch(e => console.warn('[VOX] Unload failed:', e));
            
            updateModelDisplay();
        }
        
        // 3. Reset VOX unload flags and initialize VOX models
        window.sttManuallyUnloaded = false;
        window.ttsManuallyUnloaded = false;
        
        // Initialize VOX (loads Whisper STT + TTS)
        if (typeof VOX !== 'undefined' && typeof VOX.init === 'function') {
            VOX.init().catch(err => {
                console.error('[VOX] Initialization failed:', err);
            });
        }
        
        // Start interrupt monitoring
        if (window.VOXInterrupt) {
            window.VOXInterrupt.start();
            console.log('[VOX] Interrupt monitoring started');
        }
        
    } else if (tabId === 'chat') {
        // Stop interrupt monitoring
        if (window.VOXInterrupt) {
            window.VOXInterrupt.stop();
        }

        // Full teardown: stop all ops + unload STT/TTS/speech LLM
        if (typeof teardownVOX === 'function') {
            await teardownVOX().catch(e => console.warn('[CHAT] VOX teardown failed:', e));
        }

        if (currentModel) {
            updateModelDisplay();
        }
    }
    
    // Close mobile sidebar when switching tabs
    closeMobileSidebar();
}

// Mobile menu toggle
function toggleMobileMenu() {
    const sidebar = document.getElementById('left-sidebar');
    const isOpen = sidebar.classList.contains('mobile-open');
    
    sidebar.classList.toggle('mobile-open');
    
    // Properly toggle overlay class
    if (isOpen) {
        // Closing
        document.body.classList.remove('sidebar-open');
    } else {
        // Opening
        document.body.classList.add('sidebar-open');
    }
}

function closeMobileSidebar() {
    const sidebar = document.getElementById('left-sidebar');
    sidebar.classList.remove('mobile-open');
    document.body.classList.remove('sidebar-open');
}

// Chat list toggle for mobile
function toggleChatList() {
    const chatSidebar = document.querySelector('.chat-history-sidebar');
    const isOpen = chatSidebar.classList.contains('mobile-open');
    
    chatSidebar.classList.toggle('mobile-open');
    
    // Properly toggle overlay class
    if (isOpen) {
        // Closing
        document.body.classList.remove('chat-list-open');
    } else {
        // Opening
        document.body.classList.add('chat-list-open');
        // Load chat list if not already loaded
        loadChatList();
    }
}

function closeChatList() {
    const chatSidebar = document.querySelector('.chat-history-sidebar');
    chatSidebar.classList.remove('mobile-open');
    document.body.classList.remove('chat-list-open');
}

// --- FILE CONTEXT FOR LLM ---
function buildFileContextForLLM() {
    // Build a list of available files so LLM knows what the user has uploaded
    // This helps the LLM understand which file the user is referring to
    const context = {
        files: [],
        images: [],
        embedded_files: [],  // Files embedded in RAG
        total_count: 0
    };
    
    uploadPreviews.forEach(preview => {
        const fileInfo = {
            filename: preview.filename,
            type: preview.type,
            embedded: preview.embedded
        };
        
        if (preview.isImage) {
            context.images.push(fileInfo);
        } else if (preview.embedded) {
            context.embedded_files.push(fileInfo);
        } else {
            context.files.push(fileInfo);
        }
    });
    
    context.total_count = context.files.length + context.images.length + context.embedded_files.length;
    
    return context;
}

// Check if any uploads are still in progress
function hasUploadsInProgress() {
    return uploadPreviews.some(p => p.uploadProgress >= 0 && p.uploadProgress < 100);
}

// Wait for all uploads to complete
function waitForUploadsComplete() {
    return new Promise((resolve) => {
        const check = () => {
            if (!hasUploadsInProgress()) {
                resolve();
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    });
}

// --- CHAT FUNCTIONS ---
async function sendMessage() {
    const input = document.getElementById('chat-input');
        const message = input.value.trim(); 
        if (!message || isGenerating) return;
        // Check for uploads in progress
        if (hasUploadsInProgress()) {
            // Show indicator that we're waiting for uploads
            const sendBtn = document.getElementById('send-btn');
            sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            sendBtn.disabled = true;
            await waitForUploadsComplete();
            // Restore button
            sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
            sendBtn.disabled = false;
        }
    
    if (!message || isGenerating) return;
    
    // Check for uploads in progress
    if (hasUploadsInProgress()) {
        // Show indicator that we're waiting for uploads
        const sendBtn = document.getElementById('send-btn');
        sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        sendBtn.disabled = true;
        
        await waitForUploadsComplete();
        
        // Restore button
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        sendBtn.disabled = false;
    }
    
    isGenerating = true;
    input.value = '';
    input.style.height = 'auto';
    input.disabled = true; // Disable input during generation
    
    // Create new AbortController for this request
    currentAbortController = new AbortController();
    
    // Toggle send/stop buttons
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-inference-btn');
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';
    
    // Capture current attachments before clearing
    const currentAttachments = {
        files: [...uploadedFiles],
        images: [...uploadedImages]
    };
    
    // DEBUG: Check what's in attachments
    console.log('[DEBUG] currentAttachments images:', currentAttachments.images?.length, 
        currentAttachments.images?.map(i => ({
            filename: i.filename, 
            hasBase64: !!i.base64, 
            b64Type: typeof i.base64,
            b64Len: typeof i.base64 === 'string' ? i.base64.length : (Array.isArray(i.base64) ? i.base64.length + ' pages' : 'none')
        }))
    );
    
    // Add user message to UI with attachments
    appendMessage(message, 'user', null, currentAttachments);
    
    // Clear upload previews after adding to message (they now show on the message)
    uploadPreviews = [];
    renderUploadPreviews();
    
    // Create bot placeholder
    const botMsgId = 'bot-' + Date.now();
    createBotPlaceholder(botMsgId);
    
    // Build file context info for LLM to know what files are available
    const fileContext = buildFileContextForLLM();
    
    try {
        // DEBUG: Log what we're sending
        console.log('[SEND] uploadedImages count:', uploadedImages.length);
        console.log('[SEND] uploadedFiles count:', uploadedFiles.length);
        if (uploadedImages.length > 0) {
            uploadedImages.forEach((img, idx) => {
                const base64Info = Array.isArray(img.base64) 
                    ? `${img.base64.length} pages` 
                    : `${img.base64?.length || 0} chars`;
                console.log(`[SEND] Image ${idx}:`, img.filename, 'base64:', base64Info, 'isPdf:', img.isPdf || false);
            });
        }
        
        // Build attachment metadata for saving in chat history (no base64, just info)
        const attachmentsMeta = {
            images: uploadedImages.map(img => ({
                filename: img.filename,
                isPdf: img.isPdf || false,
                pages: img.pages || null
            })),
            files: uploadedFiles.map(f => ({
                filename: f.filename,
                file_type: f.file_type,
                pages: f.pages || null
            }))
        };
        
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
                uploaded_files: uploadedFiles,
                uploaded_images: uploadedImages,  // Include images for vision models
                file_context: fileContext,  // File names/info for LLM context
                attachments_meta: attachmentsMeta  // Metadata for saving in chat history
            }),
            signal: currentAbortController.signal // Add abort signal
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
                        // console.log('SSE data received:', data);  // DEBUG VERY NOISY
                        updateBotMessage(botMsgId, data);
                        
                        if (data.session_id) {
                            const wasNewChat = !currentSessionId;
                            currentSessionId = data.session_id;
                            console.log('[CHAT] Session ID received:', data.session_id, 'wasNewChat:', wasNewChat);
                            
                            // For NEW chats: immediately refresh list so it appears
                            // The chat is already saved with placeholder title by the backend
                            if (wasNewChat) {
                                console.log('[CHAT] New chat - refreshing list immediately...');
                                loadChatList().then(() => selectChatInList(data.session_id));
                            }
                        }
                        
                        // Check if stream is done - this is when chat is DEFINITELY saved with final title
                        if (data.done) {
                            resetButtonsAfterInference();
                            
                            // Refresh chat list to show updated chat at top with proper title
                            // Both new AND existing chats get refreshed here
                            console.log('[CHAT] Stream done - final refresh...');
                            await loadChatList();
                            selectChatInList(currentSessionId);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
        
        // Clear uploaded files and images after successful send
        uploadedFiles = [];
        uploadedImages = [];
        uploadPreviews = [];  // Clear the new unified previews
        updateUploadedImagesDisplay();
        renderUploadPreviews();
        
        // Ensure buttons reset even if done flag wasn't received
        resetButtonsAfterInference();
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Request was aborted');
            updateBotMessage(botMsgId, { content: ' [STOPPED]', done: true });
        } else {
            console.error('Error sending message:', error);
            updateBotMessage(botMsgId, { error: 'Failed to get response' });
        }
        resetButtonsAfterInference();
    } finally {
        currentAbortController = null;
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
        // console.log('[THINKING] Received thinking data:', data.thinking.substring(0, 50));
        // Show thinking section
        thinkingSection.style.display = 'block';
        
        // Get or create current thinking chunk
        // After a tool call, we need a NEW thinking chunk to maintain chronological order
        let thinkingText = thinkingContent._currentThinkingChunk;
        
        // If no current chunk OR last inserted was a tool, create new chunk
        const lastInserted = thinkingContent._lastInsertPoint;
        const needNewChunk = !thinkingText || (lastInserted && lastInserted.classList && lastInserted.classList.contains('tool-call-container'));
        
        if (needNewChunk) {
            thinkingText = document.createElement('div');
            thinkingText.className = 'thinking-text';
            
            // Insert after last insertion point (maintains chronological order)
            if (lastInserted && lastInserted.nextSibling) {
                thinkingContent.insertBefore(thinkingText, lastInserted.nextSibling);
            } else {
                thinkingContent.appendChild(thinkingText);
            }
            
            // Update tracking
            thinkingContent._currentThinkingChunk = thinkingText;
            thinkingContent._lastInsertPoint = thinkingText;
            
            // Add auto-scroll behavior (only once)
            if (!thinkingContent._scrollSetup) {
                thinkingContent._scrollSetup = true;
                thinkingContent._autoScroll = true;  // Start with autoscroll enabled
                
                thinkingContent.addEventListener('scroll', () => {
                    // Check if user scrolled away from bottom
                    const isNearBottom = thinkingContent.scrollHeight - thinkingContent.scrollTop - thinkingContent.clientHeight < 100;
                    thinkingContent._autoScroll = isNearBottom;
                });
            }
        }
        
        thinkingText.textContent += data.thinking;
        
        // Auto-scroll thinking content - always scroll if autoscroll is enabled
        if (thinkingContent._autoScroll !== false) {
            requestAnimationFrame(() => {
                thinkingContent.scrollTop = thinkingContent.scrollHeight;
            });
        }
    }
    
    if (data.tool_status || data.tool || data.tool_name || data.tool_args || data.tool_result) {
        const toolMsg = data.tool_status || data.tool || (data.tool_name ? `🔧 ${data.tool_name.replace('potatool_', '').replace(/_/g, ' ')}` : 'Tool executing...');
        // console.log('[Tool]', toolMsg, data);
        
        // Show tool activity in Thinking & Tools section
        thinkingSection.style.display = 'block';
        
        // Try to find last tool container to update it, or create new one
        let toolContainer = thinkingContent._lastToolContainer;
        let detailDiv, toolId;
        
        // Check if we should update existing container or create new one
        // Update if: same tool name, or if last container still has placeholder
        const shouldUpdate = toolContainer && (
            (data.tool_name && toolContainer.dataset.toolName === data.tool_name) ||
            toolContainer.querySelector('.tool-detail-line')?.textContent?.includes('Loading details...')
        );
        
        if (shouldUpdate) {
            // Update existing container
            console.log('[Tool] Updating existing tool container');
            detailDiv = toolContainer.querySelector('.tool-detail-section');
            toolId = detailDiv.id.replace('-detail', '');
        } else {
            // Create new expandable tool status container
            console.log('[Tool] Creating new tool container');
            toolContainer = document.createElement('div');
            toolContainer.className = 'tool-call-container';
            if (data.tool_name) toolContainer.dataset.toolName = data.tool_name;
            
            // Create tool status line (clickable header)
            const toolDiv = document.createElement('div');
            toolDiv.className = 'tool-status-line';
            toolId = `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            
            // Add chevron and message
            toolDiv.innerHTML = `
                <span class="tool-toggle" id="${toolId}-toggle">
                    <i class="fas fa-chevron-right"></i>
                </span>
                <span class="tool-message">${toolMsg}</span>
            `;
            
            // Create detail section (collapsed by default)
            detailDiv = document.createElement('div');
            detailDiv.className = 'tool-detail-section collapsed';
            detailDiv.id = `${toolId}-detail`;
            
            toolContainer.appendChild(toolDiv);
            toolContainer.appendChild(detailDiv);
        }
        
        // Build/update details
        let detailHTML = '';
        
        if (data.tool_name) {
            detailHTML += `<div class="tool-detail-line"><strong>Function:</strong> ${data.tool_name}</div>`;
        }
        
        if (data.tool_args) {
            // Parse and display in field: value format
            try {
                const argsObj = typeof data.tool_args === 'string' ? JSON.parse(data.tool_args) : data.tool_args;
                Object.entries(argsObj).forEach(([key, value]) => {
                    detailHTML += `<div class="tool-detail-line"><strong>${key}:</strong> ${value}</div>`;
                });
            } catch (e) {
                // Fallback to JSON
                const argsStr = typeof data.tool_args === 'string' ? data.tool_args : JSON.stringify(data.tool_args, null, 2);
                detailHTML += `<div class="tool-detail-line"><strong>Parameters:</strong></div>`;
                detailHTML += `<pre class="tool-args-pre">${argsStr}</pre>`;
            }
        }
        
        if (data.tool_result) {
            // Format result as JSON
            const resultStr = typeof data.tool_result === 'string' ? data.tool_result : JSON.stringify(data.tool_result, null, 2);
            detailHTML += `<div class="tool-detail-line"><strong>Result:</strong></div>`;
            detailHTML += `<pre class="tool-result-pre">${resultStr}</pre>`;
        }
        
        // Only use placeholder if this is a new container with no details
        if (!detailHTML && !shouldUpdate) {
            detailHTML = `<div class="tool-detail-line" style="color: #888; font-style: italic;">Loading details...</div>`;
        }
        
        if (detailHTML) {
            detailDiv.innerHTML = detailHTML;
        }
        
        // If this is a new container (not updating), add click handler and insert into DOM
        if (!shouldUpdate) {
            // Make entire container clickable to toggle details
            toolContainer.style.cursor = 'pointer';
            toolContainer.onclick = (e) => {
                console.log('[Tool Click] Container clicked, toolId:', toolId);
                
                const details = document.getElementById(`${toolId}-detail`);
                const toggle = document.getElementById(`${toolId}-toggle`);
                
                if (details && toggle) {
                    if (details.classList.contains('collapsed')) {
                        console.log('[Tool Click] Expanding...');
                        details.classList.remove('collapsed');
                        toggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
                    } else {
                        console.log('[Tool Click] Collapsing...');
                        details.classList.add('collapsed');
                        toggle.innerHTML = '<i class="fas fa-chevron-right"></i>';
                    }
                }
            };
            
            // CRITICAL: Maintain chronological order - insert after last insertion point
            const lastPoint = thinkingContent._lastInsertPoint || thinkingContent.lastChild;
            if (lastPoint && lastPoint.nextSibling) {
                thinkingContent.insertBefore(toolContainer, lastPoint.nextSibling);
            } else {
                thinkingContent.appendChild(toolContainer);
            }
            
            // Update last insertion point to this tool container
            thinkingContent._lastInsertPoint = toolContainer;
            thinkingContent._lastToolContainer = toolContainer;
            // Clear current thinking chunk so next thinking creates a new one
            thinkingContent._currentThinkingChunk = null;
        }
        
        // Only auto-scroll if user hasn't manually scrolled up
        const isNearBottom = thinkingContent.scrollHeight - thinkingContent.scrollTop - thinkingContent.clientHeight < 20;
        if (isNearBottom) {
            setTimeout(() => {
                thinkingContent.scrollTop = thinkingContent.scrollHeight;
            }, 0);
        }
    }
    
    if (data.content) {
        // Filter out HTML-formatted reasoning that should be hidden
        // Detect patterns like: <ol start="403"><li>Let's try...</li></ol>
        const isHTMLReasoning = /^<ol[^>]*>.*?<li>.*?(Let's|We need|We should|We must|We can|We have|Better to|Actually|Wait|Looking at)/i.test(data.content);
        
        if (isHTMLReasoning) {
            // This is reasoning disguised as content - move to thinking instead
            console.log('[FILTER] Detected HTML reasoning in content, moving to thinking');
            // Optionally add to thinking section if needed, or just skip it
            // For now, we'll just not render it as content
            return;
        }
        
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
        // Normalize common LaTeX patterns so KaTeX auto-render can detect them.
        // Wrap \begin{...}...\end{...} in display math and convert
        // parenthesized fragments that contain backslash-commands into inline math.
        function normalizeLatex(src) {
            if (!src || typeof src !== 'string') return src;
            let s = src;

            // STEP 0: Shield code blocks so LaTeX patterns never touch their content.
            // math/latex labeled fenced blocks are left alone here so STEP 2 can convert them.
            const codeBlocks = [];
            let codeIdx = 0;
            // Completed fenced code blocks (``` ... ```) - skip math/latex labeled ones
            s = s.replace(/(`{3,})(?!math\b|latex\b)([\s\S]*?)\1/g, (match) => {
                const key = `\x00CODE${codeIdx}\x00`;
                codeBlocks[codeIdx] = match;
                codeIdx++;
                return key;
            });
            // Unclosed fenced code block (streaming: opening ``` without closing ```)
            // Only match if the fence starts at the beginning of a line
            s = s.replace(/(^|\n)(`{3,})(?!math\b|latex\b)([\s\S]*)$/, (match) => {
                const key = `\x00CODE${codeIdx}\x00`;
                codeBlocks[codeIdx] = match;
                codeIdx++;
                return key;
            });
            // Inline code (`...`)
            s = s.replace(/(`[^`\n]+`)/g, (match) => {
                const key = `\x00CODE${codeIdx}\x00`;
                codeBlocks[codeIdx] = match;
                codeIdx++;
                return key;
            });

            // STEP 1: Protect existing LaTeX delimiters from markdown parser
            // Store them in a map that we'll return
            const protectedBlocks = [];
            let blockIndex = 0;

            // Protect $$...$$ display math blocks first (before single $)
            s = s.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                protectedBlocks[blockIndex] = match;
                blockIndex++;
                return placeholder;
            });

            // Protect \[...\] display math blocks - convert to $$
            s = s.replace(/\\\[([\s\S]*?)\\\]/g, (match, inner) => {
                const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                protectedBlocks[blockIndex] = '$$' + inner + '$$';
                blockIndex++;
                return placeholder;
            });

            // Protect inline math $...$ (but be more careful - only if it looks like math)
            s = s.replace(/\$([^\$\n]+?)\$/g, (match, inner) => {
                // Only protect if it contains LaTeX commands or math symbols
                if (/\\[a-zA-Z]+|[\^_{}]|\\[()[\]]/.test(inner)) {
                    const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                    protectedBlocks[blockIndex] = match;
                    blockIndex++;
                    return placeholder;
                }
                return match;
            });

            // Protect \(...\) inline math blocks
            s = s.replace(/\\\(([\s\S]*?)\\\)/g, (match) => {
                const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                protectedBlocks[blockIndex] = match;
                blockIndex++;
                return placeholder;
            });

            // STEP 2: Convert fenced code blocks labeled as math/latex into display math
            s = s.replace(/```(?:math|latex)\n([\s\S]*?)\n```/g, (m, inner) => {
                const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                protectedBlocks[blockIndex] = '\n\n$$\n' + inner.trim() + '\n$$\n\n';
                blockIndex++;
                return placeholder;
            });

            // STEP 3: Wrap \begin{...}...\end{...} in $$...$$ if not already delimited
            s = s.replace(/\\begin\{[\s\S]*?\\end\{[^}]+\}/g, (m) => {
                // Check if already protected or delimited
                if (m.includes('data-latex-block') || /\$\$/.test(m)) return m;
                const placeholder = `<span data-latex-block="${blockIndex}"></span>`;
                protectedBlocks[blockIndex] = '$$' + m + '$$';
                blockIndex++;
                return placeholder;
            });

            // Restore shielded code blocks
            for (let ci = 0; ci < codeBlocks.length; ci++) {
                s = s.split(`\x00CODE${ci}\x00`).join(codeBlocks[ci]);
            }

            // Return both the normalized text and the protected blocks
            return { text: s, protectedBlocks };
        }

        const { text: normalized, protectedBlocks } = normalizeLatex(text);
        let html = marked.parse(normalized);

        // Restore protected LaTeX blocks after markdown parsing
        if (protectedBlocks && protectedBlocks.length > 0) {
            protectedBlocks.forEach((block, index) => {
                const spanPlaceholder = `<span data-latex-block="${index}"></span>`;
                html = html.split(spanPlaceholder).join(block);
            });
        }

        element.innerHTML = html;

        // Post-process: some models emit math inside code blocks. KaTeX auto-render
        // ignores code/pre tags. If a code block contains only math, unwrap it so
        // renderMathInElement can process it. This keeps code blocks that are code intact.
        (function unwrapMathCodeBlocks(root) {
            try {
                const pres = root.querySelectorAll('pre');
                pres.forEach(pre => {
                    const code = pre.querySelector('code');
                    if (!code) return;
                    const txt = code.textContent.trim();

                    // Enhanced heuristics: detect LaTeX more accurately
                    const hasLatexEnv = /\\begin\{[a-zA-Z*]+\}/.test(txt) || /\\end\{[a-zA-Z*]+\}/.test(txt);
                    const hasLatexDelim = txt.includes('$$') || txt.includes('\\[') || txt.includes('\\]') ||
                                         txt.includes('\\(') || txt.includes('\\)');
                    const hasLatexCommands = /\\(?:frac|sqrt|sum|int|prod|lim|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|nabla|partial|cdot|times|div|pm|leq|geq|neq|approx|equiv|sin|cos|tan|log|ln|exp|matrix|begin|end|text|mathbb|mathcal|boldsymbol|vec|hat|bar|tilde|dot|left|right|big|Big|[a-zA-Z]+)\{?/.test(txt);
                    const wrappedInMathDelim = /^\$\$[\s\S]+\$\$$/.test(txt) || /^\\\[[\s\S]+\\\]$/.test(txt) ||
                                               /^\$[^\$]+\$$/.test(txt) || /^\\\([\s\S]+\\\)$/.test(txt);

                    const looksLikeMath = hasLatexEnv || hasLatexDelim || hasLatexCommands || wrappedInMathDelim;

                    // Additional check: if it contains typical code patterns, don't unwrap
                    const looksLikeCode = /^(function|class|const|let|var|def|import|export|return|if|else|for|while)\s/.test(txt) ||
                                         /[;{}()]\s*$/.test(txt) ||
                                         txt.split('\n').length > 3 && /^[\s]+(function|const|let|var|if|for|while|def|class)/.test(txt);

                    if (looksLikeMath && !looksLikeCode) {
                        // Replace the <pre> with a div that contains the raw math text so KaTeX can find it
                        const container = document.createElement('div');
                        container.className = 'math-block-unwrapped';
                        // Preserve spacing/newlines
                        container.textContent = txt;
                        pre.parentElement.replaceChild(container, pre);
                    }
                });

                // Also check inline code spans that contain math delimiters and unwrap
                const codes = root.querySelectorAll('code');
                codes.forEach(code => {
                    // Skip code inside pre (already handled)
                    if (code.parentElement && code.parentElement.tagName.toLowerCase() === 'pre') return;
                    const txt = code.textContent.trim();

                    // Unwrap if it's clearly math delimited
                    const isMathDelimited = (txt.startsWith('$$') && txt.endsWith('$$')) ||
                                           (txt.startsWith('\\[') && txt.endsWith('\\]')) ||
                                           (txt.startsWith('\\(') && txt.endsWith('\\)')) ||
                                           (txt.startsWith('$') && txt.endsWith('$') && /\\[a-zA-Z]+/.test(txt));

                    if (isMathDelimited) {
                        const span = document.createElement('span');
                        span.className = 'math-inline-unwrapped';
                        span.textContent = txt;
                        code.parentElement.replaceChild(span, code);
                    }
                });
            } catch (e) {
                console.warn('unwrapMathCodeBlocks error', e);
            }
        })(element);
        
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
        
        // Render LaTeX math with KaTeX
        renderLatex(element);
    } catch (e) {
        console.error('Error rendering markdown:', e);
        element.textContent = text;
    }
}

/**
 * Render LaTeX math expressions using KaTeX
 * Supports: $inline$ and $$block$$ notation
 */
function renderLatex(element) {
    if (typeof renderMathInElement === 'undefined') {
        // KaTeX auto-render not loaded yet, try again later
        setTimeout(() => {
            if (typeof renderMathInElement !== 'undefined') {
                renderLatex(element);
            }
        }, 100);
        return;
    }
    
    try {
        renderMathInElement(element, {
            // Process longer delimiters FIRST to avoid greedy matching issues
            delimiters: [
                {left: '$$', right: '$$', display: true},    // Block math (must come before single $)
                {left: '\\[', right: '\\]', display: true},  // Block math alt
                {left: '\\(', right: '\\)', display: false}, // Inline math alt
                {left: '$', right: '$', display: false}      // Inline math (last to avoid conflicts)
            ],
            throwOnError: false,
            errorColor: '#ff6b6b',
            strict: false,
            trust: true,
            fleqn: false,
            // Don't process inside code blocks (but our unwrapMathCodeBlocks handles math in code)
            ignoredTags: ['script', 'noscript', 'style', 'textarea'],
            ignoredClasses: ['no-latex'],
            // Preprocess to handle edge cases
            preProcess: (text) => {
                // Remove any HTML entities that might break delimiters
                return text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
            }
        });
    } catch (e) {
        console.warn('LaTeX rendering error:', e);
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

function appendMessage(text, type, modelName = null, attachments = null) {
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
    } else if (type === 'user') {
        // User message with optional attachments
        let attachmentsHtml = '';
        
        if (attachments && (attachments.files?.length > 0 || attachments.images?.length > 0)) {
            attachmentsHtml = '<div class="message-attachments">';
            
            // Add images (but NOT PDFs - those go in files section)
            if (attachments.images?.length > 0) {
                for (const img of attachments.images) {
                    // Check if this is a PDF (has isPdf flag or base64 is an array)
                    const isPdf = img.isPdf || Array.isArray(img.base64);
                    
                    if (isPdf) {
                        // PDF - show as file icon, not image
                        const pages = img.pages || (Array.isArray(img.base64) ? img.base64.length : '?');
                        // Use currentSessionId for viewing (it should be set by now, or will be set when chat starts)
                        attachmentsHtml += `
                            <div class="attachment-item attachment-pdf" title="Click to view PDF" onclick="openPdfViewer(currentSessionId, '${img.filename}')">
                                <i class="fas fa-file-pdf"></i>
                                <span class="attachment-name">${img.filename} (${pages} pages)</span>
                            </div>
                        `;
                    } else if (img.base64 && typeof img.base64 === 'string') {
                        // Regular image - show thumbnail
                        const imgId = `attach-img-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
                        if (!window.attachedImagesMap) window.attachedImagesMap = {};
                        window.attachedImagesMap[imgId] = img.base64;
                        
                        attachmentsHtml += `
                            <div class="attachment-item attachment-image">
                                <img id="${imgId}" 
                                     src="data:image/png;base64,${img.base64}" 
                                     alt="${img.filename}" 
                                     title="Click to view: ${img.filename}"
                                     onclick="showImageFromMap('${imgId}', '${img.filename}')">
                                <span class="attachment-name">${img.filename}</span>
                            </div>
                        `;
                    } else {
                        // No base64, show placeholder
                        attachmentsHtml += `
                            <div class="attachment-item">
                                <i class="fas fa-image"></i>
                                <span class="attachment-name">${img.filename}</span>
                            </div>
                        `;
                    }
                }
            }
            
            // Add files (PDFs, text, etc.) - but skip PDFs already shown from images
            if (attachments.files?.length > 0) {
                const pdfFilenames = (attachments.images || [])
                    .filter(img => img.isPdf || Array.isArray(img.base64))
                    .map(img => img.filename);
                
                for (const file of attachments.files) {
                    // Skip if this PDF was already shown
                    if (pdfFilenames.includes(file.filename)) continue;
                    
                    const isPdf = file.file_type === 'pdf';
                    const icon = isPdf ? 'fa-file-pdf' : 
                                 file.file_type === 'image_description' ? 'fa-image' : 'fa-file-code';
                    const pages = file.pages ? ` (${file.pages} pages)` : '';
                    
                    // Store content for later viewing (for live uploads)
                    const fileContentId = `filecontent-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
                    if (!window.uploadedFilesContentMap) window.uploadedFilesContentMap = {};
                    if (file.content) {
                        window.uploadedFilesContentMap[fileContentId] = file.content;
                    }
                    
                    // Make clickable - use cached content if available, otherwise fetch from server
                    if (isPdf) {
                        // PDF - open PDF viewer
                        attachmentsHtml += `
                            <div class="attachment-item attachment-pdf" onclick="openPdfViewer(currentSessionId, '${file.filename}')" title="Click to view PDF">
                                <i class="fas ${icon}"></i>
                                <span class="attachment-name">${file.filename}${pages}</span>
                            </div>
                        `;
                    } else {
                        // Text file - open file viewer with cached content
                        attachmentsHtml += `
                            <div class="attachment-item attachment-file" onclick="openFileViewerCached('${fileContentId}', '${file.filename}')" title="Click to view file content">
                                <i class="fas ${icon}"></i>
                                <span class="attachment-name">${file.filename}${pages}</span>
                            </div>
                        `;
                    }
                }
            }
            
            attachmentsHtml += '</div>';
        }
        
        msgDiv.innerHTML = attachmentsHtml + `<div class="user-text">${escapeHtml(text)}</div>`;
        chatWindow.appendChild(msgDiv);
    } else {
        msgDiv.textContent = text;
        chatWindow.appendChild(msgDiv);
    }
    
    scrollToBottom();
    
    return msgId; // Return ID for later reference
}

// Helper to escape HTML in user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Truncate filename for display
function truncateFilename(filename, maxLen = 20) {
    if (filename.length <= maxLen) return filename;
    const ext = filename.split('.').pop();
    const name = filename.slice(0, -(ext.length + 1));
    const truncatedName = name.slice(0, maxLen - ext.length - 4) + '...';
    return truncatedName + '.' + ext;
}

// Append user message with saved attachments from chat history
function appendMessageWithSavedAttachments(text, attachments, chatId) {
    const chatWindow = document.getElementById('chat-window');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg user';
    
    const msgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    msgDiv.id = msgId;
    
    let attachmentsHtml = '<div class="message-attachments">';
    
    // Add images (including PDFs which are stored as images)
    if (attachments.images?.length > 0) {
        for (const img of attachments.images) {
            if (img.isPdf) {
                // PDF - show file icon with click to open viewer
                const pages = img.pages ? ` (${img.pages} pages)` : '';
                attachmentsHtml += `
                    <div class="attachment-item attachment-pdf" onclick="openPdfViewer('${chatId}', '${img.filename}')" title="Click to view PDF">
                        <i class="fas fa-file-pdf"></i>
                        <span class="attachment-name">${truncateFilename(img.filename)}${pages}</span>
                    </div>
                `;
            } else {
                // Regular image - show thumbnail placeholder that loads lazily
                const thumbId = `thumb-${msgId}-${Math.random().toString(36).substr(2, 5)}`;
                attachmentsHtml += `
                    <div class="attachment-item attachment-image" onclick="loadAndShowImage('${chatId}', '${img.filename}')" title="Click to view: ${img.filename}">
                        <div class="attachment-thumb-lazy" id="${thumbId}" data-chat-id="${chatId}" data-filename="${img.filename}">
                            <i class="fas fa-image"></i>
                        </div>
                        <span class="attachment-name">${truncateFilename(img.filename)}</span>
                    </div>
                `;
            }
        }
    }
    
    // Add text files - make them clickable to view content
    if (attachments.files?.length > 0) {
        for (const file of attachments.files) {
            // Skip PDFs (they're shown in images section if they have pages)
            if (file.file_type === 'pdf') continue;
            
            const icon = file.file_type === 'image_description' ? 'fa-image' : 'fa-file-code';
            const pages = file.pages ? ` (${file.pages} pages)` : '';
            // Make file clickable to open content viewer
            attachmentsHtml += `
                <div class="attachment-item attachment-file" onclick="openFileViewer('${chatId}', '${file.filename}')" title="Click to view file content">
                    <i class="fas ${icon}"></i>
                    <span class="attachment-name">${truncateFilename(file.filename)}${pages}</span>
                </div>
            `;
        }
    }
    
    attachmentsHtml += '</div>';
    
    // Clean up the message content - remove the context prefixes that were added for the LLM
    let cleanText = text;
    // Remove "=== Available Files ===" section
    cleanText = cleanText.replace(/\n*=== Available Files ===[\s\S]*?When the user refers to a specific file, use the content from that file\.\n*/g, '');
    // Remove "[Attached: ...]" prefix
    cleanText = cleanText.replace(/^\[Attached:[\s\S]*?\]\n*The following images are pages from the attached document\(s\)\. Please analyze them carefully to answer the user's question\.\n*/g, '');
    cleanText = cleanText.replace(/^\[Attached:[\s\S]*?\]\n*/g, '');
    // Remove "=== Uploaded Files Content ===" section
    cleanText = cleanText.replace(/\n*=== Uploaded Files Content ===[\s\S]*?(?=\n\n|$)/g, '');
    // Remove image descriptions section (for non-VL model reloads)
    cleanText = cleanText.replace(/^\[The following images were described for you since you cannot see images directly\][\s\S]*?=== User Question ===\n*/g, '');
    // Clean up any leading/trailing whitespace
    cleanText = cleanText.trim();
    
    msgDiv.innerHTML = attachmentsHtml + `<div class="user-text">${escapeHtml(cleanText)}</div>`;
    chatWindow.appendChild(msgDiv);
    
    // Lazy load thumbnails for images
    setTimeout(() => lazyLoadThumbnails(msgDiv), 100);
    
    return msgId;
}

// Lazy load image thumbnails from server
async function lazyLoadThumbnails(containerEl) {
    const thumbs = containerEl.querySelectorAll('.attachment-thumb-lazy');
    for (const thumb of thumbs) {
        const chatId = thumb.dataset.chatId;
        const filename = thumb.dataset.filename;
        
        try {
            const response = await fetch(`/api/get_uploaded_file?chat_id=${encodeURIComponent(chatId)}&filename=${encodeURIComponent(filename)}&thumbnail=true`);
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                thumb.innerHTML = `<img src="${url}" alt="${filename}" style="max-width: 100%; max-height: 100%; object-fit: cover;">`;
            }
        } catch (e) {
            console.warn('Could not load thumbnail:', filename);
        }
    }
}

// Load image from uploads folder and show fullscreen
async function loadAndShowImage(chatId, filename) {
    try {
        const response = await fetch(`/api/get_uploaded_file?chat_id=${encodeURIComponent(chatId)}&filename=${encodeURIComponent(filename)}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showImageFullscreen(url, filename);
        } else {
            console.error('Failed to load image:', response.status);
            alert('Could not load image. It may have been deleted.');
        }
    } catch (e) {
        console.error('Error loading image:', e);
        alert('Error loading image');
    }
}

// Open PDF viewer modal
async function openPdfViewer(chatId, filename) {
    // Create or get modal
    let modal = document.getElementById('pdf-viewer-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'pdf-viewer-modal';
        modal.className = 'pdf-viewer-modal';
        modal.innerHTML = `
            <div class="pdf-viewer-content">
                <div class="pdf-viewer-header">
                    <span class="pdf-viewer-title"></span>
                    <div class="pdf-viewer-actions">
                        <a class="pdf-viewer-download" title="Open in new tab" target="_blank">
                            <i class="fas fa-external-link-alt"></i>
                        </a>
                        <span class="pdf-viewer-close" onclick="closePdfViewer()">&times;</span>
                    </div>
                </div>
                <div class="pdf-viewer-body">
                    <div class="pdf-viewer-loading"><i class="fas fa-spinner fa-spin"></i> Loading PDF...</div>
                    <object class="pdf-viewer-object" type="application/pdf" style="display:none;">
                        <p>Your browser cannot display PDFs. <a class="pdf-fallback-link" target="_blank">Click here to download</a>.</p>
                    </object>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closePdfViewer();
        });
        
        // Close on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'flex') {
                closePdfViewer();
            }
        });
    }
    
    const title = modal.querySelector('.pdf-viewer-title');
    const loading = modal.querySelector('.pdf-viewer-loading');
    const pdfObject = modal.querySelector('.pdf-viewer-object');
    const downloadLink = modal.querySelector('.pdf-viewer-download');
    const fallbackLink = modal.querySelector('.pdf-fallback-link');
    
    // Build the URL
    const url = `/api/get_uploaded_file?chat_id=${encodeURIComponent(chatId)}&filename=${encodeURIComponent(filename)}`;
    
    title.textContent = filename;
    loading.style.display = 'flex';
    pdfObject.style.display = 'none';
    downloadLink.href = url;
    fallbackLink.href = url;
    modal.style.display = 'flex';
    
    // Set the PDF source
    pdfObject.data = url;
    
    // Check if loaded after a short delay
    setTimeout(() => {
        loading.style.display = 'none';
        pdfObject.style.display = 'block';
    }, 500);
}

function closePdfViewer() {
    const modal = document.getElementById('pdf-viewer-modal');
    if (modal) {
        modal.style.display = 'none';
        const pdfObject = modal.querySelector('.pdf-viewer-object');
        if (pdfObject) pdfObject.data = '';
    }
}

// ===============================================
// FILE CONTENT VIEWER MODAL
// ===============================================

// Store current file content for copy functionality
let currentFileContent = '';

// Open file content viewer modal
async function openFileViewer(chatId, filename, cachedContent = null) {
    const modal = document.getElementById('file-viewer-modal');
    if (!modal) return;
    
    const title = modal.querySelector('.file-viewer-title');
    const loading = modal.querySelector('.file-viewer-loading');
    const codeEl = modal.querySelector('.file-viewer-code');
    
    title.textContent = filename;
    loading.style.display = 'flex';
    codeEl.style.display = 'none';
    codeEl.textContent = '';
    currentFileContent = '';
    modal.style.display = 'flex';
    
    // If we have cached content, use it
    if (cachedContent) {
        loading.style.display = 'none';
        codeEl.style.display = 'block';
        codeEl.textContent = cachedContent;
        currentFileContent = cachedContent;
        return;
    }
    
    // Otherwise fetch from server
    try {
        const response = await fetch(`/api/get_uploaded_file?chat_id=${encodeURIComponent(chatId)}&filename=${encodeURIComponent(filename)}&as_text=true`);
        
        if (response.ok) {
            const text = await response.text();
            loading.style.display = 'none';
            codeEl.style.display = 'block';
            codeEl.textContent = text;
            currentFileContent = text;
        } else {
            loading.style.display = 'none';
            codeEl.style.display = 'block';
            codeEl.textContent = `Error loading file: ${response.status}`;
            currentFileContent = '';
        }
    } catch (e) {
        loading.style.display = 'none';
        codeEl.style.display = 'block';
        codeEl.textContent = `Error loading file: ${e.message}`;
        currentFileContent = '';
    }
}

function closeFileViewer() {
    const modal = document.getElementById('file-viewer-modal');
    if (modal) {
        modal.style.display = 'none';
        currentFileContent = '';
    }
}

// Open file viewer with cached content (for live uploads where content is in memory)
function openFileViewerCached(contentId, filename) {
    if (window.uploadedFilesContentMap && window.uploadedFilesContentMap[contentId]) {
        openFileViewer(null, filename, window.uploadedFilesContentMap[contentId]);
    } else {
        // Fallback to fetching from server
        openFileViewer(currentSessionId, filename);
    }
}

function copyFileContent() {
    if (currentFileContent) {
        navigator.clipboard.writeText(currentFileContent).then(() => {
            const btn = document.querySelector('.file-viewer-copy');
            if (btn) {
                const originalHtml = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                btn.style.background = 'var(--secondary-color)';
                btn.style.borderColor = 'var(--secondary-color)';
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.background = '';
                    btn.style.borderColor = '';
                }, 2000);
            }
        });
    }
}

// Close file viewer on Escape key and backdrop click
document.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeFileViewer();
        }
    });
    
    const modal = document.getElementById('file-viewer-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeFileViewer();
        });
    }
});

// Show image in fullscreen modal
function showImageFullscreen(src, filename) {
    // Create modal if doesn't exist
    let modal = document.getElementById('image-fullscreen-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'image-fullscreen-modal';
        modal.className = 'image-fullscreen-modal';
        modal.innerHTML = `
            <div class="image-fullscreen-content">
                <span class="image-fullscreen-close" onclick="closeImageFullscreen()">&times;</span>
                <img id="fullscreen-image" src="" alt="">
                <div id="fullscreen-caption"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on click outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeImageFullscreen();
        });
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeImageFullscreen();
        });
    }
    
    document.getElementById('fullscreen-image').src = src;
    document.getElementById('fullscreen-caption').textContent = filename;
    modal.style.display = 'flex';
}

// Show image from stored map (for large base64 images)
function showImageFromMap(imgId, filename) {
    if (window.attachedImagesMap && window.attachedImagesMap[imgId]) {
        showImageFullscreen('data:image/png;base64,' + window.attachedImagesMap[imgId], filename);
    }
}

function closeImageFullscreen() {
    const modal = document.getElementById('image-fullscreen-modal');
    if (modal) modal.style.display = 'none';
}

async function stopInference() {
    try {
        console.log('[STOP] Stopping inference...');
        
        // Prevent new messages while stopping
        isGenerating = true; // Keep it true until fully stopped
        
        // Abort the fetch stream FIRST
        if (currentAbortController) {
            currentAbortController.abort();
            console.log('[STOP] Aborted fetch stream');
        }
        
        // Tell backend to stop (this sets stop flag and unloads model)
        await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        
        console.log('[STOP] Backend notified, waiting for cleanup...');
        
        // Wait for backend to finish stopping
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        console.log('[STOP] Stop complete');
        appendMessage('Inference stopped', 'system');
        
    } catch (error) {
        console.error('Error stopping inference:', error);
    } finally {
        // Now it's safe to allow new messages
        isGenerating = false;
        resetButtonsAfterInference();
    }
}

async function unloadChatModel() {
    try {
        console.log('[UNLOAD] Unloading model...');
        
        // If currently generating, abort and stop first
        if (isGenerating) {
            console.log('[UNLOAD] Generation active, stopping first...');
            
            // Abort fetch
            if (currentAbortController) {
                currentAbortController.abort();
            }
            
            // Call stop endpoint
            await fetch('/api/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: currentModel })
            });
            
            // Wait for stop to complete
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            isGenerating = false;
            resetButtonsAfterInference();
            
            console.log('[UNLOAD] Generation stopped, now unloading...');
        }
        
        document.getElementById('current-model-display').textContent = 'Unloading model...';
        
        // Now unload the model
        const response = await fetch('/api/unload_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: currentModel })
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('[UNLOAD] Model unloaded successfully');
            appendMessage(`Model ${currentModel} unloaded from VRAM`, 'system');
        } else {
            console.error('[UNLOAD] Error:', result.error);
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
            const li = document.createElement('li');
            li.dataset.chatId = chat.id;  // Store chat ID on the li element
            
            const titleSpan = document.createElement('span');
            titleSpan.textContent = chat.title;
            titleSpan.onclick = () => {
                loadSession(chat.id);
                // Close chat list on mobile after selection
                if (window.innerWidth <= 768) {
                    closeChatList();
                }
            };
            titleSpan.style.flex = '1';
            titleSpan.style.cursor = 'pointer';
            
            const deleteIcon = document.createElement('i');
            deleteIcon.className = 'fas fa-trash delete-chat-icon';
            deleteIcon.title = 'Delete chat';
            deleteIcon.style.color = '#ff4757';
            deleteIcon.style.cursor = 'pointer';
            deleteIcon.style.marginLeft = 'auto';
            deleteIcon.style.padding = '5px';
            deleteIcon.dataset.chatId = chat.id;
            deleteIcon.dataset.chatTitle = chat.title;
            
            li.appendChild(titleSpan);
            li.appendChild(deleteIcon);
            
            // Check if this is the current session
            if (chat.id === currentSessionId) {
                li.classList.add('active');
                console.log('[CHAT LIST] Marked active:', chat.id);
            }
            newList.appendChild(li);
        });
        
        // Double-check: if currentSessionId is set but no item is active, find and activate it
        if (currentSessionId && !newList.querySelector('li.active')) {
            const targetLi = newList.querySelector(`li[data-chat-id="${currentSessionId}"]`);
            if (targetLi) {
                targetLi.classList.add('active');
                console.log('[CHAT LIST] Force-activated:', currentSessionId);
            } else {
                console.log('[CHAT LIST] WARNING: currentSessionId not found in list:', currentSessionId);
            }
        }
        
        console.log('[CHAT LIST] Done rendering', chats.length, 'items');
    } catch (error) {
        console.error('Error loading chat list:', error);
    }
}

// Helper function to select a chat in the list by ID
function selectChatInList(chatId) {
    const chatList = document.getElementById('chat-list');
    if (!chatList) return;
    
    // Remove active class from all items
    chatList.querySelectorAll('li.active').forEach(li => li.classList.remove('active'));
    
    // Add active class to the target chat
    const targetLi = chatList.querySelector(`li[data-chat-id="${chatId}"]`);
    if (targetLi) {
        targetLi.classList.add('active');
        // Scroll it into view if needed
        targetLi.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        console.log('[CHAT LIST] Selected chat:', chatId);
    } else {
        console.log('[CHAT LIST] Could not find chat to select:', chatId);
    }
}

async function loadSession(id) {
    try {
        const response = await fetch(`/api/chats/${id}`);
        const chat = await response.json();
        
        currentSessionId = id;
        
        const chatWindow = document.getElementById('chat-window');
        chatWindow.innerHTML = '';
        
        chat.messages.forEach((msg, index) => {
            if (msg.role !== 'system') {
                if (msg.role === 'user') {
                    // Check for saved attachments metadata
                    const savedAttachments = msg._attachments;
                    if (savedAttachments && (savedAttachments.images?.length > 0 || savedAttachments.files?.length > 0)) {
                        // Render with attachments from history
                        appendMessageWithSavedAttachments(msg.content, savedAttachments, id);
                    } else {
                        appendMessage(msg.content, 'user');
                    }
                } else if (msg.role === 'assistant') {
                    // Skip assistant messages that are ONLY unexecuted tool call JSON text
                    if (msg.content && msg.content.trim().match(/^\{"name":\s*"potatool_\w+"/)) {
                        return; // Don't render unexecuted tool call JSON
                    }
                    
                    // Skip intermediate tool call messages (empty content with tool_calls) - NO EXTRA BUBBLES
                    if ((!msg.content || msg.content.trim() === '')) {
                        return; // Don't render messages without actual response content
                    }
                    
                    // Bot message - create ONE bubble with correct model label
                    const msgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    const msgDiv = document.createElement('div');
                    msgDiv.className = 'msg bot';
                    msgDiv.id = msgId;
                    
                    // Use model from message JSON, NOT currentModel
                    const displayModel = msg.model || 'Unknown Model';
                    
                    // Collect tool calls from preceding messages (backward scan until last user message)
                    const allToolCalls = [];
                    for (let i = index - 1; i >= 0; i--) {
                        const prevMsg = chat.messages[i];
                        if (prevMsg.role === 'user') break; // Stop at previous user message
                        if (prevMsg.role === 'assistant' && prevMsg.tool_calls && prevMsg.tool_calls.length > 0) {
                            // Add tool calls in order (since we're going backwards, unshift instead of push)
                            allToolCalls.unshift(...prevMsg.tool_calls);
                        }
                    }
                    // Also add tool calls from current message if any
                    if (msg.tool_calls && msg.tool_calls.length > 0) {
                        allToolCalls.push(...msg.tool_calls);
                    }
                    
                    // ALWAYS build thinking section (even if empty) - matches live generation structure
                    const hasThinking = msg._thinking && msg._thinking.trim();
                    const hasToolCalls = allToolCalls.length > 0;
                    
                    // Always create thinking section structure
                    let thinkingSectionHTML = `
                    <div class="thinking-section" id="${msgId}-thinking-section" style="display: block;">
                      <div class="thinking-header" onclick="toggleThinking('${msgId}')">
                        <span><i class="fas fa-brain"></i> Thinking & Tools</span>
                        <span class="thinking-toggle collapsed" id="${msgId}-thinking-toggle"><i class="fas fa-chevron-down"></i></span>
                      </div>
                      <div class="thinking-content collapsed" id="${msgId}-thinking-content">`;
                    
                    // Helper function to render a single tool call
                    const renderToolCall = (toolCall, idx) => {
                        const toolName = toolCall.function.name;
                        const toolArgs = toolCall.function.arguments;
                        const toolResult = toolCall.function.result;
                        const toolId = `tool-${msgId}-${idx}`;
                        
                        let argsDisplay = '';
                        try {
                            const argsObj = typeof toolArgs === 'string' ? JSON.parse(toolArgs) : toolArgs;
                            argsDisplay = Object.entries(argsObj)
                                .map(([key, value]) => `<div class="tool-detail-line"><strong>${key}:</strong> ${value}</div>`)
                                .join('');
                        } catch (e) {
                            argsDisplay = `<pre class="tool-args-pre">${typeof toolArgs === 'string' ? toolArgs : JSON.stringify(toolArgs, null, 2)}</pre>`;
                        }
                        
                        let resultDisplay = '';
                        if (toolResult) {
                            const resultStr = typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult, null, 2);
                            resultDisplay = `
                                <div class="tool-detail-line"><strong>Result:</strong></div>
                                <pre class="tool-result-pre">${resultStr}</pre>
                            `;
                        }
                        
                        return `
                        <div class="tool-call-container" data-tool-id="${toolId}" style="cursor: pointer;">
                            <div class="tool-status-line">
                                <span class="tool-toggle" id="${toolId}-toggle">
                                    <i class="fas fa-chevron-right"></i>
                                </span>
                                <span class="tool-message">🔧 ${toolName.replace('potatool_', '').replace(/_/g, ' ')}</span>
                            </div>
                            <div class="tool-detail-section collapsed" id="${toolId}-detail">
                                <div class="tool-detail-line"><strong>Function:</strong> ${toolName}</div>
                                ${argsDisplay}
                                ${resultDisplay}
                            </div>
                        </div>`;
                    };
                    
                    // Parse thinking text for [[TOOL_CALL:n]] markers to interleave tool calls chronologically
                    if (hasThinking && hasToolCalls) {
                        // Split by tool call markers - pattern: [[TOOL_CALL:0]], [[TOOL_CALL:1]], etc.
                        const toolCallPattern = /\[\[TOOL_CALL:(\d+)\]\]/g;
                        const thinkingText = msg._thinking;
                        
                        // Find all markers and their positions
                        let lastIndex = 0;
                        let match;
                        const usedToolIndices = new Set();
                        
                        while ((match = toolCallPattern.exec(thinkingText)) !== null) {
                            // Add thinking text before this marker
                            const textBefore = thinkingText.substring(lastIndex, match.index).trim();
                            if (textBefore) {
                                thinkingSectionHTML += `<div class="thinking-text">${textBefore}</div>`;
                            }
                            
                            // Add the corresponding tool call
                            const toolIndex = parseInt(match[1], 10);
                            if (toolIndex < allToolCalls.length) {
                                thinkingSectionHTML += renderToolCall(allToolCalls[toolIndex], toolIndex);
                                usedToolIndices.add(toolIndex);
                            }
                            
                            lastIndex = match.index + match[0].length;
                        }
                        
                        // Add any remaining thinking text after the last marker
                        const remainingText = thinkingText.substring(lastIndex).trim();
                        if (remainingText) {
                            thinkingSectionHTML += `<div class="thinking-text">${remainingText}</div>`;
                        }
                        
                        // Add any tool calls that weren't referenced by markers (fallback)
                        allToolCalls.forEach((toolCall, idx) => {
                            if (!usedToolIndices.has(idx)) {
                                thinkingSectionHTML += renderToolCall(toolCall, idx);
                            }
                        });
                    } else if (hasThinking) {
                        // No tool calls, just add thinking text (strip any orphaned markers)
                        const cleanThinking = msg._thinking.replace(/\[\[TOOL_CALL:\d+\]\]/g, '').trim();
                        if (cleanThinking) {
                            thinkingSectionHTML += `<div class="thinking-text">${cleanThinking}</div>`;
                        }
                    } else if (hasToolCalls) {
                        // No thinking text, just add tool calls
                        allToolCalls.forEach((toolCall, idx) => {
                            thinkingSectionHTML += renderToolCall(toolCall, idx);
                        });
                    }
                    
                    // Close thinking section (always present)
                    thinkingSectionHTML += `</div></div>`;
                    
                    msgDiv.innerHTML = `
                        <div class="model-badge">${displayModel}</div>
                        ${thinkingSectionHTML}
                        <div class="markdown-content" id="${msgId}-content"></div>
                    `;
                    
                    chatWindow.appendChild(msgDiv);
                    
                    // Attach click handlers to tool containers
                    if (hasToolCalls) {
                        msgDiv.querySelectorAll('.tool-call-container').forEach(container => {
                            container.addEventListener('click', function(e) {
                                e.stopPropagation();
                                const toolId = this.getAttribute('data-tool-id');
                                const details = document.getElementById(toolId + '-detail');
                                const toggle = document.getElementById(toolId + '-toggle');
                                if (details && toggle) {
                                    if (details.classList.contains('collapsed')) {
                                        details.classList.remove('collapsed');
                                        toggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
                                    } else {
                                        details.classList.add('collapsed');
                                        toggle.innerHTML = '<i class="fas fa-chevron-right"></i>';
                                    }
                                }
                            });
                        });
                    }
                    
                    // Render markdown content
                    const contentDiv = document.getElementById(`${msgId}-content`);
                    if (contentDiv) {
                        renderMarkdown(contentDiv, msg.content);
                    }
                }
                // Skip 'tool' role messages - they're internal
            }
        });
        
        // Refresh chat list and select this chat
        await loadChatList();
        selectChatInList(id);
    } catch (error) {
        console.error('Error loading session:', error);
    }
}

function startNewChat() {
    currentSessionId = null;
    document.getElementById('chat-window').innerHTML = '';
    uploadedFiles = [];
    uploadedImages = [];  // Clear uploaded images too
    uploadPreviews = [];  // Clear upload previews
    
    // Remove active class from all chat items
    const chatItems = document.querySelectorAll('#chat-list li');
    chatItems.forEach(item => item.classList.remove('active'));
    
    // Update image count display and preview
    updateUploadedImagesDisplay();
    renderUploadPreviews();
    
    // Reload chat list to reflect no active session
    loadChatList();
}

// --- FILE UPLOAD ---
async function uploadFile(input) {
    if (input.files.length === 0) return;
    
    const file = input.files[0];
    const filename = file.name.toLowerCase();
    const isImage = /\.(png|jpg|jpeg|gif|bmp|webp|avif|heic|heif)$/i.test(filename);
    
    // If it's an image, use the image upload handler
    if (isImage) {
        await uploadImageFile(file);
        return;
    }
    
    // Regular file upload to chat-specific directory
    // Generate UUID for new chats
    if (!currentSessionId) {
        currentSessionId = crypto.randomUUID();
        console.log('[UPLOAD] Generated new session ID:', currentSessionId);
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chat_id', currentSessionId);
    formData.append('model', currentModel);
    
    try {
        appendMessage(`Uploading file: ${file.name}...`, 'system');
        
        const response = await fetch('/api/upload_chat_file', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            uploadedFiles.push({
                filename: data.filename,
                path: data.path,
                content: data.content,
                file_type: data.file_type
            });
            
            let msg = `✓ File uploaded: ${data.filename}`;
            if (data.file_type === 'pdf') {
                msg += ` (${data.pages || 0} pages)`;
            }
            appendMessage(msg, 'system');
        } else {
            appendMessage(`Error uploading file: ${data.error || 'Unknown error'}`, 'system');
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        appendMessage('Error uploading file', 'system');
    }
}

async function uploadImageFile(fileOrInput) {
    // Handle both input elements and direct file objects
    let file;
    if (fileOrInput instanceof HTMLInputElement) {
        if (fileOrInput.files.length === 0) return;
        file = fileOrInput.files[0];
    } else if (fileOrInput instanceof File) {
        file = fileOrInput;
    } else {
        console.error('uploadImageFile: Invalid argument');
        return;
    }
    
    // Check if we can process images
    if (!canProcessImages) {
        appendMessage('Cannot upload images - no vision model available. Install a vision model like llava:7b or similar via Ollama.', 'system');
        return;
    }
    
    // Generate UUID for new chats
    if (!currentSessionId) {
        currentSessionId = crypto.randomUUID();
        console.log('[UPLOAD] Generated new session ID:', currentSessionId);
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chat_id', currentSessionId);
    formData.append('model', currentModel);
    
    try {
        appendMessage(`Uploading image: ${file.name}...`, 'system');
        
        const response = await fetch('/api/upload_chat_file', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error === 'vision_required') {
            appendMessage(`⚠️ ${data.message}`, 'system');
            return;
        }
        
        if (data.success) {
            let msg = `✓ Image uploaded: ${data.filename}`;
            
            if (data.is_vl_model && data.image_base64) {
                // VL model - store base64 for Ollama's images array
                uploadedImages.push({
                    filename: data.filename,
                    base64: data.image_base64,
                    is_vl_model: true
                });
                msg += ' (ready for vision model)';
            } else if (data.content) {
                // Non-VL model - image was described, treat as file
                uploadedFiles.push({
                    filename: data.filename,
                    content: data.content,
                    file_type: 'image_description'
                });
                msg += ` (described by ${data.fallback_model || 'fallback model'})`;
            }
            
            appendMessage(msg, 'system');
            
            // Update count display
            updateUploadedImagesDisplay();
        } else {
            appendMessage(`Error uploading image: ${data.error || 'Unknown error'}`, 'system');
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        appendMessage('Error uploading image', 'system');
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

// Track if models are unloaded
let sttManuallyUnloaded = false;
let ttsManuallyUnloaded = false;

async function unloadSTT() {
    try {
        const response = await fetch('/api/stt_unload', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            sttManuallyUnloaded = true;
            window.sttManuallyUnloaded = true;  // Expose globally for VOX
            whisperPreloaded = false;
            window.whisperPreloaded = false;
            window.whisperModelLoaded = false;
            console.log('[STT] Whisper model unloaded');
            
            const btn = document.getElementById('unload-stt-btn');
            if (btn) {
                btn.innerHTML = '<i class="fas fa-check"></i> STT Unloaded';
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-ear-listen"></i> Unload STT';
                }, 2000);
            }
        }
    } catch (error) {
        console.error('[STT] Error unloading Whisper:', error);
        alert('Error unloading STT: ' + error.message);
    }
}

async function unloadTTS() {
    try {
        console.log('[TTS] Unloading TTS models...');
        const response = await fetch('/api/unload_tts', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            ttsManuallyUnloaded = true;
            window.ttsManuallyUnloaded = true;  // Expose globally
            console.log('[TTS] TTS model unloaded:', data.message);
            
            const btn = document.getElementById('unload-tts-btn');
            if (btn) {
                btn.innerHTML = '<i class="fas fa-check"></i> TTS Unloaded';
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-volume-high"></i> Unload TTS';
                }, 2000);
            }
        } else {
            console.error('[TTS] Unload failed:', data.error);
        }
    } catch (error) {
        console.error('[TTS] Error unloading TTS:', error);
        alert('Error unloading TTS: ' + error.message);
    }
}

// Auto-reload STT when recording starts
function reloadSTTIfNeeded() {
    if (sttManuallyUnloaded) {
        console.log('[STT] Model was manually unloaded, will reload on next transcription');
        sttManuallyUnloaded = false;
        whisperPreloaded = false;
        window.whisperPreloaded = false;
        window.whisperModelLoaded = false;
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
            pasteUploadsUseRag = settings.bools.PASTE_UPLOADS_USE_RAG || false;
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
        case 'system_prompts':
            renderSystemPromptsTab(container);
            return;
        case 'config':
            renderConfigurationTab(container);
            return;
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
    
    // Fallback/generated descriptions (marked with "(?)" prefix)
    const fallbackDescriptions = {
        // Core LLM Config
        'MAX_INTERMEDIATE_TOKENS': '(?) Maximum tokens for intermediate reasoning steps (-1 for unlimited)',
        'MAX_TOKENS': 'Maximum response length in tokens',
        'MAX_SPEECH_TOKENS': 'Maximum tokens for speech/TTS responses',
        'TEMPERATURE': 'Creativity level (0.0 = deterministic, 1.0 = creative)',
        
        // Online Search Config
        'SEARXNG_NUM_RESULTS': 'Number of search results to retrieve from SearXNG API for each query',
        'SMART_SCANNING': 'Enable smart scanning of search results to extract relevant information',
        'RECURSIVE_SEARCH': 'Enable recursive search to further explore links found in initial search results',
        'RECURSIVE_SEARCH_DEPTH': 'Depth of recursive search when enabled',
        'SUMMARIZE_SEARCH_RESULTS': 'Summarize search results before using them for context generation',
        
        // RAG Config
        'RAG_TOP_K': 'Number of top relevant documents to retrieve for RAG',
        'RAG_MAX_TOKENS': 'Max tokens for RAG context generation',
        'CHUNK_SIZE': 'Size of text chunks for vectorization and retrieval',
        'CHUNK_OVERLAP': 'Overlap size between text chunks for better context retention',
        'RELEVANCE_CHECK_RAG': 'Enable relevance checking for RAG context documents',
        'RELEVANCE_CHECK_THRESHOLD': 'Threshold for relevance checking during RAG',
        'SUMMARIZE_RETRIEVED_DOCS': 'Summarize retrieved documents before using them for context generation',
        
        // Voice Config
        'VOX_MODEL': 'Ollama model used for VOX voice chat (leave empty to use CORE_OLLAMA_MODEL)',
        'WAKE_WORD': 'Wake word for voice activation',
        'SILENCE_THRESHOLD': 'Silence threshold in seconds to detect end of speech',
        'SLEEP_WORDS': 'Comma-separated list of words/phrases to put the assistant to sleep',
        'RESET_CONTEXT_WORDS': 'Comma-separated list of words/phrases to reset the conversation context',
        
        // System Prompts
        'VOX_SYSTEM_PROMPT': 'Custom system prompt prepended to all VOX Core voice interactions',
        'CHAT_SYSTEM_PROMPT': 'Custom system prompt prepended to all chat conversations',
        'TTS_SYSTEM_PROMPT': 'System prompt specifically for TTS voice output - controls speaking style',
        
        // Ollama Models
        'CORE_OLLAMA_MODEL': 'The main model used for core chat functionalities (required)',
        'CHAT_NAMING_MODEL': 'Model used to generate chat titles (leave empty to use CORE_OLLAMA_MODEL)',
        'SUMMARIZATION_OLLAMA_MODEL': 'Model used for summarization tasks (leave empty to use CORE_OLLAMA_MODEL)',
        'RELEVANCE_OLLAMA_MODEL': 'Model used for determining relevance during RAG (leave empty to use CORE_OLLAMA_MODEL)',
        'EMBEDDING_OLLAMA_MODEL': 'Model used for generating text embeddings for local vector database',
        'BACKUP_VISION_MODEL': 'Backup model for processing images when current model lacks vision capability',
        'DEFAULT_OCR_MODEL': 'Default model for OCR and image text extraction',
        'CHATTERBOX_TTS_MODEL_PATH': 'Path to custom TTS model for chatterbox mode (optional)',
        
        // API Endpoints
        'OLLAMA_API_URL': 'Your Ollama API endpoint URL',
        'DISCORD_BOT_TOKEN': '(?) Discord bot token for bot commands',
        'DISCORD_WEBHOOK_URL': '(?) Discord webhook URL for sending messages to a channel',
        
        // Data Paths
        'LOCAL_DATA_PATH': 'Path to local data files for retrieval-augmented generation',
        'VECTOR_DB_PATH': 'Path to the local vector database for embeddings and RAG',
        'OTHER_DATA_PATH': 'Path to other data files, such as chat histories',
        
        // Web UI Config
        'ENABLE_WEBUI': '(?) Enable the web UI interface',
        'WEBUI_PORT': '(?) Port for the web UI server',
        'WEBUI_HOST': '(?) Host address for the web UI server',
        'ENABLE_WEBUI_AUTH': 'Set to true to enable authentication for the web UI',
        'WEBUI_USERNAME': 'Username for web UI authentication',
        'WEBUI_PASSWORD': 'Password for web UI authentication',
        'ACCEPT_TEXT_FILE_TYPES': 'Comma-separated list of accepted text-based file types for upload',
        'ACCEPT_IMAGE_FILE_TYPES': 'Comma-separated list of accepted image file types for upload',
        
        // Safe Code Execution
        'CHECK_CODE_SAFETY': 'Enable safety checks for code execution using static analysis',
        'CODE_EXECUTION_TIMEOUT': 'Timeout in seconds for code execution',
        'SANDBOX_PATH': 'Path to sandbox directory for safe code execution',
        
        // Bools
        'USE_TOR': 'Route web requests through Tor network for enhanced privacy',
        'ENABLE_HISTORY': 'Enable chat history saving and retrieval',
        'RESET_HISTORY_ON_START': 'Reset chat history when the application starts',
        'SAVE_SENT_FILES': 'Save files sent by users in chats',
        'ENABLE_FILE_UPLOAD': 'Enable file upload functionality in the web UI',
        'ENABLE_FOLDER_UPLOAD': 'Enable folder upload functionality in the web UI',
        'ENABLE_CODE_UPLOAD': 'Enable code file upload functionality in the web UI',
        'ENABLE_IMAGE_PROCESSING': 'Enable image processing capabilities (e.g., OCR, image analysis)',
        'ENABLE_SENT_URLS': 'Enable processing of URLs sent by users in chats',
        'SAFE_CODE_EXECUTION': 'Enable safe code execution environment for running code snippets',
        'USE_LATEX_RENDERING': 'Enable LaTeX rendering for mathematical expressions in responses',
        'ENABLE_DISCORD_BOT': '(?) Enable Discord bot integration',
        'ENABLE_TTS': 'Enable Text-to-Speech output',
        'ENABLE_STT': 'Enable Speech-to-Text input',
        'ENABLE_SEARXNG_SEARCH': 'Enable web search using SearXNG API',
        'ENABLE_DANGEROUS_TOOLS': 'Enable potentially dangerous tools that can modify or delete files',
        'ENABLE_OLLAMA_STREAMING': 'Enable streaming responses from Ollama for faster response times',
        'ENABLE_AGENTS': 'Enable agent functionalities for more complex tasks',
        'ENABLE_AUTONOMOUS_MODE': 'Enable autonomous mode for the assistant to perform tasks without user intervention',
        'ENABLE_DEVICE_CONTROL': 'Enable device control functionalities',
        'ENABLE_LOGS': 'Enable logging of interactions and events for debugging and analysis',
        'ENABLE_ALWAYS_HITL': 'Enable Always Human-in-the-Loop mode for continuous human oversight',
        'IDIOT_MODE': 'Enable idiot mode for simplified interactions and reduced complexity',
        'PASTE_UPLOADS_USE_RAG': 'When enabled, pasted/uploaded files are embedded to vector DB for RAG. When disabled, files are sent directly to the model as context.'
    };
    
    return fallbackDescriptions[key] || '(?) No description available';
}

function renderSystemPromptsTab(container) {
    const systemPrompts = settings.system_prompts || {
        VOX_SYSTEM_PROMPT: '',
        CHAT_SYSTEM_PROMPT: '',
        TTS_SYSTEM_PROMPT: ''
    };
    
    const promptConfigs = [
        {
            key: 'VOX_SYSTEM_PROMPT',
            label: 'VOX System Prompt',
            description: 'Custom instructions prepended to all VOX Core voice interactions. This allows you to customize the AI\'s personality and behavior specifically for voice chats.',
            placeholder: 'e.g., You are a helpful, conversational AI assistant...'
        },
        {
            key: 'CHAT_SYSTEM_PROMPT',
            label: 'Chat System Prompt',
            description: 'Custom instructions prepended to all text chat conversations. This allows you to customize the AI\'s personality and behavior for text-based interactions.',
            placeholder: 'e.g., You are a helpful AI assistant...'
        }
    ];
    
    promptConfigs.forEach(config => {
        const card = document.createElement('div');
        card.className = 'setting-card prompt-card';
        
        const value = systemPrompts[config.key] || '';
        const expandedClass = value ? '' : 'collapsed';
        
        card.innerHTML = `
            <div class="prompt-header" onclick="togglePromptExpand(this)">
                <label>${config.label}</label>
                <i class="fas fa-chevron-down expand-icon"></i>
            </div>
            <div class="prompt-content ${expandedClass}">
                <div class="setting-description">${config.description}</div>
                <textarea 
                    id="setting-${config.key}" 
                    rows="6" 
                    placeholder="${config.placeholder}"
                >${value}</textarea>
            </div>
        `;
        
        container.appendChild(card);
    });
}

function togglePromptExpand(headerElement) {
    const content = headerElement.nextElementSibling;
    const icon = headerElement.querySelector('.expand-icon');
    
    content.classList.toggle('collapsed');
    icon.classList.toggle('rotated');
}

function renderConfigurationTab(container) {
    // Configuration tab combines: potato, ollama_models, data_paths, web_ui_config, safe_code_execution_config
    const sections = [
        { title: 'Ollama Models', data: settings.configuration?.ollama_models || {} },
        { title: 'API Endpoints', data: settings.configuration?.potato || {} },
        { title: 'Data Paths', data: settings.data_paths || {} },
        { title: 'Web UI Configuration', data: settings.web_ui_config || {} },
        { title: 'Safe Code Execution', data: settings.safe_code_execution_config || {} }
    ];
    
    sections.forEach(section => {
        if (Object.keys(section.data).length === 0) return;
        
        const sectionHeader = document.createElement('h3');
        sectionHeader.className = 'settings-section-header';
        sectionHeader.textContent = section.title;
        container.appendChild(sectionHeader);
        
        Object.keys(section.data).forEach(key => {
            const card = document.createElement('div');
            card.className = 'setting-card';
            
            const value = section.data[key];
            const inputType = typeof value === 'boolean' ? 'checkbox' : 
                             typeof value === 'number' ? 'number' : 'text';
            
            let inputHtml = '';
            if (inputType === 'checkbox') {
                inputHtml = `<input type="checkbox" id="setting-${section.title.replace(/ /g, '_')}-${key}" ${value ? 'checked' : ''} />`;
            } else {
                inputHtml = `<input type="${inputType}" id="setting-${section.title.replace(/ /g, '_')}-${key}" value="${value}" />`;
            }
            
            card.innerHTML = `
                <label>${key.replace(/_/g, ' ')}</label>
                ${inputHtml}
                <div class="setting-description">${getSettingDescription(key)}</div>
            `;
            
            container.appendChild(card);
        });
    });
}

async function saveSettings() {
    const newSettings = { ...settings };
    
    // Initialize system_prompts if it doesn't exist
    if (!newSettings.system_prompts) {
        newSettings.system_prompts = {};
    }
    
    // Gather all inputs
    const inputs = document.querySelectorAll('[id^="setting-"]');
    inputs.forEach(input => {
        let key = input.id.replace('setting-', '');
        let value = input.type === 'checkbox' ? input.checked : input.value;
        
        if (input.type === 'number') {
            value = parseFloat(value);
        }
        
        // Handle Configuration tab inputs (they have prefixes like "Ollama_Models-KEY")
        if (key.includes('-')) {
            const parts = key.split('-');
            const section = parts[0].replace(/_/g, ' ');
            const actualKey = parts.slice(1).join('-');
            
            // Map section names to settings structure
            if (section === 'Ollama Models') {
                if (!newSettings.configuration) newSettings.configuration = {};
                if (!newSettings.configuration.ollama_models) newSettings.configuration.ollama_models = {};
                newSettings.configuration.ollama_models[actualKey] = value;
            } else if (section === 'API Endpoints') {
                if (!newSettings.configuration) newSettings.configuration = {};
                if (!newSettings.configuration.potato) newSettings.configuration.potato = {};
                newSettings.configuration.potato[actualKey] = value;
            } else if (section === 'Data Paths') {
                if (!newSettings.data_paths) newSettings.data_paths = {};
                newSettings.data_paths[actualKey] = value;
            } else if (section === 'Web UI Configuration') {
                if (!newSettings.web_ui_config) newSettings.web_ui_config = {};
                newSettings.web_ui_config[actualKey] = value;
            } else if (section === 'Safe Code Execution') {
                if (!newSettings.safe_code_execution_config) newSettings.safe_code_execution_config = {};
                newSettings.safe_code_execution_config[actualKey] = value;
            }
        } 
        // Handle system prompts
        else if (key === 'VOX_SYSTEM_PROMPT' || key === 'CHAT_SYSTEM_PROMPT' || key === 'TTS_SYSTEM_PROMPT') {
            newSettings.system_prompts[key] = value;
        }
        // Handle regular settings
        else if (newSettings.core_llm_config && key in newSettings.core_llm_config) {
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
    const sidebar = document.getElementById('right-sidebar');
    const isOpen = sidebar.classList.contains('open');
    
    sidebar.classList.toggle('open');
    
    // Load embedded files when opening
    if (!isOpen) {
        loadEmbeddedFiles();
        checkWeaviateStatus();
    }
    
    // On mobile, add/remove overlay
    if (window.innerWidth <= 768) {
        let overlay = document.getElementById('rag-overlay');
        
        if (!isOpen) {
            // Opening sidebar - create overlay
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'rag-overlay';
                overlay.className = 'rag-mobile-overlay';
                overlay.onclick = () => toggleRightSidebar(); // Close on click
                document.body.appendChild(overlay);
            }
            // Trigger reflow to enable transition
            overlay.offsetHeight;
            overlay.classList.add('visible');
        } else {
            // Closing sidebar - remove overlay
            if (overlay) {
                overlay.classList.remove('visible');
                setTimeout(() => overlay.remove(), 300); // Wait for fade transition
            }
        }
    }
}

// === RAG PANEL STATE ===
let ragPendingFiles = [];  // {file, type: 'document'|'image', name}

// === RAG STATUS CHECK ===
async function checkWeaviateStatus() {
    const statusEl = document.getElementById('rag-weaviate-status');
    if (!statusEl) return;
    
    try {
        const response = await fetch('/api/rag_status');
        const data = await response.json();
        
        if (data.weaviate?.status === 'connected') {
            statusEl.innerHTML = '<i class="fas fa-circle"></i> <span>Weaviate Connected</span>';
            statusEl.className = 'rag-status-indicator connected';
        } else {
            statusEl.innerHTML = '<i class="fas fa-circle"></i> <span>Weaviate Disconnected - Start Docker container</span>';
            statusEl.className = 'rag-status-indicator disconnected';
        }
    } catch (e) {
        statusEl.innerHTML = '<i class="fas fa-circle"></i> <span>Weaviate Error</span>';
        statusEl.className = 'rag-status-indicator disconnected';
    }
}

// === LOAD EMBEDDED FILES FOR CURRENT CHAT ===
async function loadEmbeddedFiles() {
    const listEl = document.getElementById('embedded-files-list');
    if (!listEl) return;
    
    if (!currentSessionId) {
        listEl.innerHTML = '<div class="embedded-files-empty">Start a chat to embed files</div>';
        return;
    }
    
    try {
        const response = await fetch(`/api/rag_files/${currentSessionId}`);
        const data = await response.json();
        
        if (data.files && data.files.length > 0) {
            listEl.innerHTML = data.files.map(file => {
                const icon = getFileIconClass(file.content_type);
                return `
                    <div class="embedded-file-item">
                        <div class="embedded-file-info">
                            <i class="fas ${icon}"></i>
                            <span class="filename" title="${file.filename}">${truncateFilename(file.filename, 25)}</span>
                            <span class="chunk-count">(${file.chunk_count} chunks)</span>
                        </div>
                        <button class="embedded-file-delete" onclick="deleteEmbeddedFile('${file.filename}')" title="Delete embeddings">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
            }).join('');
        } else {
            listEl.innerHTML = '<div class="embedded-files-empty">No files embedded for this chat</div>';
        }
    } catch (e) {
        listEl.innerHTML = '<div class="embedded-files-empty">Error loading embedded files</div>';
        console.error('Error loading embedded files:', e);
    }
}

function getFileIconClass(contentType) {
    switch (contentType) {
        case 'image_ocr':
        case 'image':
            return 'fa-image';
        case 'pdf':
            return 'fa-file-pdf';
        default:
            return 'fa-file-alt';
    }
}

// === DELETE EMBEDDED FILE ===
async function deleteEmbeddedFile(filename) {
    if (!confirm(`Delete embeddings for "${filename}"?`)) return;
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = `Deleting embeddings for ${filename}...`;
    
    try {
        const response = await fetch('/api/delete_file_embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chat_id: currentSessionId,
                filename: filename
            })
        });
        
        const data = await response.json();
        if (data.success) {
            statusLog.textContent = `✓ Deleted embeddings for ${filename}`;
            loadEmbeddedFiles();  // Refresh list
        } else {
            statusLog.textContent = `✗ Error: ${data.error || 'Unknown error'}`;
        }
    } catch (e) {
        statusLog.textContent = `✗ Error: ${e.message}`;
    }
}

// === FILE SELECTION HANDLERS ===
function handleRagFolderSelect(input) {
    const files = Array.from(input.files);
    const validFiles = files.filter(f => isValidRagFile(f.name));
    
    validFiles.forEach(file => {
        ragPendingFiles.push({
            file: file,
            type: isImageFile(file.name) ? 'image' : 'document',
            name: file.webkitRelativePath || file.name
        });
    });
    
    updateRagPendingUI();
    input.value = '';  // Reset input
}

function handleRagFileSelect(input) {
    const files = Array.from(input.files);
    
    files.forEach(file => {
        ragPendingFiles.push({
            file: file,
            type: 'document',
            name: file.name
        });
    });
    
    updateRagPendingUI();
    input.value = '';
}

function handleRagImageSelect(input) {
    const files = Array.from(input.files);
    
    files.forEach(file => {
        ragPendingFiles.push({
            file: file,
            type: 'image',
            name: file.name
        });
    });
    
    updateRagPendingUI();
    input.value = '';
}

function isValidRagFile(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const validExtensions = [
        // Documents
        'pdf', 'txt', 'md', 'csv', 'tsv',
        // Config
        'json', 'yaml', 'yml', 'xml', 'env', 'ini', 'toml', 'conf', 'cfg',
        // Web
        'html', 'htm', 'css', 'scss', 'sass', 'less',
        // JS/TS
        'js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs',
        // Python
        'py', 'pyw', 'pyi',
        // C/C++
        'c', 'h', 'cpp', 'hpp', 'cc', 'cxx', 'hxx',
        // Other languages
        'java', 'kt', 'scala', 'go', 'rs', 'rb', 'php', 'pl', 'pm',
        'swift', 'm', 'mm', 'r', 'lua', 'sh', 'bash', 'zsh',
        'bat', 'cmd', 'ps1', 'psm1',
        // Data
        'sql', 'graphql', 'proto',
        // Misc
        'log', 'rst', 'tex', 'dockerfile', 'gitignore', 'editorconfig',
        // Images
        'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif'
    ];
    return validExtensions.includes(ext);
}

function isImageFile(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif'].includes(ext);
}

// === UPDATE PENDING FILES UI ===
function updateRagPendingUI() {
    const container = document.getElementById('rag-pending-container');
    const listEl = document.getElementById('rag-pending-list');
    const embedBtn = document.getElementById('rag-embed-btn');
    
    // Update counts
    const folderCount = ragPendingFiles.filter(f => f.type === 'document').length;
    const imageCount = ragPendingFiles.filter(f => f.type === 'image').length;
    
    document.getElementById('rag-folder-count').textContent = `${folderCount} files`;
    document.getElementById('rag-file-count').textContent = `${folderCount} files`;
    document.getElementById('rag-image-count').textContent = `${imageCount} images`;
    
    // Update pending list
    if (ragPendingFiles.length > 0) {
        container.style.display = 'block';
        listEl.innerHTML = ragPendingFiles.map((item, idx) => {
            const icon = item.type === 'image' ? 'fa-image' : 'fa-file-alt';
            return `
                <div class="rag-pending-item">
                    <span><i class="fas ${icon}"></i> ${truncateFilename(item.name, 30)}</span>
                    <button class="rag-pending-remove" onclick="removeRagPending(${idx})">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        }).join('');
        embedBtn.disabled = false;
    } else {
        container.style.display = 'none';
        embedBtn.disabled = true;
    }
}

function removeRagPending(idx) {
    ragPendingFiles.splice(idx, 1);
    updateRagPendingUI();
}

// === EMBED FILES TO VECTOR DB ===
async function embedRagFiles() {
    if (ragPendingFiles.length === 0) {
        alert('Please select files first');
        return;
    }
    
    if (!currentSessionId) {
        // Generate session ID if none exists
        currentSessionId = crypto.randomUUID();
        console.log('[RAG] Generated new session ID:', currentSessionId);
    }
    
    const statusLog = document.getElementById('rag-status');
    const embedBtn = document.getElementById('rag-embed-btn');
    const pdfVisualCheckbox = document.getElementById('rag-pdf-visual');
    const pdfVisualEnabled = pdfVisualCheckbox?.checked || false;
    
    embedBtn.disabled = true;
    embedBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Embedding...';
    statusLog.textContent = 'Preparing files for embedding...\n';
    
    try {
        const formData = new FormData();
        formData.append('chat_id', currentSessionId);
        formData.append('pdf_visual_analysis', pdfVisualEnabled ? 'true' : 'false');
        
        // Separate documents and images
        const documents = ragPendingFiles.filter(f => f.type === 'document');
        const images = ragPendingFiles.filter(f => f.type === 'image');
        
        for (const item of documents) {
            formData.append('files', item.file, item.name);
        }
        
        for (const item of images) {
            formData.append('images', item.file, item.name);
        }
        
        statusLog.textContent += `Uploading ${documents.length} documents and ${images.length} images...\n`;
        if (pdfVisualEnabled) {
            statusLog.textContent += `PDF visual analysis ENABLED (using llava for graphs/images)\n`;
        }
        
        const response = await fetch('/api/embed_to_rag', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            statusLog.textContent += `✓ Embedded ${result.embedded_count} items\n`;
            statusLog.textContent += `Collection: ${result.collection_name}\n`;
            
            // Clear pending files
            ragPendingFiles = [];
            updateRagPendingUI();
            
            // Refresh embedded files list
            loadEmbeddedFiles();
        } else {
            statusLog.textContent += `✗ Error: ${result.error}\n`;
        }
    } catch (error) {
        statusLog.textContent += `✗ Error: ${error.message}\n`;
        console.error('RAG embedding error:', error);
    }
    
    embedBtn.disabled = ragPendingFiles.length === 0;
    embedBtn.innerHTML = '<i class="fas fa-database"></i> Embed to Vector DB';
}

// === SAVE RAG PREFERENCE ===
function saveRagPreference() {
    const enabled = document.getElementById('rag-enable')?.checked || false;
    localStorage.setItem('ragEnabled', enabled);
}

// Load RAG preference on init
document.addEventListener('DOMContentLoaded', () => {
    const ragCheckbox = document.getElementById('rag-enable');
    if (ragCheckbox) {
        ragCheckbox.checked = localStorage.getItem('ragEnabled') === 'true';
    }
});

// Keep old functions for backwards compatibility
let ragFolderFiles = [];
let ragImageFiles = [];

function selectRagFolder() {
    // Trigger the hidden file input
    document.getElementById('rag-folder-input').click();
}

async function embedFolderContents() {
    if (ragFolderFiles.length === 0 && ragImageFiles.length === 0) {
        alert('Please select a folder or images first');
        return;
    }
    
    if (!currentSessionId) {
        alert('Please start a chat first before embedding to RAG');
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
        formData.append('chat_id', currentSessionId); // Include chat ID for namespacing
        
        statusLog.textContent += `Uploading ${ragFolderFiles.length} files and ${ragImageFiles.length} images...\n`;
        statusLog.textContent += `Chat ID: ${currentSessionId}\n`;
        
        const response = await fetch('/api/embed_to_rag', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            statusLog.textContent += `✓ Embedded ${result.embedded_count} items to vector DB\n`;
            statusLog.textContent += `Collection: ${result.collection_name}\n`;
            
            // Clear the selected files after successful embedding
            ragFolderFiles = [];
            ragImageFiles = [];
            document.getElementById('rag-folder-display').value = '';
            document.getElementById('rag-image-count').textContent = '0 images';
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
    
    if (!currentSessionId) {
        alert('Please start a chat first before searching RAG');
        return;
    }
    
    const statusLog = document.getElementById('rag-status');
    statusLog.textContent = `Searching vector DB for: ${query}...\n`;
    statusLog.textContent += `Chat ID: ${currentSessionId}\n`;
    
    fetch('/api/search_rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            query, 
            top_k: 5,
            chat_id: currentSessionId  // Include chat ID for namespaced search
        })
    })
    .then(res => res.json())
    .then(result => {
        if (result.results && result.results.length > 0) {
            statusLog.textContent += `Found ${result.results.length} results:\n`;
            result.results.forEach((r, i) => {
                const preview = r.content.substring(0, 100).replace(/\n/g, ' ');
                statusLog.textContent += `${i+1}. ${preview}... (score: ${r.score.toFixed(3)})\n`;
            });
        } else if (result.results && result.results.length === 0) {
            statusLog.textContent += `No results found. The RAG collection may be empty for this chat.\n`;
        } else if (result.error) {
            statusLog.textContent += `✗ Error: ${result.error}\n`;
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
