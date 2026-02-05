// VOX Interrupt Detection Module
// Handles interrupt words during TTS playback to prevent echo

// Interrupt words that can stop TTS
const INTERRUPT_WORDS = [
    'okay thanks',
    'alright stop',
    'okay enough',
    'shut up',
];

// Buffer for audio captured during TTS
let duringTTSAudioChunks = [];
let afterTTSAudioChunks = [];
let isTTSSpeaking = false;
let interruptCheckInterval = null;

// Check if text contains any interrupt words
function containsInterruptWord(text) {
    if (!text) return false;
    
    const lowerText = text.toLowerCase().trim();
    
    for (const word of INTERRUPT_WORDS) {
        if (lowerText.includes(word)) {
            console.log(`[VOX-INTERRUPT] Detected interrupt word: "${word}" in "${text}"`);
            return true;
        }
    }
    
    return false;
}

// Start monitoring TTS state
function startTTSMonitoring() {
    if (interruptCheckInterval) {
        clearInterval(interruptCheckInterval);
    }
    
    // Check TTS state every 500ms
    interruptCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/tts_is_speaking');
            const data = await response.json();
            
            const wasSpeaking = isTTSSpeaking;
            isTTSSpeaking = data.is_speaking;
            
            // TTS just started
            if (isTTSSpeaking && !wasSpeaking) {
                console.log('[VOX-INTERRUPT] TTS started speaking');
                duringTTSAudioChunks = [];
                afterTTSAudioChunks = [];
            }
            
            // TTS just stopped
            if (!isTTSSpeaking && wasSpeaking) {
                console.log('[VOX-INTERRUPT] TTS finished speaking');
                // Move "during" chunks to "after" buffer since TTS is done
                if (duringTTSAudioChunks.length > 0) {
                    console.log(`[VOX-INTERRUPT] Moving ${duringTTSAudioChunks.length} chunks to after-TTS buffer`);
                    afterTTSAudioChunks = [...duringTTSAudioChunks, ...afterTTSAudioChunks];
                    duringTTSAudioChunks = [];
                }
            }
            
        } catch (error) {
            console.error('[VOX-INTERRUPT] Error checking TTS state:', error);
        }
    }, 500);
}

// Stop monitoring
function stopTTSMonitoring() {
    if (interruptCheckInterval) {
        clearInterval(interruptCheckInterval);
        interruptCheckInterval = null;
    }
    isTTSSpeaking = false;
    duringTTSAudioChunks = [];
    afterTTSAudioChunks = [];
}

// Process audio chunk during recording - THIS IS CRITICAL FOR PREVENTING ECHO
function processAudioChunk(audioBlob) {
    if (isTTSSpeaking) {
        console.log('[VOX-INTERRUPT] Audio captured during TTS, buffering for interrupt check only');
        duringTTSAudioChunks.push(audioBlob);
    } else {
        console.log('[VOX-INTERRUPT] Audio captured after TTS, adding to user prompt buffer');
        afterTTSAudioChunks.push(audioBlob);
    }
}

// Get the audio to transcribe (only post-TTS audio)
function getTranscriptionAudio() {
    if (isTTSSpeaking) {
        // Still speaking - check for interrupts only
        console.log('[VOX-INTERRUPT] TTS still speaking, checking for interrupts only');
        return {
            audio: duringTTSAudioChunks.length > 0 ? duringTTSAudioChunks[duringTTSAudioChunks.length - 1] : null,
            interruptOnly: true
        };
    } else {
        // TTS finished - use all post-TTS audio
        if (afterTTSAudioChunks.length > 0) {
            console.log(`[VOX-INTERRUPT] Using ${afterTTSAudioChunks.length} post-TTS audio chunks`);
            // Combine all chunks
            const combined = new Blob(afterTTSAudioChunks, { type: 'audio/webm' });
            afterTTSAudioChunks = []; // Clear buffer
            return {
                audio: combined,
                interruptOnly: false
            };
        } else {
            return {
                audio: null,
                interruptOnly: false
            };
        }
    }
}

// Handle interrupt detection from transcription
async function handleInterrupt(transcribedText) {
    if (!containsInterruptWord(transcribedText)) {
        console.log('[VOX-INTERRUPT] No interrupt word detected, ignoring...');
        return false;
    }
    
    console.log('[VOX-INTERRUPT] INTERRUPT! Stopping TTS...');
    
    // Stop TTS
    try {
        await fetch('/api/vox_stop', { method: 'POST' });
        isTTSSpeaking = false;
    } catch (error) {
        console.error('[VOX-INTERRUPT] Error stopping TTS:', error);
    }
    
    // Clear during-TTS buffer (don't want to process the interrupt itself)
    duringTTSAudioChunks = [];
    
    return true;
}

// Export functions
window.VOXInterrupt = {
    start: startTTSMonitoring,
    stop: stopTTSMonitoring,
    processChunk: processAudioChunk,
    getAudio: getTranscriptionAudio,
    handleInterrupt: handleInterrupt,
    containsInterruptWord: containsInterruptWord,
    isSpeaking: () => isTTSSpeaking
};
