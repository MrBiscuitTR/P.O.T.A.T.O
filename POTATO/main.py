# POTATO/main.py
# The core of P.O.T.A.T.O -- Practical Omnipurpose Technical AI Tool Operator
# Uses gpt-oss:20b as main model.
## Uses ollama module to interface with local LLMs.
## Uses langchain to manage chains, agents, and memory.(?)
## Uses a modular approach to add tools , utilities and functionalities. 
###(- ./components/vocal_tools/ , - ./components/visual_tools/ , - ./components/online_tools/ , - ./components/local_tools/ , - ./components/dangerous_tools/ , - ./components/n8n/ , - ./components/utilities/ )
### Uses a webui (./webui/) to provide an interface for users to interact with P.O.T.A.T.O. shows status, logs, system usage, prompts, desktop/browser, other useful parameters.

# POTATO/main.py
"""
Main entry point for the POTATO assistant system.
Handles the boot phase, event loop, and safety gates as described in project-notes.txt.

Architecture Overview:
1. Boot Phase: Load config, detect GPU/VRAM, load models, start background workers
2. Event Loop: Always listening (STT stream), push utterances to queue, agent decides → tool calls
3. Safety Gate: Any dangerous_tools call must pass intent check and explicit human confirmation, not vocal.

Key Design Principles:
- STT runs in its own thread/process, always on.
- TTS runs in its own thread/process. do not use piper. use something more realistic and with more emotion even if its more resource heavy.
- LLM reasoning never blocks audio I/O
- Streaming responses for natural conversation flow
- certain block words like ok stop, enough, thank you... will stop generation and tts.
"""

import os
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import ollama

# Import MCP tools directly instead of subprocess
try:
    from POTATO.MCP.searx_mcp import _potatool_web_search_urls_impl as potatool_web_search_urls, _potatool_extract_content_impl as potatool_extract_content
    _searx_available = True
    print("[MCP] Loaded searx tools directly")
except Exception as e:
    print(f"[MCP] Failed to import searx tools: {e}")
    _searx_available = False
    potatool_web_search_urls = None
    potatool_extract_content = None
import glob
import threading
import torch
import scipy.io.wavfile
import numpy as np
import ollama
from POTATO.components.utilities.get_system_info import json_get_instant_system_info
from POTATO.components.online_tools.searxng_get_urls import searx_get_urls_topic_page
from POTATO.components.visual_tools.image_gen import generate_image
from POTATO.components.utilities.summarize import summarize_content
from POTATO.components.utilities.combine_context import combine_context
from POTATO.components.utilities.check_relevance import check_preview_relevance, check_summary_relevance
from POTATO.components.online_tools.scrape_url_content import scrape_and_clean_url_content

initial_system_status = json_get_instant_system_info()

# Load config.json
def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("config.json not found, using default settings")
        return {}

# disable all telemetry/logging from third party libraries
os.environ["LANGCHAIN_HANDLER"] = "langchain_core.handlers.noop_handler.NoOpHandler"
os.environ["OPENAI_LOG"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFF"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["OLLAMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_METADATA"]   = "1" # block extra hub metadata requests
os.environ["HF_HUB_DISABLE_TOKENS"]     = "1" # prevent token usage telemetry

# --- CONSTANTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(BASE_DIR, "rules")
MODEL_INFO_PATH = os.path.join(BASE_DIR, '.data', '.modelinfos.json')

# --- PART 1: TOOL RULES LOADER ---
def load_rules():
    """Reads all .md files from POTATO/rules/ and returns combined string."""
    rules_text = ""
    if os.path.exists(RULES_DIR):
        files = glob.glob(os.path.join(RULES_DIR, "*.md"))
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    rules_text += f"\n\n--- RULE: {os.path.basename(f)} ---\n"
                    rules_text += file.read()
            except Exception as e:
                print(f"Error loading rule {f}: {e}")
    return rules_text

# --- PART 2: MCP CLIENT (Direct function calls) ---
class MCPClient:
    def __init__(self, script_path=None):
        # No longer needed but keeping for compatibility
        pass
    
    def start(self):
        """No-op - tools are imported directly"""
        if _searx_available:
            print("[MCP] Searx tools ready")
        else:
            print("[MCP] Searx tools not available")
    
    def call_tool(self, name, arguments):
        """Call tool functions directly instead of subprocess"""
        if not _searx_available:
            return {"error": "Searx tools not available"}
        
        try:
            print(f"[MCP] Calling {name} with args: {arguments}")
            
            if name == "potatool_web_search_urls":
                result = potatool_web_search_urls(**arguments)
            elif name == "potatool_extract_content":
                result = potatool_extract_content(**arguments)
            else:
                return {"error": f"Unknown tool: {name}"}
            
            print(f"[MCP] Success: {str(result)[:200]}...")
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

MCP_SCRIPT_PATH = r"POTATO\MCP\searx_mcp.py"
_mcp_client = MCPClient(MCP_SCRIPT_PATH)

# --- PART 4: CHAT LOGIC ---

mcp_tools_schema = [
    {
        'type': 'function',
        'function': {
            'name': 'potatool_web_search_urls',
            'description': 'Search the web for information.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string'}, 'num_results': {'type': 'integer'}},
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'potatool_extract_content',
            'description': 'Extract text from a URL.',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string'}},
                'required': ['url']
            }
        }
    }
]

def simple_stream_test(messages, model="qwen3-vl:8b", enable_search=False, stealth_mode=False):
    """
    Generator that handles the LLM Stream + Tool execution loop with thinking detection.
    
    Args:
        messages: Chat history
        model: Model name
        enable_search: Whether web search tools are available
        stealth_mode: Whether to operate in stealth mode (affects tool selection)
    """
    # Auto-detect tags if this is first time using model
    if needs_tag_detection(model):
        auto_detect_model_tags(model)
    
    # Get model-specific thinking tags
    thinking_tags = get_model_thinking_tags(model)
    print(f"[MODEL] Using {model}, thinking tags: {thinking_tags}")
    
    # Yield model metadata at start of stream
    yield {
        'metadata': {
            'model': model,
            'thinking_tags': thinking_tags,
            'web_search': enable_search,
            'stealth_mode': stealth_mode
        }
    }
    # 1. Build system context with current settings
    web_search_instructions = """You have access to web search tools via Ollama's native tool calling system.

**AVAILABLE TOOLS:**
- potatool_web_search_urls: Search the web and get URLs with snippets
- potatool_extract_content: Extract full text content from a URL

**CRITICAL TOOL USAGE RULES:**
1. USE NATIVE TOOL CALLING - Ollama will automatically format your tool calls
2. Keep search queries SIMPLE and SHORT (2-5 words) - NO search operators like "site:", "OR", etc.
   ✓ CORRECT: "python tutorials 2026"
   ✗ WRONG: "python tutorials site:example.com OR site:tutorial.com"
3. DO NOT explain what you're going to search for - JUST CALL THE TOOLS IMMEDIATELY
   ✗ WRONG: "Let me search for information about X..."
   ✓ CORRECT: [calls tool directly without announcement]
4. When user asks a question requiring current information, call tools FIRST before responding

**SEARCH QUERY GUIDELINES:**
- Use natural language keywords only
- 2-5 words maximum
- No special operators (site:, OR, AND, quotes, etc.)
- Example: "weather paris" not "weather site:weather.com paris france"

**WORKFLOW:**
1. User asks question → Call potatool_web_search_urls IMMEDIATELY (no explanation)
2. Get results → Call potatool_extract_content on best URL
3. Read content → Formulate answer using the information"""
    
    no_search_instructions = """Web search is DISABLED. Your knowledge may be outdated (current date: 2026). 

**IMPORTANT:** Even without web search, you should STILL ANSWER the user's question to the best of your ability using your training data and conversation history. Briefly mention that your information might not be current or up-to-date, but provide the most helpful answer you can from your existing knowledge."""
    
    stealth_note = "STEALTH MODE: Avoid tools that leave traces or logs. Prioritize privacy-focused operations." if stealth_mode else ""
    
    settings_context = f"""
[CURRENT SYSTEM SETTINGS]
- Web Search: {'ENABLED' if enable_search else 'DISABLED'}
- Stealth Mode: {'ENABLED' if stealth_mode else 'DISABLED'}
- Current Date: February 2026

{web_search_instructions if enable_search else no_search_instructions}
{stealth_note}

ALWAYS consider the full conversation history when determining what to search for or how to respond.
"""
    
    # 2. Inject Rules ONLY if web search is enabled (to avoid conflicting instructions)
    if enable_search:
        rules = load_rules()
        if rules:
            settings_context += f"\n[OPERATIONAL RULES]{rules}"
            print(f"[RULES] Loaded {len(rules)} characters of rules")
        else:
            print("[RULES] No rules found in rules directory")
    
    # 3. Update or create system prompt
    sys_idx = next((i for i, m in enumerate(messages) if m['role'] == 'system'), -1)
    if sys_idx >= 0:
        # Update existing system prompt with current settings
        messages[sys_idx]['content'] += f"\n\n{settings_context}"
    else:
        # Create new system prompt
        messages.insert(0, {"role": "system", "content": settings_context})

    active_tools = mcp_tools_schema if enable_search else None
    
    # Start MCP client if web search is enabled
    if enable_search:
        try:
            _mcp_client.start()
            print("[MCP] Client started successfully")
        except Exception as e:
            print(f"[MCP] Failed to start client: {e}")
            # Continue without tools if MCP fails
            active_tools = None
    
    # 2. Start Chat
    stream = ollama.chat(
        model=model, 
        messages=messages, 
        tools=active_tools, 
        stream=True,
        options={
            'repeat_penalty': 1.02,
            'num_gpu': 999  # Load all layers to GPU
        }
    )

    final_content = ""
    accumulated_text = ""  # Buffer for detecting tool calls in text
    has_tool_intent = False
    
    def parse_text_tool_calls(text):
        """Parse tool calls from text output - supports multiple formats"""
        import re
        
        # First, extract content from thinking tags if present
        thinking_open = thinking_tags.get('open', '')
        thinking_close = thinking_tags.get('close', '')
        if thinking_open and thinking_close and thinking_open in text:
            # Look for tool calls after thinking ends
            try:
                parts = text.split(thinking_close)
                if len(parts) > 1:
                    # Check content after thinking tags
                    after_thinking = parts[-1]
                    if '[TOOL CALL:' in after_thinking or '{' in after_thinking:
                        text = after_thinking
            except:
                pass
        
        # Format 1: [TOOL CALL: tool_name] or [TOOL CALL: tool_name{...}]
        # Examples: [TOOL CALL: potatool_web_search_urls{"query":"Deniz Honigs"}]
        #          [TOOL CALL: potatool_web_search_urls]
        simple_match = re.search(r'\[TOOL CALL:\s*(\w+)(?:\s*(\{[^}]+\}))?\s*\]', text)
        if simple_match:
            tool_name = simple_match.group(1)
            json_args = simple_match.group(2)
            
            args = {}
            
            # If JSON arguments provided inline, parse them
            if json_args:
                try:
                    args = json.loads(json_args)
                    print(f"[TOOL PARSER] Found [TOOL CALL: {tool_name}] with inline JSON args: {args}")
                except Exception as e:
                    print(f"[TOOL PARSER] Failed to parse inline JSON: {e}")
                    # Try to extract from text after tool call
                    remaining = text[simple_match.end():].strip()
                    if tool_name == 'potatool_web_search_urls':
                        query_match = re.search(r'["\']([^"]+)["\']', remaining)
                        if query_match:
                            args['query'] = query_match.group(1)
            else:
                # No inline JSON, extract from text after tool call
                print(f"[TOOL PARSER] Found [TOOL CALL: {tool_name}], looking for args in text")
                remaining = text[simple_match.end():].strip()
                
                if tool_name == 'potatool_web_search_urls':
                    # Try multiple patterns
                    query_match = (re.search(r'query[:\s]+["\']([^"\n]+)["\']', remaining, re.IGNORECASE) or
                                  re.search(r'search[:\s]+["\']([^"\n]+)["\']', remaining, re.IGNORECASE) or
                                  re.search(r'["\']([^"]+)["\']', remaining))  # Any quoted string
                    if query_match:
                        args['query'] = query_match.group(1).strip()
                        print(f"[TOOL PARSER] Extracted query: {args['query']}")
                    elif messages and messages[-1]['role'] == 'user':
                        # Fallback: use user's query
                        args['query'] = messages[-1]['content'][:200]
                        print(f"[TOOL PARSER] Using fallback query: {args['query'][:50]}...")
                
                elif tool_name == 'potatool_extract_content':
                    url_match = re.search(r'(?:url|link)[\s:]+["\']?([^"\s]+)["\']?', remaining, re.IGNORECASE)
                    if url_match:
                        args['url'] = url_match.group(1).strip('"\'')
            
            return [{
                'function': {
                    'name': tool_name,
                    'arguments': args
                }
            }]
        
        # Format 2: JSON format [{"name": "potatool_web_search_urls", "arguments": {...}}]
        # Also handles multiple separate JSON objects on different lines
        # SECURITY: Only parse tool calls with potatool_ prefix to prevent accidental execution
        if '"name"' in text and '"arguments"' in text and 'potatool_' in text:
            try:
                json_str = None
                
                # First check if JSON is in a markdown code block
                code_block_match = re.search(r'```(?:json)?\s*\n?(\{[^`]+\})\s*\n?```', text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1).strip()
                    # Wrap single object in array
                    if json_str.startswith('{'):
                        json_str = '[' + json_str + ']'
                    print(f"[TOOL PARSER] Found tool call in code block (model ignored instructions!)")
                else:
                    # Try to find and parse multiple separate JSON objects
                    # Look for all occurrences of {"name": "potatool_
                    json_objects = []
                    search_pos = 0
                    
                    # First, clean up common model mistakes:
                    # - Remove trailing ] after JSON object: {"name":...}]
                    # - Remove markdown-style formatting
                    text_cleaned = text
                    # Remove trailing ] that some models add
                    text_cleaned = re.sub(r'(\{[^}]+\})\]\s*$', r'\1', text_cleaned)
                    
                    while True:
                        # Find next potential JSON object start
                        start_pos = text_cleaned.find('{"name"', search_pos)
                        if start_pos == -1:
                            start_pos = text_cleaned.find('{\"name\"', search_pos)
                        if start_pos == -1:
                            break
                        
                        # Check if it's a potatool_ call
                        if 'potatool_' not in text_cleaned[start_pos:start_pos+100]:
                            search_pos = start_pos + 1
                            continue
                        
                        # Try to parse JSON from this position with increasing lengths
                        found_valid_json = False
                        for end_offset in range(50, min(len(text_cleaned) - start_pos, 2000), 50):
                            try:
                                candidate = text_cleaned[start_pos:start_pos+end_offset]
                                # Find the last complete } 
                                last_brace = candidate.rfind('}')
                                if last_brace == -1:
                                    continue
                                candidate = candidate[:last_brace+1]
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and 'name' in obj and obj['name'].startswith('potatool_'):
                                    json_objects.append(candidate)
                                    search_pos = start_pos + last_brace + 1
                                    found_valid_json = True
                                    break
                            except json.JSONDecodeError:
                                continue
                        
                        if not found_valid_json:
                            # Couldn't parse, move past this start
                            search_pos = start_pos + 1
                    
                    if json_objects:
                        json_str = '[' + ','.join(json_objects) + ']'
                        print(f"[TOOL PARSER] Found {len(json_objects)} separate JSON tool call(s)")
                    else:
                        # Find JSON array or object in plain text
                        json_start = text.find('[{')
                        if json_start == -1:
                            json_start = text.find('{')
                            if json_start == -1:
                                return None
                            # Single object, wrap in array
                            json_end = text.rfind('}') + 1
                            json_str = '[' + text[json_start:json_end] + ']'
                        else:
                            # Array found
                            json_end = text.find(']', json_start) + 1
                            json_str = text[json_start:json_end]
                
                print(f"[TOOL PARSER] Attempting to parse JSON: {json_str[:200]}")
                parsed = json.loads(json_str)
                
                # Normalize to list
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                # Convert to ollama format - ONLY accept tool calls with potatool_ prefix
                tool_calls = []
                for tc in parsed:
                    if 'name' in tc:
                        tool_name = tc.get('name')
                        # SECURITY: Only execute if tool has potatool_ prefix
                        if tool_name.startswith('potatool_'):
                            tool_calls.append({
                                'function': {
                                    'name': tool_name,
                                    'arguments': tc.get('arguments', {})
                                }
                            })
                        else:
                            print(f"[TOOL PARSER] Rejected tool without potatool_ prefix: {tool_name}")
                
                if tool_calls:
                    print(f"[TOOL PARSER] Extracted {len(tool_calls)} tool calls from JSON")
                return tool_calls if tool_calls else None
            except Exception as e:
                print(f"[TOOL PARSER] JSON parse error: {e}")
                return None
        return None
    
    try:
        for chunk in stream:
            msg = chunk.get('message', {})
            is_done = chunk.get('done', False)
            
            # A. Handle Native Tool Calls (proper Ollama format)
            if 'tool_calls' in msg and msg['tool_calls']:
                print(f"[TOOL] Native Ollama tool call detected: {msg['tool_calls']}")
                for tool in msg['tool_calls']:
                    try:
                        fn = tool.function.name
                        args = tool.function.arguments
                        
                        yield {'tool': f"MCP: {fn}..."}
                        
                        # Parse args if string
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except: pass
                        
                        # Call MCP
                        try:
                            result = _mcp_client.call_tool(fn, args)
                        except Exception as e:
                            result = {"error": str(e)}

                        # Update history for ReAct - keep arguments as dict
                        tool_dict = {
                            'type': 'function',
                            'function': {
                                'name': fn,
                                'arguments': args  # Keep as dict, not JSON string
                            }
                        }
                        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool_dict]})
                        messages.append({'role': 'tool', 'content': json.dumps(result), 'name': fn})
                        
                        # Recursion: Call simple_stream_test() again to handle potential additional tool calls
                        yield {'tool': "Analyzing..."}
                        sub_stream = simple_stream_test(messages, model=model, enable_search=enable_search, stealth_mode=stealth_mode)
                        
                        # Stream sub-response - this will recursively handle MORE tool calls if present
                        for sub in sub_stream:
                            if 'content' in sub:
                                final_content += sub['content']
                            yield sub
                    except Exception as tool_err:
                        # Handle XML parsing errors and other tool execution errors
                        error_msg = f"Tool execution error: {str(tool_err)}"
                        yield {'content': f"\n\n[Error: {error_msg}]\n\n"}
                        print(f"[ERROR] {error_msg}")
                return # End logic after tools processed

            # B. Handle Standard Content
            if 'content' in msg:
                c = msg['content']
                
                # BUFFER content when web search is enabled to detect tool calls
                if enable_search:
                    accumulated_text += c
                    
                    # Check for complete tool call patterns with REQUIRED potatool_ prefix
                    # Format 1: [TOOL CALL: name{...}]
                    # Format 2: [{"name":"potatool_web_search_urls","arguments":{...}}]
                    # Format 3: {"name":"potatool_web_search_urls",...} (single object per line)
                    # Format 4: ```json\n{"name":"potatool_...",...}\n``` (when model uses code blocks)
                    # SECURITY: Must have potatool_ prefix to prevent accidental execution
                    has_bracket_format = '[TOOL CALL:' in accumulated_text and ']' in accumulated_text
                    has_json_array_format = ('[{' in accumulated_text or '[\n{' in accumulated_text) and '"name"' in accumulated_text and '"arguments"' in accumulated_text and 'potatool_' in accumulated_text and ']' in accumulated_text
                    has_json_object_format = ('{' in accumulated_text and '}' in accumulated_text) and '"name"' in accumulated_text and '"arguments"' in accumulated_text and 'potatool_' in accumulated_text
                    has_code_block_format = '```' in accumulated_text and accumulated_text.count('```') >= 2 and '"name"' in accumulated_text and '"arguments"' in accumulated_text and 'potatool_' in accumulated_text
                    
                    if has_bracket_format or has_json_array_format or has_json_object_format or has_code_block_format:
                        detected_tools = parse_text_tool_calls(accumulated_text)
                        
                        if detected_tools:
                            # SUCCESS - Found complete tool call!
                            print(f"[TOOL EXECUTION] Executing: {detected_tools}")
                            # DON'T yield thinking text - just execute tools silently
                            accumulated_text = ""
                            
                            # Execute tools
                            for tool in detected_tools:
                                fn = tool['function']['name']
                                args = tool['function']['arguments']
                                
                                # Yield detailed tool info for frontend (Thinking & Tools section)
                                tool_display_name = fn.replace('potatool_', '').replace('_', ' ').title()
                                yield {'tool': f"🔧 Calling {tool_display_name}...", 'tool_name': fn, 'tool_args': args}
                                print(f"[MCP] Calling {fn} with args: {args}")
                                
                                try:
                                    result = _mcp_client.call_tool(fn, args)
                                    print(f"[MCP] Result: {str(result)[:300]}")
                                    
                                    # Yield success message
                                    yield {'tool': f"✓ {tool_display_name} completed", 'tool_result': str(result)[:100]}
                                    
                                    # Check if tool failed
                                    if isinstance(result, dict) and 'error' in result:
                                        print(f"[MCP] Tool failed: {result['error']} - skipping this tool")
                                        # Tool failed - continue to next tool instead of breaking
                                        continue
                                    
                                except Exception as e:
                                    result = {"error": str(e)}
                                    print(f"[MCP] Error: {e} - skipping this tool")
                                    # Tool failed - continue to next tool instead of breaking
                                    continue
                                
                                # Tool succeeded - add to history
                                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool]})
                                messages.append({'role': 'tool', 'content': json.dumps(result), 'name': fn})
                            
                            # After ALL tools executed, synthesize response with results
                            # Only continue with tool results if we actually got at least one success
                            if messages and messages[-1].get('role') == 'tool':
                                yield {'tool': "📊 Analyzing results..."}
                                # Recursively handle sub-stream - model might make MORE tool calls!
                                # Call the same function recursively to handle potential additional tool calls
                                sub_stream = simple_stream_test(messages, model=model, enable_search=enable_search, stealth_mode=stealth_mode)
                                
                                for sub in sub_stream:
                                    # Forward all chunks from sub-stream (content, tool, thinking)
                                    if 'content' in sub:
                                        final_content += sub['content']
                                    yield sub
                                
                                return
                            else:
                                # Tool failed - restart stream WITHOUT tool results to get natural response
                                print("[TOOL EXECUTION] Tool failed, generating response without tool results")
                                sub_stream = ollama.chat(model=model, messages=messages, tools=None, stream=True)
                                
                                for sub in sub_stream:
                                    sub_msg = sub.get('message', {})
                                    if 'content' in sub_msg:
                                        final_content += sub_msg['content']
                                        yield {'content': sub_msg['content']}
                                    if 'thinking' in sub_msg:
                                        yield {'thinking': sub_msg['thinking']}
                                
                                return
                        else:
                            # Has pattern but parser returned None - likely false positive
                            # Stream ended or buffer too large, yield as content
                            if is_done or len(accumulated_text) > 1000:
                                print(f"[TOOL DETECTION] Pattern found but no valid tool call, yielding as content")
                                final_content += accumulated_text
                                yield {'content': accumulated_text}
                                accumulated_text = ""
                    else:
                        # No complete pattern yet - yield frequently for smooth streaming
                        # Only buffer when we detect the START of a potential tool call
                        potential_tool_start = ('{' in accumulated_text[-20:] and '"' in accumulated_text[-20:]) or \
                                              '[' in accumulated_text[-10:] or \
                                              '`' in accumulated_text[-10:]
                        
                        # Check if this might be the start of a tool call
                        # More aggressive: if we see 'potatool_' anywhere, hold the buffer
                        might_be_tool_call = 'potatool_' in accumulated_text or \
                                            ('"name"' in accumulated_text and '"arguments"' in accumulated_text)
                        
                        if is_done:
                            # Stream ended, check one last time for tool calls before yielding
                            if might_be_tool_call:
                                detected_tools = parse_text_tool_calls(accumulated_text)
                                if detected_tools:
                                    print(f"[TOOL EXECUTION] Found tool call at stream end: {detected_tools}")
                                    # Execute tools (same logic as above)
                                    for tool in detected_tools:
                                        fn = tool['function']['name']
                                        args = tool['function']['arguments']
                                        tool_display_name = fn.replace('potatool_', '').replace('_', ' ').title()
                                        yield {'tool': f"🔧 Calling {tool_display_name}...", 'tool_name': fn, 'tool_args': args}
                                        print(f"[MCP] Calling {fn} with args: {args}")
                                        try:
                                            result = _mcp_client.call_tool(fn, args)
                                            yield {'tool': f"✓ {tool_display_name} completed", 'tool_result': str(result)[:100]}
                                            if isinstance(result, dict) and 'error' in result:
                                                continue
                                        except Exception as e:
                                            result = {"error": str(e)}
                                            continue
                                        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool]})
                                        messages.append({'role': 'tool', 'content': json.dumps(result), 'name': fn})
                                    if messages and messages[-1].get('role') == 'tool':
                                        yield {'tool': "📊 Analyzing results..."}
                                        sub_stream = simple_stream_test(messages, model=model, enable_search=enable_search, stealth_mode=stealth_mode)
                                        for sub in sub_stream:
                                            if 'content' in sub:
                                                final_content += sub['content']
                                            yield sub
                                        return
                                    accumulated_text = ""
                            # No tool call found, yield as normal content
                            if accumulated_text:
                                final_content += accumulated_text
                                yield {'content': accumulated_text}
                                accumulated_text = ""
                        elif len(accumulated_text) > 100 and not potential_tool_start and not might_be_tool_call:
                            # Buffer getting large and no tool call starting - yield it
                            final_content += accumulated_text
                            yield {'content': accumulated_text}
                            accumulated_text = ""
                        elif len(accumulated_text) > 20 and not potential_tool_start:
                            # Small buffer with no tool call indicators - yield for smooth streaming
                            final_content += accumulated_text
                            yield {'content': accumulated_text}
                            accumulated_text = ""
                else:
                    # Web search disabled - yield immediately
                    final_content += c
                    yield {'content': c}
            
            # C. Handle Thinking (if model outputs it directly)
            if 'thinking' in msg:
                yield {'thinking': msg['thinking']}
            
            # D. Yield any remaining accumulated text when stream ends
            if is_done and accumulated_text:
                final_content += accumulated_text
                yield {'content': accumulated_text}
                accumulated_text = ""
    except Exception as stream_err:
        # Catch XML parsing errors from Ollama
        error_msg = f"Stream error: {str(stream_err)}"
        yield {'error': error_msg}
        print(f"[ERROR] {error_msg}")
    
    # Warn if model talked about using tools but didn't actually call them
    if has_tool_intent and not final_content:
        warning = "\n\n**[SYSTEM WARNING: Model expressed intent to use tools but did not generate actual tool calls. This model may not properly support tool calling. Try a different model like qwen2.5:14b or similar.]**"
        yield {'content': warning}
        if 'XML syntax error' in str(stream_err) or 'failed to parse XML' in str(stream_err):
            error_msg = "Model generated malformed tool call. Try disabling web search or using a different model."
        else:
            error_msg = f"Stream error: {str(stream_err)}"
        yield {'content': f"\n\n[Error: {error_msg}]\n\n"}
        print(f"[ERROR] {error_msg}")

def main():
    print("POTATO Assistant is ready. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
              
            if not query:
                continue
              
            print(f"Processing query: {query}")
            
            # Get system info
            system_info = json_get_instant_system_info()
            print("System Information:")
            for key, value in system_info.items():
                print(f"  {key}: {value}")
            print()
            
            # Process the query
            # response = process_user_query(query)  # TODO: Define this function
            print(f"Response: [Test mode not implemented]")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue



# TEST; UPDATE LATER
# --- Define Tools Schema ---
# You can extend this list based on your components
tools_schema = [
    # {
    #     'type': 'function',
    #     'function': {
    #         'name': 'get_current_weather',
    #         'description': 'Get the current weather for a location',
    #         'parameters': {
    #             'type': 'object',
    #             'properties': {
    #                 'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'},
    #             },
    #             'required': ['location'],
    #         },
    #     },
    # },
    # Add your other tools here (e.g., from components.visual_tools)
]

# --- MODEL INFO MANAGEMENT ---

def load_model_info():
    """Load model information from JSON file"""
    try:
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[MODEL_INFO] Error loading model info: {e}")
    return {}

def save_model_info(model_info):
    """Save model information to JSON file"""
    try:
        os.makedirs(os.path.dirname(MODEL_INFO_PATH), exist_ok=True)
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f, indent=2)
    except Exception as e:
        print(f"[MODEL_INFO] Error saving model info: {e}")

def update_model_thinking_capability(model_name, can_think):
    """Update a specific model's thinking capability"""
    model_info = load_model_info()
    
    if model_name not in model_info:
        model_info[model_name] = {"can_think": "unknown", "tool_calls": False}
    
    model_info[model_name]["can_think"] = can_think
    save_model_info(model_info)
    print(f"[MODEL_INFO] Updated {model_name}: can_think={can_think}")

def get_model_thinking_capability(model_name):
    """Get model's thinking capability. Returns True/False/"unknown" """
    model_info = load_model_info()
    if model_name in model_info:
        cap = model_info[model_name].get("can_think", "unknown")
        # Handle string "true"/"false" from JSON
        if cap == "true": return True
        if cap == "false": return False
        return cap
    return "unknown"

def get_model_thinking_tags(model_name):
    """
    Get model-specific thinking tags. Returns dict with 'open' and 'close' tags.
    Defaults to <think></think> if not detected.
    """
    model_info = load_model_info()
    if model_name in model_info and 'thinking_tags' in model_info[model_name]:
        return model_info[model_name]['thinking_tags']
    # Default tags
    return {'open': '<think>', 'close': '</think>'}

def needs_tag_detection(model_name):
    """
    Check if a model needs tag detection to be run.
    Returns True if model doesn't exist or doesn't have detection data.
    """
    model_info = load_model_info()
    
    # Model doesn't exist in modelinfos.json
    if model_name not in model_info:
        return True
    
    # Model exists but no detection data
    if 'thinking_tags' not in model_info[model_name]:
        return True
    
    # Has detection data, no need to re-detect
    return False

def auto_detect_model_tags(model_name):
    """
    Automatically run tag detection for a model if needed.
    Called when model is first used without detection data.
    """
    if not needs_tag_detection(model_name):
        print(f"[DETECTION] {model_name} already has tag info, skipping detection")
        return
    
    print(f"[DETECTION] First time using {model_name}, running auto-detection...")
    
    try:
        # Import detection function
        from POTATO.components.utilities.initial_model_inference_setup import update_model_with_tags
        
        # Run detection
        success = update_model_with_tags(model_name)
        
        if success:
            print(f"[DETECTION] Auto-detection complete for {model_name}")
        else:
            print(f"[DETECTION] Auto-detection failed for {model_name}, using defaults")
    except Exception as e:
        print(f"[DETECTION] Error during auto-detection: {e}")
        print(f"[DETECTION] Using default tags for {model_name}")

def detect_thinking_capability(test_response):
    """
    Detect if a model supports thinking by checking for <think> tags in response.
    Returns True if thinking tags detected, False otherwise.
    """
    if not test_response:
        return False
    
    # Check if response contains <think> tags
    has_thinking = '<think>' in test_response and '</think>' in test_response
    return has_thinking


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import subprocess
        subprocess.run(["ollama", "stop", "gpt-oss:20b"])
        print("Exiting...")
        import sys 
        sys.exit(1)
        print("Bye!")
