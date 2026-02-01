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

# --- PART 2: MCP CLIENT (STDIO WRAPPER) ---
class MCPClient:
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.lock = threading.Lock()

    def start(self):
        """Starts the MCP server subprocess."""
        if self.process and self.process.poll() is None:
            return 
        
        if not os.path.exists(self.script_path):
            print(f"MCP Error: Script not found at {self.script_path}")
            return

        # Start python script with unbuffered binary stdio
        self.process = subprocess.Popen(
            ["python", self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1 
        )
        print(f"MCP Server Started: {self.script_path}")

    def call_tool(self, name, arguments):
        """Sends a JSON-RPC request to the running MCP subprocess."""
        with self.lock:
            if not self.process or self.process.poll() is not None:
                self.start()
                if not self.process: return {"error": "Failed to start MCP"}

            request_id = 1
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
                "id": request_id
            }

            try:
                json_str = json.dumps(payload)
                self.process.stdin.write(json_str + "\n")
                self.process.stdin.flush()

                while True:
                    line = self.process.stdout.readline()
                    if not line: break
                    try:
                        response = json.loads(line)
                        if response.get("id") == request_id:
                            if "result" in response: return response["result"]
                            if "error" in response: return {"error": response["error"]}
                    except json.JSONDecodeError:
                        continue 
            except Exception as e:
                return {"error": f"MCP IPC Error: {str(e)}"}
            return {"error": "No response"}

MCP_SCRIPT_PATH = "POTATO\MCP\searx_mcp.py"
_mcp_client = MCPClient(MCP_SCRIPT_PATH)

# --- PART 4: CHAT LOGIC ---

mcp_tools_schema = [
    {
        'type': 'function',
        'function': {
            'name': 'web_search_urls',
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
            'name': 'extract_content',
            'description': 'Extract text from a URL.',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string'}},
                'required': ['url']
            }
        }
    }
]

def simple_stream_test(messages, model="qwen3-vl:8b", enable_search=False):
    """
    Generator that handles the LLM Stream + Tool execution loop.
    """
    # 1. Inject Rules
    rules = load_rules()
    if rules:
        # Check if system prompt exists
        sys_idx = next((i for i, m in enumerate(messages) if m['role'] == 'system'), -1)
        if sys_idx >= 0: messages[sys_idx]['content'] += f"\n\n[OPERATIONAL RULES]{rules}"
        else: messages.insert(0, {"role": "system", "content": f"[OPERATIONAL RULES]{rules}"})

    active_tools = mcp_tools_schema if enable_search else None
    
    # 2. Start Chat
    stream = ollama.chat(model=model, messages=messages, tools=active_tools, stream=True)

    final_content = ""
    
    for chunk in stream:
        msg = chunk.get('message', {})
        
        # A. Handle Tool Calls (ReAct)
        if 'tool_calls' in msg and msg['tool_calls']:
            for tool in msg['tool_calls']:
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

                # Update history for ReAct
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool]})
                messages.append({'role': 'tool', 'content': json.dumps(result), 'name': fn})
                
                # Recursion: Call LLM again with tool output
                yield {'tool': "Analyzing..."}
                sub_stream = ollama.chat(model=model, messages=messages, stream=True)
                
                for sub in sub_stream:
                    if 'content' in sub.get('message', {}):
                        c = sub['message']['content']
                        final_content += c
                        yield {'content': c}
            return # End logic after tools processed

        # B. Handle Standard Content
        if 'content' in msg:
            c = msg['content']
            final_content += c
            yield {'content': c}

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
            response = process_user_query(query)
            print(f"Response: {response}")
            
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

def simple_stream_test(messages_history, stream=False, think=False):
    """
    Unified chat function that handles both streaming and non-streaming.
    """
    response = ollama.chat(
        # model="qwen2.5:32b", # Ensure model name matches your Ollama library
        model="qwen3-vl:8b",
        messages=messages_history,
        # tools=tools_schema, # Inject tools
        stream=stream,
        think=think,
        keep_alive=600,
    )
    return response


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
