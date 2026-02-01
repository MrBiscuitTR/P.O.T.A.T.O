import subprocess
import json

# Start MCP server
proc = subprocess.Popen(
    ["python", "-u", "POTATO/MCP/searx_mcp.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0
)

# Send initialize request
init = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    },
    "id": 0
}

print("Sending initialize...")
proc.stdin.write(json.dumps(init) + "\n")
proc.stdin.flush()

# Read response
print("Waiting for response...")
response = proc.stdout.readline()
print(f"Init response: {response}")

# Send tool call
tool_call = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "potatool_web_search_urls",
        "arguments": {"query": "test", "num_results": 3}
    },
    "id": 1
}

print("\nSending tool call...")
proc.stdin.write(json.dumps(tool_call) + "\n")
proc.stdin.flush()

# Read response
print("Waiting for tool response...")
response = proc.stdout.readline()
print(f"Tool response: {response[:500]}")

proc.terminate()
