# gets n8n workflows from api. n8n is running on docker locally
import requests
import os
from dotenv import load_dotenv
import time
# Load .env from POTATO folder
dotenv_path = os.path.join("POTATO", ".local.env")
# print("Loading dotenv from:", dotenv_path)
loaded = load_dotenv(dotenv_path, override=True)

if not loaded:
    print(f"Warning: Could not load .env file at {dotenv_path}. Make sure it exists and is properly formatted.")

API_KEY = os.getenv("N8N_API_KEY")
# print(f"Loaded N8N_API_KEY: {API_KEY}")  # Debug print to verify API key is loaded

def get_n8n_workflow_summaries(n8n_url="http://localhost:5678", api_key=None):
    if api_key is None:
        api_key = os.getenv("N8N_API_KEY")

    if not api_key:
        raise RuntimeError("N8N_API_KEY is NOT set in POTATO/.local.env. Please set it to your n8n API key.")

    url = f"{n8n_url}/api/v1/workflows"
    headers = {
        "Accept": "application/json",
        "X-N8N-API-KEY": api_key,
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching n8n workflows: {e}")
        return []

    data = response.json()
    # print("Raw response from n8n API:", data)  # Debug: see what structure API returns

    # Handle different response structures
    if isinstance(data, dict) and "data" in data:
        workflows = data["data"]
    elif isinstance(data, list):
        workflows = data
    else:
        print("Unexpected API response format")
        return []

    # Ensure each workflow is a dict with id, name, active, description, and tags
    summaries = []
    for wf in workflows:
        if isinstance(wf, dict):
            summaries.append({
                "id": wf.get("id"),
                "name": wf.get("name"),
                "active": wf.get("active", False),          # Defaults to False if not set
                "description": str(wf.get("description")),       # Can be None if not set
                "tags": wf.get("tags", [])                  # Defaults to empty list if not set
            })
        else:
            # fallback if workflow is just a string
            summaries.append({
                "id": None,
                "name": str(wf),
                "active": False,
                "description": None,
                "tags": []
            })

    return summaries


def get_n8n_workflow_details(workflow_id, n8n_url="http://localhost:5678", api_key=None):
    if api_key is None:
        api_key = os.getenv("N8N_API_KEY")

    if not api_key:
        raise RuntimeError("N8N_API_KEY is NOT set in POTATO/.local.env. Please set it to your n8n API key.")

    url = f"{n8n_url}/api/v1/workflows/{workflow_id}"
    headers = {
        "Accept": "application/json",
        "X-N8N-API-KEY": api_key,
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching n8n workflow details for ID {workflow_id}: {e}")
        return None
    return response.json()

def execute_n8n_workflow(
    workflow_id,
    n8n_url="http://localhost:5678",
    api_key=None,
    input_data=None,
    headers_override=None,
):
    """
    Trigger an n8n workflow's webhook URL and return its JSON response.

    – Uses the workflow's webhook trigger URL defined in the workflow.
    – Returns the JSON response the webhook returns.
    """

    # Resolve API key for protected workflows (will only be needed to fetch workflow metadata)
    if api_key is None:
        api_key = os.getenv("N8N_API_KEY")
    if not api_key:
        raise RuntimeError("N8N_API_KEY must be set (env or passed in).")

    # Headers to use when asking n8n API for workflow details
    metadata_headers = {
        "Accept": "application/json",
        "X-N8N-API-KEY": api_key,
    }

    # Step 1: Fetch workflow metadata to find webhook info
    try:
        wf_resp = requests.get(
            f"{n8n_url}/api/v1/workflows/{workflow_id}",
            headers=metadata_headers,
            timeout=10,
        )
        wf_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch workflow metadata: {e}"}

    workflow_json = wf_resp.json()

    # Step 2: Extract webhook path from workflow JSON
    webhook_path = None

    # n8n workflow nodes are found under "nodes" list
    for node in workflow_json.get("nodes", []):
        # Identify a webhook trigger node
        # Usually webhook trigger nodes have type "n8n-nodes-base.webhook" in workflow JSON
        if node.get("type") and "webhook" in node.get("type").lower():
            # The webhook “path” property is usually in node["parameters"]["path"]
            params = node.get("parameters", {})
            webhook_path = params.get("path") or params.get("webhookId") or params.get("webhookPath")
            if webhook_path:
                break

    if not webhook_path:
        return {"error": "No webhook path found in workflow – ensure the workflow has a Webhook Trigger node."}

    # Step 3: Construct full webhook URL
    # n8n webhook URLs use /webhook/<path>
    webhook_url = f"{n8n_url.rstrip('/')}/webhook/{webhook_path}"

    # Optional extra headers for the webhook call (e.g., authentication headers if your webhook requires them)
    post_headers = {"Content-Type": "application/json"}
    if headers_override:
        post_headers.update(headers_override)

    # Step 4: Send request to webhook URL
    try:
        response = requests.post(webhook_url, json=input_data or {}, headers=post_headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error executing webhook: {e}"}

    # Step 5: Try to return JSON from the webhook response
    try:
        result = response.json()
    except ValueError:
        result = {"raw_response": response.text}

    print(f"Workflow '{workflow_json.get('name',workflow_id)}' executed successfully via webhook.")

    return result


if __name__ == "__main__":
    import json
    summaries = get_n8n_workflow_summaries()
    print(json.dumps(summaries, indent=2))
    print("\n" + "="*50 + "\n")
    if summaries and summaries[0]["id"]:
        # details = get_n8n_workflow_details(summaries[-8]["id"])
        details = get_n8n_workflow_details("TZfY2T0j5wuTFeDp")
        print(json.dumps(details, indent=2))
    print("\n" + "="*50 + "\n")
    print("Testing workflow execution...")
    exec_result = execute_n8n_workflow(workflow_id="TZfY2T0j5wuTFeDp")
    print(json.dumps(exec_result, indent=2))

