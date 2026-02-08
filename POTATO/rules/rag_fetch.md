# RAG Fetch Tool

## Purpose

- Provide on-demand retrieval of relevant context for a chat session by querying the vector store (Weaviate).

## Usage

- Endpoint: `/api/rag_fetch`
- Accepts JSON: `{ "session_id": "<chat id>", "query": "<user query>", "top_k": <int, optional> }`
- Returns JSON: `{ "success": true, "results": [ { "source": "<path|id>", "content": "<snippet>", "score": <float> }, ... ] }`

## Behavior and rules for the LLM

- When called, the tool returns short, relevant snippets only (avoid returning full files unless explicitly requested).
- The tool should be used when the model needs external document context to answer user questions, not for general chit-chat.
- The model should call the tool when it detects the user refers to files, attachments, or domain-specific content likely stored in RAG.
- Tool responses are authoritative: the model may quote or cite returned `source` items when producing final answers.

## Examples

- Request: `{"session_id": "abc123", "query": "Summarize the user's uploaded contract about termination clauses", "top_k": 5}`
- Response: `{"success": true, "results": [{"source": "uploads/abc123/contract.pdf", "content": "...termination occurs if...", "score": 0.92}, ...]}`

## Implementation notes for engineers

- Use the existing `components/local_tools/embed.py` helpers (`query_weaviate`) which already support REST/GraphQL fallbacks.
- Keep `top_k` bounded (suggest default 5-10) to avoid overly large system contexts.
- Return `content` truncated to a reasonable length (e.g., 2000 chars) and include `source` and `score` fields.
- Mark tool outputs as safe to be added to the model's system context, not as visible user messages.
