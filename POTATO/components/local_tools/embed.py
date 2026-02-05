#  use nomic-embed-text:latest model from ollama for local embeddings, save vectors in a local vector db like pgvector or weaviate or similar. use docker/podman
# POTATO/components/local_tools/embed.py
# weaviate container running locally at http://127.0.0.1:8081

import weaviate
from weaviate.collections.classes.config import DataType
from weaviate.classes.init import Auth
from weaviate.connect.base import ConnectionParams, ProtocolParams
from ollama import Client as Ollama
import traceback
import re
import requests

# -------------------------------
# Configuration
# -------------------------------
WEAVIATE_URL = "http://127.0.0.1:8081"
DEFAULT_CLASS_NAME = "Document"


def sanitize_collection_name(name):
    """
    Sanitize a string to be a valid Weaviate collection name.
    Weaviate collection names must start with uppercase letter and contain only alphanumeric chars.
    """
    # Remove non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Ensure it starts with uppercase letter
    if sanitized and not sanitized[0].isupper():
        sanitized = 'Chat_' + sanitized
    elif not sanitized:
        sanitized = 'Document'
    # Capitalize first letter
    sanitized = sanitized[0].upper() + sanitized[1:] if len(sanitized) > 1 else sanitized.upper()
    return sanitized


def get_collection_name(chat_id=None):
    """
    Get the collection name for a specific chat.
    Each chat gets its own collection to prevent cross-chat RAG pollution.
    """
    if chat_id:
        # Create unique collection per chat session
        return sanitize_collection_name(f"Chat_{chat_id.replace('-', '_')}")
    return DEFAULT_CLASS_NAME


def create_weaviate_rest_client(host: str = "127.0.0.1", port: int = 8081):
    """
    Create a Weaviate client configured to use REST only by
    disabling the gRPC connection step (monkeypatching open_connection_grpc).
    """
    # Build connection params (grpc params still required by the API)
    conn_params = ConnectionParams(
        http=ProtocolParams(host=host, port=port, secure=False),
        grpc=ProtocolParams(host=host, port=50051, secure=False),
    )

    client = weaviate.WeaviateClient(connection_params=conn_params, skip_init_checks=True)
    # Prevent the client from trying to open a gRPC channel during connect
    try:
        client._connection.open_connection_grpc = lambda colour: None  # type: ignore
        client._connection._grpc_channel = None  # type: ignore
        client._connection._grpc_stub = None  # type: ignore
    except Exception:
        pass

    # Now connect (will perform REST init only)
    client.connect()
    return client


def embed_to_weaviate(texts, chat_id=None, source_filename=None):
    """
    Embed a list of texts using Ollama Nomic embedding model
    and upload vectors to a local Weaviate instance.

    Args:
        texts (str or list of str): Single text string or list of texts.
        chat_id (str, optional): Chat session ID for namespacing. Creates separate collection per chat.
        source_filename (str, optional): Source filename to store as metadata.
    
    Returns:
        dict: Result with status, count, and collection name
    """
    if isinstance(texts, str):
        texts = [texts]
    
    collection_name = get_collection_name(chat_id)

    # 1. Connect to Weaviate using v4 API
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        # Check if collection exists, create if not
        collections = client.collections.list_all()
        
        if collection_name not in collections:
            # Create collection with manual vectorization
            client.collections.create(
                name=collection_name,
                vectorizer_config=None,  # Manual vectorization
                properties=[
                    {
                        "name": "content",
                        "data_type": DataType.TEXT
                    },
                    {
                        "name": "source_file",
                        "data_type": DataType.TEXT
                    },
                    {
                        "name": "chat_id",
                        "data_type": DataType.TEXT
                    },
                    {
                        "name": "content_type",
                        "data_type": DataType.TEXT  # 'text', 'image_ocr', 'pdf'
                    }
                ]
            )
            print(f"Created collection: {collection_name}")
        
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # 3. Initialize Ollama client
        ollama = Ollama()
        
        # 4. Embed texts and upload
        embedded_count = 0
        for text in texts:
            if not text or not text.strip():
                continue
                
            # Get embedding from Ollama
            response = ollama.embeddings(
                model="nomic-embed-text:latest",
                prompt=text
            )
            
            embedding = response['embedding']
            
            # Insert into Weaviate with manual vector
            properties = {
                "content": text,
                "chat_id": chat_id or "",
                "source_file": source_filename or "",
                "content_type": "text"
            }
            
            collection.data.insert(
                properties=properties,
                vector=embedding
            )
            
            embedded_count += 1
            print(f"Uploaded: {text[:50]}...")
        
        print(f"\n{embedded_count} texts uploaded to Weaviate collection '{collection_name}'.")
        client.close()
        
        return {
            "status": "success",
            "embedded_count": embedded_count,
            "collection_name": collection_name
        }
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name
        }


def embed_ocr_content(text, chat_id, source_filename, content_type="image_ocr"):
    """
    Embed OCR-extracted content from images or PDFs.
    
    Args:
        text (str): OCR-extracted text content
        chat_id (str): Chat session ID for namespacing
        source_filename (str): Original filename
        content_type (str): Type of content ('image_ocr', 'pdf')
    
    Returns:
        dict: Result with status
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Empty text content"}
    
    collection_name = get_collection_name(chat_id)
    
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        # Check if collection exists, create if not
        collections = client.collections.list_all()
        
        if collection_name not in collections:
            client.collections.create(
                name=collection_name,
                vectorizer_config=None,
                properties=[
                    {"name": "content", "data_type": DataType.TEXT},
                    {"name": "source_file", "data_type": DataType.TEXT},
                    {"name": "chat_id", "data_type": DataType.TEXT},
                    {"name": "content_type", "data_type": DataType.TEXT}
                ]
            )
            print(f"Created collection: {collection_name}")
        
        collection = client.collections.get(collection_name)
        
        # Get embedding
        ollama = Ollama()
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=text
        )
        
        embedding = response['embedding']
        
        # Insert
        collection.data.insert(
            properties={
                "content": text,
                "chat_id": chat_id,
                "source_file": source_filename,
                "content_type": content_type
            },
            vector=embedding
        )
        
        client.close()
        print(f"Embedded {content_type} content from: {source_filename}")
        
        return {"status": "success", "collection_name": collection_name}
        
    except Exception as e:
        print(f"Error embedding OCR content: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def query_weaviate(query_text, chat_id=None, limit=5):
    """
    Query Weaviate for similar documents using vector search.
    
    Args:
        query_text (str): Query text to search for
        chat_id (str, optional): Chat session ID to search within. If provided, only searches that chat's collection.
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of matching documents with their content and scores
    """
    collection_name = get_collection_name(chat_id)
    
    # Connect to Weaviate and ensure the client is closed
    client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
    try:
        # Check if collection exists
        collections = client.collections.list_all()
        if collection_name not in collections:
            print(f"Collection '{collection_name}' does not exist")
            return []

        # Get collection
        collection = client.collections.get(collection_name)

        # Get query embedding from Ollama
        ollama = Ollama()
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=query_text,
        )

        query_vector = response['embedding']

        # Build vector literal for GraphQL
        vector_literal = "[" + ",".join(map(str, map(float, query_vector))) + "]"
        # Construct GraphQL Get query via concatenation to avoid format-brace issues
        gql = (
            "{ Get { "
            + collection_name
            + "(nearVector: {vector: "
            + vector_literal
            + "}, limit: "
            + str(limit)
            + ") { content source_file content_type _additional { distance } } } }"
        )

        gql_res = client.graphql_raw_query(gql)

        # Normalize GraphQL response into a list of items
        items = []
        try:
            # Case 1: client returns a dict
            if isinstance(gql_res, dict):
                # common shapes: {'data': {'Get': {ClassName: [...]}}} or {'Get': {ClassName: [...]}}
                data_block = gql_res.get('data') or gql_res.get('Get') or gql_res.get('get') or gql_res
                if isinstance(data_block, dict):
                    # try nested 'Get' first
                    get_block = data_block.get('Get') or data_block.get('get') or data_block
                    if isinstance(get_block, dict):
                        items = get_block.get(collection_name, [])
            else:
                # Case 2: client returns a RawGQLReturn-like object with attribute `get`
                if hasattr(gql_res, 'get') and not callable(getattr(gql_res, 'get')):
                    get_block = getattr(gql_res, 'get')
                    if isinstance(get_block, dict):
                        items = get_block.get(collection_name, [])
                else:
                    # Last resort: try to access attribute named 'get' and treat as dict
                    try:
                        maybe = getattr(gql_res, 'get')
                        if isinstance(maybe, dict):
                            items = maybe.get(collection_name, [])
                    except Exception:
                        items = []
        except Exception:
            items = []

        # If still empty, fall back to direct HTTP POST to /v1/graphql and log response for debugging
        if not items:
            try:
                resp = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=10)
                text = resp.text
                try:
                    jr = resp.json()
                except Exception:
                    jr = None
                print(f"[RAG DEBUG] GraphQL fallback response status={resp.status_code}, json_keys={list(jr.keys()) if isinstance(jr, dict) else None}")
                if isinstance(jr, dict):
                    data_block = jr.get('data') or jr.get('Get') or jr.get('get') or jr
                    if isinstance(data_block, dict):
                        get_block = data_block.get('Get') or data_block.get('get') or data_block
                        if isinstance(get_block, dict):
                            items = get_block.get(collection_name, [])
            except Exception as e:
                print(f"[RAG DEBUG] GraphQL fallback error: {e}")

        formatted_results = []
        for item in items:
            props = item or {}
            additional = props.get("_additional", {}) if isinstance(props, dict) else {}
            formatted_results.append({
                'content': props.get('content', ''),
                'source_file': props.get('source_file', ''),
                'content_type': props.get('content_type', 'text'),
                'distance': additional.get('distance') if isinstance(additional, dict) else None,
            })

        return formatted_results
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        traceback.print_exc()
        return []
    finally:
        try:
            client.close()
        except Exception:
            pass


def delete_chat_collection(chat_id):
    """
    Delete a chat's entire collection from Weaviate.
    Called when a chat is deleted.
    
    Args:
        chat_id (str): Chat session ID
        
    Returns:
        bool: True if successful
    """
    collection_name = get_collection_name(chat_id)
    
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        collections = client.collections.list_all()
        if collection_name in collections:
            client.collections.delete(collection_name)
            print(f"Deleted collection: {collection_name}")
        
        client.close()
        return True
        
    except Exception as e:
        # Silently fail if Weaviate isn't running - this is expected
        # when user isn't using RAG features
        if "Connection" in str(e) or "refused" in str(e).lower():
            # Weaviate not running - silently return True (nothing to delete)
            return True
        # Only print errors for unexpected issues
        print(f"Error deleting collection: {e}")
        return False


def delete_file_from_collection(chat_id, filename):
    """
    Delete all embeddings for a specific file from a chat's collection.
    
    Args:
        chat_id (str): Chat session ID
        filename (str): Source filename to delete embeddings for
        
    Returns:
        bool: True if successful
    """
    collection_name = get_collection_name(chat_id)
    
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)

        # Check if collection exists
        collections = client.collections.list_all()
        if collection_name not in collections:
            return True  # Nothing to delete

        # Fetch objects for this class via REST and delete those matching the filename
        url = f"{WEAVIATE_URL}/v1/objects?class={collection_name}&limit=1000"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        deleted_count = 0
        for obj in data.get("objects", []):
            obj_id = obj.get("id") or obj.get("_id") or obj.get("uuid")
            props = obj.get("properties", {}) or {}
            if props.get("source_file") == filename:
                # Delete object by id via REST
                if not obj_id:
                    continue
                del_url = f"{WEAVIATE_URL}/v1/objects/{obj_id}"
                try:
                    dresp = requests.delete(del_url, timeout=10)
                    if dresp.status_code in (200, 204):
                        deleted_count += 1
                except Exception:
                    pass

        print(f"Deleted {deleted_count} embeddings for file: {filename} from {collection_name}")
        return True
    except Exception as e:
        print(f"Error deleting file embeddings: {e}")
        traceback.print_exc()
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def list_embedded_files(chat_id):
    """
    List all unique source files that have been embedded for a specific chat.
    
    Args:
        chat_id (str): Chat session ID
        
    Returns:
        list: List of dicts with filename and content_type info
    """
    collection_name = get_collection_name(chat_id)
    
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        # Check if collection exists
        collections = client.collections.list_all()
        if collection_name not in collections:
            client.close()
            return []
        
        collection = client.collections.get(collection_name)
        
        # Query all objects via REST and extract unique source files
        url = f"{WEAVIATE_URL}/v1/objects?class={collection_name}&limit=1000"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        files_map = {}
        for obj in data.get("objects", []):
            props = obj.get("properties", {}) or {}
            source = props.get('source_file', 'Unknown')
            content_type = props.get('content_type', 'text')

            if source not in files_map:
                files_map[source] = {
                    'filename': source,
                    'content_type': content_type,
                    'chunk_count': 0
                }
            files_map[source]['chunk_count'] += 1
        
        client.close()
        return list(files_map.values())
        
    except Exception as e:
        print(f"Error listing embedded files: {e}")
        traceback.print_exc()
        return []


def list_collections():
    """
    List all collections in Weaviate.
    
    Returns:
        list: List of collection names
    """
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        collections = client.collections.list_all()
        collection_names = collections
        
        client.close()
        return collection_names
        
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


def check_weaviate_connection():
    """
    Check if Weaviate is running and accessible.
    
    Returns:
        dict: Connection status
    """
    try:
        client = create_weaviate_rest_client(host="127.0.0.1", port=8081)
        
        is_ready = client.is_ready()
        client.close()
        
        return {"status": "connected", "ready": is_ready}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


# -------------------------------
# Run as standalone script
# -------------------------------
if __name__ == "__main__":
    # Test connection
    print("Testing Weaviate connection...")
    conn_status = check_weaviate_connection()
    print(f"Connection status: {conn_status}")
    
    if conn_status.get('status') == 'connected':
        test_chat_id = "test-chat-123"
        test_texts = [
            "This is a test document about machine learning.",
            "Another piece of text to embed locally about artificial intelligence.",
            "Python is a great programming language for data science."
        ]
        
        print(f"\nEmbedding test texts for chat: {test_chat_id}...")
        result = embed_to_weaviate(test_texts, chat_id=test_chat_id)
        print(f"Result: {result}")
        
        print(f"\nQuerying for 'machine learning' in chat {test_chat_id}...")
        results = query_weaviate("machine learning", chat_id=test_chat_id, limit=3)
        for i, res in enumerate(results):
            print(f"{i+1}. {res['content']} (distance: {res['distance']})")
        
        print("\nListing all collections...")
        collections = list_collections()
        print(f"Collections: {collections}")
        
        # Cleanup test
        print(f"\nCleaning up test collection...")
        delete_chat_collection(test_chat_id)
