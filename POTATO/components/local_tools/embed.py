#  use nomic-embed-text:latest model from ollama for local embeddings, save vectors in a local vector db like pgvector or weaviate or similar. use docker/podman
# POTATO/components/local_tools/embed.py
# weaviate container running locally at http://127.0.0.1:8081

import weaviate
from weaviate.classes.init import Auth
from ollama import Client as Ollama

# -------------------------------
# Configuration
# -------------------------------
WEAVIATE_URL = "http://127.0.0.1:8081"
CLASS_NAME = "Document"


def embed_to_weaviate(texts):
    """
    Embed a list of texts using Ollama Nomic embedding model
    and upload vectors to a local Weaviate instance.

    Args:
        texts (str or list of str): Single text string or list of texts.
    """
    if isinstance(texts, str):
        texts = [texts]

    # 1. Connect to Weaviate using v4 API
    try:
        client = weaviate.connect_to_local(
            host="127.0.0.1",
            port=8081
        )
        
        # Check if collection exists, create if not
        collections = client.collections.list_all()
        
        if CLASS_NAME not in [c.name for c in collections]:
            # Create collection with manual vectorization
            client.collections.create(
                name=CLASS_NAME,
                vectorizer_config=None,  # Manual vectorization
                properties=[
                    {
                        "name": "content",
                        "dataType": ["text"]
                    }
                ]
            )
            print(f"Created collection: {CLASS_NAME}")
        
        # Get the collection
        collection = client.collections.get(CLASS_NAME)
        
        # 3. Initialize Ollama client
        ollama = Ollama()
        
        # 4. Embed texts and upload
        for text in texts:
            # Get embedding from Ollama
            response = ollama.embeddings(
                model="nomic-embed-text:latest",
                prompt=text
            )
            
            embedding = response['embedding']
            
            # Insert into Weaviate with manual vector
            collection.data.insert(
                properties={
                    "content": text
                },
                vector=embedding
            )
            
            print(f"Uploaded: {text[:50]}...")
        
        print(f"\n{len(texts)} texts uploaded to Weaviate successfully.")
        client.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def query_weaviate(query_text, limit=5):
    """
    Query Weaviate for similar documents using vector search.
    
    Args:
        query_text (str): Query text to search for
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of matching documents with their content and scores
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host="127.0.0.1",
            port=8081
        )
        
        # Get collection
        collection = client.collections.get(CLASS_NAME)
        
        # Get query embedding from Ollama
        ollama = Ollama()
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=query_text
        )
        
        query_vector = response['embedding']
        
        # Perform vector search
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=True
        )
        
        # Format results
        formatted_results = []
        for obj in results.objects:
            formatted_results.append({
                'content': obj.properties.get('content', ''),
                'distance': obj.metadata.distance if hasattr(obj.metadata, 'distance') else None
            })
        
        client.close()
        return formatted_results
        
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        import traceback
        traceback.print_exc()
        return []


# -------------------------------
# Run as standalone script
# -------------------------------
if __name__ == "__main__":
    test_texts = [
        "This is a test document about machine learning.",
        "Another piece of text to embed locally about artificial intelligence.",
        "Python is a great programming language for data science."
    ]
    print("Embedding test texts...")
    embed_to_weaviate(test_texts)
    
    print("\nQuerying for 'machine learning'...")
    results = query_weaviate("machine learning", limit=3)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']} (distance: {result['distance']})")
