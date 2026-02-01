#  use nomic-embed-text:latest model from ollama for local embeddings, save vectors in a local vector db like pgvector or weaviate or similar. use docker/podman
# POTATO/components/local_tools/embed.py
# weaviate container running locally at http://127.0.0.1:8081

import weaviate
import ollama

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

    cparams = weaviate.ConnectionParams(url=WEAVIATE_URL)
    # 1. Connect to Weaviate
    client = weaviate.WeaviateClient(cparams)

    # 2. Create schema if not exists
    schema = {
        "classes": [
            {
                "class": CLASS_NAME,
                "vectorizer": "none",  # use our own embeddings
                "properties": [
                    {"name": "content", "dataType": ["text"]}
                ]
            }
        ]
    }

    existing_classes = [c["class"] for c in client.schema.get()["classes"]]
    if CLASS_NAME not in existing_classes:
        client.schema.create(schema)

    # 3. Initialize Ollama client
    ollama = Ollama()

    # 4. Embed texts and upload
    for text in texts:
        embedding = ollama.embed(model="nomic/embedding", text=text)
        vector = embedding["embedding"]
        client.data_object.create(
            data_object={"content": text},
            class_name=CLASS_NAME,
            vector=vector
        )

    print(f"{len(texts)} texts uploaded to Weaviate.")


# -------------------------------
# Run as standalone script
# -------------------------------
if __name__ == "__main__":
    test_texts = [
        "This is a test document.",
        "Another piece of text to embed locally."
    ]
    embed_to_weaviate(test_texts)
