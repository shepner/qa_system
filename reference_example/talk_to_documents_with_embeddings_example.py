"""
@file: talk_to_documents_with_embeddings_example.py
Example: Talk to documents with Gemini embeddings (in-memory search)
Reference: https://github.com/google-gemini/cookbook/blob/main/examples/Talk_to_documents_with_embeddings.ipynb

This script demonstrates:
- Creating document embeddings with Gemini
- Storing documents and embeddings in memory
- Performing similarity search over documents
- Printing top results for user queries
"""

import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List, Tuple

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_texts(client, texts: List[str], task_type: str) -> List[np.ndarray]:
    """Embed a list of texts using Gemini embeddings."""
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        emb = np.array(result.embeddings[0].values, dtype=np.float32)
        embeddings.append(emb)
    return embeddings

def main():
    """Main function to demonstrate document search with Gemini embeddings."""
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast, dark-colored fox leaps above a sleepy canine.",
        "Quantum computing is the future of technology.",
        "Dogs are domesticated mammals, not natural wild animals."
    ]
    metadatas = [
        {"category": "animal", "id": 1},
        {"category": "animal", "id": 2},
        {"category": "technology", "id": 3},
        {"category": "animal", "id": 4}
    ]

    print("Embedding documents...")
    doc_embeddings = embed_texts(client, documents, task_type="RETRIEVAL_DOCUMENT")

    # Example queries
    queries = [
        "What animal jumps over a dog?",
        "Tell me about quantum technology.",
        "Describe a domesticated animal."
    ]

    print("\nDocument search results:")
    for query in queries:
        print(f"\nQuery: {query}")
        query_emb = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")[0]
        # Compute similarities
        sims = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
        # Get top 2 results
        top_indices = np.argsort(sims)[::-1][:2]
        for idx in top_indices:
            print(f"  - Score: {sims[idx]:.4f}, Doc: {documents[idx]}, Metadata: {metadatas[idx]}")

if __name__ == "__main__":
    main() 