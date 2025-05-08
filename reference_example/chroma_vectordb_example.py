"""
@file: chroma_vectordb_example.py
Example: Using ChromaDB as a vector database with Gemini embeddings
Reference: https://github.com/google-gemini/cookbook/blob/main/examples/chromadb/Vectordb_with_chroma.ipynb

This script demonstrates:
- Creating a Chroma collection
- Adding documents with Gemini-generated embeddings and metadata
- Demonstrating persistence and reloading
- Performing similarity search with multiple queries
- Printing document IDs, metadata, and results
- Deleting/resetting a collection
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings
import shutil
import sys

# Add at the top after imports
logfile_path = "chroma_vectordb_example.log"
def log(msg):
    with open(logfile_path, "a") as f:
        f.write(str(msg) + "\n")

def global_exception_handler(exctype, value, traceback):
    log(f"[FATAL ERROR] {value}")
sys.excepthook = global_exception_handler

log(f"Python executable: {sys.executable}")

# Minimal debug file test
try:
    with open("test_debug.txt", "w") as f:
        f.write("Debug file test\n")
    log("[DEBUG] Wrote test_debug.txt")
except Exception as e:
    log(f"[FATAL ERROR] Could not write test_debug.txt: {e}")
    sys.exit(1)

# --- Setup ---
load_dotenv()
log("[DEBUG] Loaded .env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
log(f"[DEBUG] GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)}")
if not GEMINI_API_KEY:
    log("[FATAL ERROR] GEMINI_API_KEY is missing")
    sys.exit(1)
log("[DEBUG] About to initialize Gemini client")

# Initialize Gemini client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    log("[DEBUG] Gemini client initialized")
except Exception as e:
    log(f"[FATAL ERROR] Gemini client init failed: {e}")
    sys.exit(1)
log("[DEBUG] After Gemini client init")

# ChromaDB persistence directory
persist_dir = "./chroma_db"
collection_name = "gemini_example"

# --- Optional: Clean up/reset collection (delete directory) ---
if os.path.exists(persist_dir):
    log(f"Removing existing ChromaDB directory: {persist_dir}")
    shutil.rmtree(persist_dir)
    log(f"[DEBUG] Removed {persist_dir}")
else:
    log(f"[DEBUG] {persist_dir} does not exist at start")
log("[DEBUG] After cleanup/reset")

# --- Example documents and metadata ---
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
ids = [f"doc_{i}" for i in range(len(documents))]
log("[DEBUG] Prepared documents, metadatas, and ids")

# --- Generate embeddings for each document using Gemini ---
embeddings = []
try:
    for i, doc in enumerate(documents):
        log(f"[DEBUG] Generating embedding for doc {i}")
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=doc,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        # Extract the embedding vector as a list of floats
        # result.embeddings is a list of ContentEmbedding, so get .values from the first
        embedding_values = result.embeddings[0].values
        embeddings.append(embedding_values)  # Now we're appending just the values
        log(f"[DEBUG] Got embedding for doc {i} with {len(embedding_values)} dimensions")
    log("[DEBUG] Generated all embeddings")
except Exception as e:
    log(f"[FATAL ERROR] Embedding generation failed: {e}")
    sys.exit(1)
log("[DEBUG] After embedding generation")

# --- Initialize ChromaDB persistent client and collection (with persistence) ---
try:
    log("[DEBUG] Initializing ChromaDB PersistentClient")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    log("[DEBUG] ChromaDB PersistentClient initialized")
    
    # Check if collection exists and delete it if it does
    try:
        existing_collection = chroma_client.get_collection(name=collection_name)
        chroma_client.delete_collection(name=collection_name)
        log("[DEBUG] Deleted existing collection")
    except Exception:
        pass  # Collection doesn't exist, which is fine
        
    collection = chroma_client.create_collection(name=collection_name)
    log("[DEBUG] ChromaDB collection created")
except Exception as e:
    log(f"[FATAL ERROR] ChromaDB client/collection failed: {e}")
    sys.exit(1)
log("[DEBUG] After ChromaDB client/collection")

# --- Add documents, embeddings, and metadata ---
try:
    log("[DEBUG] Adding documents to collection")
    log(f"[DEBUG] Number of documents: {len(documents)}")
    log(f"[DEBUG] Number of embeddings: {len(embeddings)}")
    log(f"[DEBUG] Number of metadatas: {len(metadatas)}")
    log(f"[DEBUG] Number of ids: {len(ids)}")
    log(f"[DEBUG] First embedding dimensions: {len(embeddings[0])}")
    
    collection.add(
        documents=documents,
        embeddings=embeddings,  # Now contains proper list of float lists
        metadatas=metadatas,
        ids=ids
    )
    log("[DEBUG] Added documents to collection")
except Exception as e:
    log(f"[FATAL ERROR] ChromaDB add failed: {e}")
    sys.exit(1)
log("[DEBUG] After add to collection")

# --- Remove persist() call, persistence is automatic ---
log(f"Collection '{collection_name}' persisted to {persist_dir}\n")
log("[DEBUG] After (automatic) persist")

# Debug: Check if directory exists and list contents (limit to 10 lines)
try:
    debug_file = "chroma_db_debug.txt"
    lines_written = 0
    with open(debug_file, "w") as f:
        if os.path.exists(persist_dir):
            f.write(f"[DEBUG] Directory '{persist_dir}' exists.\n")
            for root, dirs, files in os.walk(persist_dir):
                if lines_written >= 10:
                    f.write("[DEBUG] ...output truncated...\n")
                    break
                f.write(f"\n[DEBUG] {root}:\n"); lines_written += 1
                for name in dirs:
                    if lines_written >= 10:
                        f.write("[DEBUG] ...output truncated...\n")
                        break
                    f.write(f"  [DIR]  {name}\n"); lines_written += 1
                for name in files:
                    if lines_written >= 10:
                        f.write("[DEBUG] ...output truncated...\n")
                        break
                    f.write(f"  [FILE] {name}\n"); lines_written += 1
        else:
            f.write(f"[DEBUG] Directory '{persist_dir}' does NOT exist after persist().\n")
    log(f"[DEBUG] Directory listing written to {debug_file}")
except Exception as e:
    log(f"[FATAL ERROR] Could not write {debug_file}: {e}")
    sys.exit(1)
log("[DEBUG] After directory debug")

# --- Demonstrate reloading the collection ---
log("Reloading ChromaDB client and collection from disk...")
chroma_client_reloaded = chromadb.PersistentClient(path=persist_dir)
collection_reloaded = chroma_client_reloaded.get_collection(collection_name)

# --- Multiple queries ---
queries = [
    "What animal jumps over a dog?",
    "Tell me about quantum technology.",
    "Describe a domesticated animal."
]

for query in queries:
    embedding_result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = embedding_result.embeddings[0].values  # Extract the list of floats
    results = collection_reloaded.query(
        query_embeddings=[query_embedding],  # This is now a list of floats
        n_results=2,
        include=["documents", "distances", "metadatas"]
    )
    log(f"\nQuery: {query}")
    log("Top results:")
    for doc, score, meta, doc_id in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
        results["ids"][0]
    ):
        log(f"  - ID: {doc_id}, Distance: {score:.4f}, Metadata: {meta}, Document: {doc}")

# --- Optional: Clean up/reset collection at the end ---
# Uncomment the following lines to delete the persisted collection after running
# if os.path.exists(persist_dir):
#     log(f"\nCleaning up: removing {persist_dir}")
#     shutil.rmtree(persist_dir) 