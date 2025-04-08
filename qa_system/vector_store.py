"""
Vector database management for document embeddings
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path

# Get logger for this module
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages storage and retrieval of document embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Use new config structure
        vector_store_config = config.get("VECTOR_STORE", {})
        self.db_path = str(vector_store_config.get("PERSIST_DIRECTORY") or config.get("VECTOR_DB_PATH", "./data/vectordb"))
        self.db_type = vector_store_config.get("TYPE") or config.get("VECTOR_DB_TYPE", "chroma")
        
        logger.info(f"Initializing vector store type {self.db_type} at {self.db_path}")
        
        if self.db_type.lower() == "chroma":
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Create or get the collection with configured name
            collection_name = vector_store_config.get("COLLECTION_NAME", "documents")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for QA system"}
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.db_type}")
        
        logger.info("Vector store initialized successfully")
        
    def get_document_by_path(self, file_path: str) -> Optional[Dict]:
        """Find a document in the store by its file path.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata if found, None otherwise
        """
        # Normalize input path
        normalized_path = str(Path(file_path).resolve())
        
        # Get all metadata
        all_metadata = self.collection.get()["metadatas"]
        
        # Find document with matching path
        for metadata in all_metadata:
            # Compare normalized paths
            stored_path = metadata.get("filename")
            if stored_path and str(Path(stored_path).resolve()) == normalized_path:
                return {
                    "id": metadata["doc_id"],
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "processing_status": metadata.get("processing_status", "unknown")
                }
        return None
    
    async def store_embeddings(self, embeddings: List[Dict], doc_metadata: Dict) -> None:
        """Store document embeddings in the vector store.
        
        Args:
            embeddings: List of embeddings with metadata
            doc_metadata: Document metadata
        """
        # Normalize the path before storing
        normalized_path = str(Path(doc_metadata["path"]).resolve())
        
        # Convert embeddings to format expected by Chroma
        vectors = [np.array(e["embedding"]) for e in embeddings]
        metadatas = [{
            "doc_id": e["doc_id"],
            "chunk_index": e["chunk_index"],
            "filename": normalized_path,  # Store normalized path
            "file_type": doc_metadata["file_type"],
            "processing_status": "success"
        } for e in embeddings]
        ids = [f"{e['doc_id']}_{e['chunk_index']}" for e in embeddings]
        
        # Store in collection
        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )
        
    async def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
            
        return matches
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents in the store.
        
        Returns:
            List of document metadata
        """
        # Get all metadata
        all_metadata = self.collection.get()["metadatas"]
        
        # Group by doc_id
        docs = {}
        for metadata in all_metadata:
            doc_id = metadata["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "id": doc_id,
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "chunk_count": 0
                }
            docs[doc_id]["chunk_count"] += 1
            
        return list(docs.values())
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful
        """
        try:
            # Find all chunks for this document
            logger.info(f"Finding chunks for document {doc_id}")
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if not results["ids"]:
                logger.warning(f"No chunks found for document {doc_id}")
                return True  # Document doesn't exist, so technically it's removed
                
            # Log what we're about to remove
            chunk_count = len(results["ids"])
            logger.info(f"Removing {chunk_count} chunks for document {doc_id}")
            
            # Remove all chunks - ChromaDB may show "Add of existing embedding ID" warnings
            # but these are internal to ChromaDB and don't affect the deletion
            self.collection.delete(
                ids=results["ids"]
            )
            
            logger.info(f"Successfully removed document {doc_id} ({chunk_count} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {doc_id}: {str(e)}", exc_info=True)
            return False 