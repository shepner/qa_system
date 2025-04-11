"""
Vector database management for document embeddings
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages storage and retrieval of document embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store.
        
        Args:
            config: Configuration dictionary containing vector store settings
        """
        vector_store_config = config.get("VECTOR_STORE", {})
        self.db_path = vector_store_config.get("PERSIST_DIRECTORY", "./data/vector_store")
        self.db_type = vector_store_config.get("TYPE", "chroma").lower()
        self.collection_name = vector_store_config.get("COLLECTION_NAME", "qa_documents")
        self.distance_metric = vector_store_config.get("DISTANCE_METRIC", "cosine")
        self.top_k = vector_store_config.get("TOP_K", 40)
        self.embedding_dim = vector_store_config.get("EMBEDDING_DIM", 768)
        self._store = None
        self._initialized = False
        
        logger.info(f"Initializing vector store type {self.db_type} at {self.db_path}")
        
    async def initialize(self):
        """Initialize the vector store."""
        if self.db_type != "chroma":
            raise ValueError(f"Unsupported vector store type: {self.db_type}")
            
        try:
            client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self._store = client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "distance_metric": self.distance_metric,
                    "embedding_dim": self.embedding_dim
                }
            )
            
            self._initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    async def store_embeddings(self, embeddings: List[Dict], doc_metadata: Dict) -> None:
        """Store document embeddings in the vector store.
        
        Args:
            embeddings: List of embeddings with metadata. Each dict must contain:
                - embedding: List[float] - The embedding vector
                - doc_id: str - Unique document identifier
                - chunk_index: int - Index of the chunk within the document
                - text: str - The text content for this chunk
            doc_metadata: Document metadata containing:
                - path: str - Path to the document
                - file_type: str - Type of document
                - classification: str - Document classification
                - created_at: str - Creation timestamp
                - last_modified: str - Last modification timestamp
                - processing_status: str - Processing status
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
            
        try:
            # Validate inputs
            if not embeddings:
                raise ValueError("No embeddings provided")
                
            if not doc_metadata or not doc_metadata.get("path"):
                raise ValueError("Invalid document metadata")
                
            path = str(Path(doc_metadata["path"]).resolve())
            
            # Validate and prepare embeddings
            vectors = []
            metadatas = []
            ids = []
            texts = []
            
            for e in embeddings:
                # Validate embedding format
                if not isinstance(e.get("embedding"), list):
                    raise ValueError(f"Invalid embedding format for chunk {e.get('chunk_index')}")
                    
                embedding = np.array(e["embedding"])
                if embedding.shape[0] != self.embedding_dim:
                    raise ValueError(
                        f"Invalid embedding dimension {embedding.shape[0]}, "
                        f"expected {self.embedding_dim}"
                    )
                
                # Required fields
                if not all(k in e for k in ["doc_id", "chunk_index", "text"]):
                    raise ValueError("Missing required fields in embedding metadata")
                
                vectors.append(embedding)
                metadatas.append({
                    "doc_id": e["doc_id"],
                    "chunk_index": e["chunk_index"],
                    "filename": path,
                    "file_type": doc_metadata["file_type"],
                    "classification": doc_metadata.get("classification", "internal"),
                    "created_at": doc_metadata.get("created_at"),
                    "last_modified": doc_metadata.get("last_modified"),
                    "processing_status": doc_metadata.get("processing_status", "success"),
                    "chunk_size": len(e["text"])
                })
                ids.append(f"{e['doc_id']}_{e['chunk_index']}")
                texts.append(e["text"])
            
            # Remove any existing embeddings for this document
            await self.remove_document(embeddings[0]["doc_id"])
            
            # Store new embeddings
            self._store.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                f"Stored {len(embeddings)} embeddings for document {path} "
                f"with status {doc_metadata.get('processing_status', 'success')}"
            )
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise

    async def similarity_search(
        self, 
        query_embedding: List[float], 
        k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (defaults to config TOP_K)
            filters: Optional metadata filters for search
            
        Returns:
            List of similar documents with metadata and distance scores
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
            
        try:
            results = self._store.query(
                query_embeddings=[query_embedding],
                n_results=k or self.top_k,
                where=filters,
                include=["metadatas", "documents", "distances"]
            )
            
            return [{
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - (results["distances"][0][i] / 2.0)  # Normalize to 0-1 score
            } for i in range(len(results["ids"][0]))]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self._initialized and self._store is not None:
            try:
                # ChromaDB PersistentClient handles persistence automatically
                self._store = None
                self._initialized = False
                logger.info("Vector store cleanup complete")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                raise

    def get_document_by_path(self, file_path: str) -> Optional[Dict]:
        """Find a document in the store by its file path.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
            
        try:
            path = str(Path(file_path).resolve())
            results = self._store.get(where={"filename": path})
            
            if not results["ids"]:
                return None
                
            return {
                "id": results["ids"][0],
                "filename": results["metadatas"][0]["filename"],
                "file_type": results["metadatas"][0]["file_type"],
                "classification": results["metadatas"][0].get("classification"),
                "created_at": results["metadatas"][0].get("created_at"),
                "last_modified": results["metadatas"][0].get("last_modified")
            }
            
        except Exception as e:
            logger.error(f"Error getting document by path: {str(e)}")
            return None

    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
            
        try:
            results = self._store.get(where={"doc_id": doc_id})
            if results["ids"]:
                self._store.delete(ids=results["ids"])
                logger.info(f"Removed document {doc_id} ({len(results['ids'])} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {doc_id}: {str(e)}")
            return False

    def get_index_size(self) -> int:
        """Get the total number of embeddings in the store."""
        if not self._initialized:
            return 0
            
        try:
            return len(self._store.get()["ids"])
        except Exception as e:
            logger.error(f"Error getting index size: {str(e)}")
            return 0