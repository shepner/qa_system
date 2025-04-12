"""
Vector database management for document embeddings
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
import json
import uuid

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
        
        # Get embedding dimension from EMBEDDING_MODEL config
        embedding_config = config.get("EMBEDDING_MODEL", {})
        self.embedding_dim = embedding_config.get("DIMENSIONS", 3072)
        
        self._store = None
        self._initialized = False
        
        logger.info(f"Initializing vector store type {self.db_type} at {self.db_path} with embedding dimension {self.embedding_dim}")
        
    async def initialize(self) -> None:
        """Initialize the vector store.
        
        Raises:
            ValueError: If store type is not supported
        """
        if self._initialized:
            return
            
        if self.db_type != "chroma":
            raise ValueError(f"Unsupported vector store type: {self.db_type}")
            
        try:
            # ChromaDB client initialization is synchronous
            self._store = chromadb.PersistentClient(path=self.db_path)
            
            # Collection creation/getting is also synchronous
            self._collection = self._store.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_metric,
                    "dimension": self.embedding_dim
                }
            )
            
            self._initialized = True
            logger.info(f"Initialized vector store at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _validate_metadata(self, metadata: Dict) -> Dict:
        """Validate and transform document metadata.
        
        Args:
            metadata: Document metadata to validate
            
        Returns:
            Validated and transformed metadata
        """
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
            
        required_fields = ["path", "file_type"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
                
        # Ensure path is resolved and converted to string
        metadata["path"] = str(Path(metadata["path"]).resolve())
        
        # Set defaults for optional fields
        metadata.setdefault("classification", "internal")
        metadata.setdefault("created_at", "")
        metadata.setdefault("last_modified", "")
        
        # Ensure doc_id exists for document-level identification
        if "doc_id" not in metadata:
            metadata["doc_id"] = str(uuid.uuid4())
            
        return metadata

    async def store_embeddings(self, embeddings: List[List[float]], chunks: List[str], doc_id: str, metadata: Optional[Dict] = None) -> None:
        """Store document embeddings in the vector store.
        
        Args:
            embeddings: List of embeddings to store
            chunks: List of text chunks corresponding to embeddings
            doc_id: Document ID to associate with embeddings
            metadata: Optional metadata to store with embeddings
            
        Raises:
            RuntimeError: If vector store is not initialized
            ValueError: If embeddings or chunks are empty or lengths don't match
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
        if not embeddings or not chunks:
            raise ValueError("Embeddings and chunks cannot be empty")
            
        if len(embeddings) != len(chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of chunks ({len(chunks)})")
            
        # Create chunk-level metadata
        chunk_metadata = []
        for i in range(len(chunks)):
            chunk_meta = {"doc_id": doc_id, "chunk_index": i}
            if metadata:
                chunk_meta.update(metadata)
            chunk_metadata.append(chunk_meta)
            
        # Store embeddings and chunks
        try:
            self._collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadata,
                ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
            )
            logger.debug(f"Stored {len(embeddings)} embeddings for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings for document {doc_id}: {e}")
            raise

    async def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar documents using query embedding.
        
        Args:
            query_embedding: Query embedding to search with
            k: Number of results to return (default: 5)
            
        Returns:
            List of dictionaries containing document chunks and metadata
            
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
                
            logger.debug(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    async def remove_document(self, doc_id: str) -> bool:
        """Remove all chunks for a document from the vector store.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if document was removed, False otherwise
            
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
        try:
            # Find all chunks for this document
            results = self._collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                logger.warning(f"No chunks found for document {doc_id}")
                return False
                
            # Delete all chunks
            self._collection.delete(
                where={"doc_id": doc_id}
            )
            
            logger.info(f"Removed {len(results['metadatas'])} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            raise

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store.
        
        Returns:
            List[Dict[str, Any]]: List of document metadata dictionaries containing:
                - id: Document ID
                - filename: Document filename
                - path: Full path to document
                - file_type: Type of document
                - chunk_count: Number of chunks for this document
                
        Raises:
            RuntimeError: If vector store is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
            
        try:
            results = self._collection.get(
                include=["metadatas"]
            )
            
            # Group chunks by document ID to get document-level metadata
            docs = {}
            for metadata in results["metadatas"]:
                if not metadata or "doc_id" not in metadata:
                    continue
                    
                doc_id = metadata["doc_id"]
                path = metadata.get("path", "")
                if doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "filename": Path(path).name if path else "",
                        "path": path,  # Include the full path
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunk_count": 1
                    }
                else:
                    docs[doc_id]["chunk_count"] += 1
                    
            logger.debug(f"Found {len(docs)} unique documents")
            return list(docs.values())
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if self._initialized and hasattr(self, '_store'):
                # First reset the collection reference
                self._store = None
                
                # Then close the client if it exists
                if hasattr(self, '_client') and self._client is not None:
                    try:
                        self._client.reset()  # Reset the client state
                    except Exception as e:
                        logger.warning(f"Error resetting ChromaDB client: {str(e)}")
                    self._client = None
                
                self._initialized = False
                logger.info("Vector store cleanup complete")
            else:
                logger.debug("Vector store already cleaned up or not initialized")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Don't re-raise the exception to allow other cleanup to continue
            # Just log it and continue

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
            results = self._store.get(where={"path": path})
            
            if not results["ids"]:
                return None
                
            # Get the first chunk's metadata since all chunks share document metadata
            metadata = results["metadatas"][0]
            return {
                "id": metadata.get("doc_id"),  # Use doc_id instead of chunk id
                "path": metadata.get("path"),
                "file_type": metadata.get("file_type"),
                "classification": metadata.get("classification", "internal"),
                "created_at": metadata.get("created_at", ""),
                "last_modified": metadata.get("last_modified", "")
            }
            
        except Exception as e:
            logger.error(f"Error getting document by path: {str(e)}")
            return None

    def get_index_size(self) -> int:
        """Get the total number of embeddings in the store."""
        if not self._initialized:
            return 0
            
        try:
            return len(self._store.get()["ids"])
        except Exception as e:
            logger.error(f"Error getting index size: {str(e)}")
            return 0