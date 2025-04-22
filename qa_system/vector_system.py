"""
Vector database system for storing and managing embeddings.

This module provides a VectorStore class that handles all vector database operations
using ChromaDB as the backend. It supports storing embeddings with metadata,
querying similar vectors, and managing the vector database lifecycle.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection

from qa_system.config import get_config
from qa_system.logging_setup import setup_logging

class VectorStore:
    """Vector database management system using ChromaDB.
    
    This class handles all vector database operations including:
    - Initialization and connection management
    - Embedding storage and retrieval
    - Similarity search
    - Collection management
    
    Attributes:
        client: ChromaDB client instance
        collection: Active ChromaDB collection
        config: System configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the vector store with configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Setup logging
        setup_logging(
            log_file=self.config.get_nested('LOGGING.LOG_FILE'),
            log_level=self.config.get_nested('LOGGING.LEVEL', "INFO"),
            enable_debug=self.config.get_nested('LOGGING.DEBUG', False)
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VectorStore")
        
        # Get vector store configuration
        self.persist_directory = self.config.get_nested(
            'VECTOR_STORE.PERSIST_DIRECTORY',
            './data/vector_store'
        )
        self.collection_name = self.config.get_nested(
            'VECTOR_STORE.COLLECTION_NAME',
            'embeddings'
        )
        self.distance_metric = self.config.get_nested(
            'VECTOR_STORE.DISTANCE_METRIC',
            'cosine'
        )
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        self.logger.info(f"VectorStore initialized with collection: {self.collection_name}")
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one if it doesn't exist."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
        except Exception as e:
            self.logger.error(f"Error getting/creating collection: {str(e)}")
            raise
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add embeddings with metadata to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each embedding
            ids: Optional list of IDs for the embeddings. If not provided,
                 will be generated automatically.
        """
        try:
            # Validate input lengths match
            if len(embeddings) != len(metadata):
                raise ValueError("Number of embeddings must match number of metadata entries")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(embeddings))]
            elif len(ids) != len(embeddings):
                raise ValueError("Number of IDs must match number of embeddings")
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            
            self.logger.info(f"Added {len(embeddings)} embeddings to collection")
            
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {str(e)}")
            raise
    
    def query_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query for similar vectors in the database.
        
        Args:
            query_embedding: Vector to find similar embeddings for
            n_results: Number of results to return (default: 5)
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Dictionary containing:
            - ids: List of matching document IDs
            - distances: List of distances to query vector
            - metadatas: List of metadata for matching documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter
            )
            
            self.logger.debug(f"Query returned {len(results['ids'][0])} results")
            
            return {
                'ids': results['ids'][0],
                'distances': results['distances'][0],
                'metadatas': results['metadatas'][0]
            }
            
        except Exception as e:
            self.logger.error(f"Error querying similar vectors: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'distance_metric': self.distance_metric,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the current collection, removing all data."""
        try:
            self.logger.warning(f"Resetting collection: {self.collection_name}")
            self.client.reset()
            self.collection = self._get_or_create_collection()
            self.logger.info("Collection reset complete")
        except Exception as e:
            self.logger.error(f"Error resetting collection: {str(e)}")
            raise 