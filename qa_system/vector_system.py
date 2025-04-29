"""
Vector database system for storing and managing embeddings.

This module provides a VectorStore class that handles all vector database operations
using ChromaDB as the backend. It supports storing embeddings with metadata,
querying similar vectors, and managing the vector database lifecycle.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection
import time
from datetime import datetime

from qa_system.config import get_config
from qa_system.logging_setup import setup_logging
from .exceptions import (
    QASystemError,
    StorageError,
    VectorStoreError,
    ConfigurationError,
    handle_exception
)

logger = logging.getLogger(__name__)

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
            
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            # Load configuration
            self.config = get_config(config_path)
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing VectorStore", extra={
                'component': 'vector_store',
                'operation': 'initialization'
            })
            
            # Get vector store configuration with architecture-specified defaults
            vector_config = self.config.get_nested('VECTOR_STORE', {})
            self.persist_directory = vector_config.get('PERSIST_DIRECTORY', './data/vector_store')
            self.collection_name = vector_config.get('COLLECTION_NAME', 'qa_documents')
            self.distance_metric = vector_config.get('DISTANCE_METRIC', 'cosine')
            self.top_k = vector_config.get('TOP_K', 40)
            
            # Ensure persist directory exists
            persist_path = Path(self.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with architecture-aligned settings
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self._get_or_create_collection()
            
            self.logger.info(
                "VectorStore initialization complete",
                extra={
                    'component': 'vector_store',
                    'collection': self.collection_name,
                    'persist_directory': str(persist_path),
                    'distance_metric': self.distance_metric
                }
            )
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "VectorStore initialization failed",
                reraise=False
            )
            raise VectorStoreError(
                f"VectorStore initialization failed: {error_details['message']}"
            ) from e
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one if it doesn't exist."""
        try:
            start_time = time.time()
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_metric,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            duration = time.time() - start_time
            
            self.logger.debug(
                "Collection operation complete",
                extra={
                    'component': 'vector_store',
                    'operation': 'get_or_create_collection',
                    'collection': self.collection_name,
                    'duration_seconds': duration
                }
            )
            return collection
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error getting/creating collection",
                reraise=False
            )
            raise VectorStoreError(
                f"Failed to get/create collection: {error_details['message']}"
            ) from e
    
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
                 
        Raises:
            ValueError: If input validation fails
            VectorStoreError: If storage operation fails
        """
        try:
            if not embeddings or not metadata:
                raise ValueError("Embeddings and metadata cannot be empty")
            if len(embeddings) != len(metadata):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) must match "
                    f"number of metadata entries ({len(metadata)})"
                )
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{time.time()}" for i in range(len(embeddings))]
            elif len(ids) != len(embeddings):
                raise ValueError(
                    f"Number of IDs ({len(ids)}) must match "
                    f"number of embeddings ({len(embeddings)})"
                )
            
            # Add to collection
            start_time = time.time()
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            duration = time.time() - start_time
            
            self.logger.info(
                "Added embeddings to vector store",
                extra={
                    'component': 'vector_store',
                    'operation': 'add_embeddings',
                    'embedding_count': len(embeddings),
                    'duration_seconds': duration
                }
            )
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error adding embeddings",
                reraise=False
            )
            raise VectorStoreError(
                f"Failed to add embeddings: {error_details['message']}"
            ) from e
    
    def query_similar(
        self,
        query_embedding: List[float],
        n_results: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query for similar vectors in the database.
        
        Args:
            query_embedding: Vector to find similar embeddings for
            n_results: Number of results to return (default: uses TOP_K from config)
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            Dictionary containing:
            - ids: List of matching document IDs
            - distances: List of distances to query vector
            - metadatas: List of metadata for matching documents
            
        Raises:
            ValueError: If query embedding is empty
            VectorStoreError: If query operation fails
        """
        try:
            if not query_embedding:
                raise ValueError("Query embedding cannot be empty")
            
            n_results = n_results or self.top_k
            
            start_time = time.time()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter
            )
            duration = time.time() - start_time
            
            self.logger.info(
                "Query complete",
                extra={
                    'component': 'vector_store',
                    'operation': 'query_similar',
                    'n_results': n_results,
                    'duration_seconds': duration
                }
            )
            
            return {
                'ids': results['ids'][0],
                'distances': results['distances'][0],
                'metadatas': results['metadatas'][0]
            }
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error querying vector store",
                reraise=False
            )
            raise VectorStoreError(
                f"Failed to query similar vectors: {error_details['message']}"
            ) from e
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics
            
        Raises:
            RuntimeError: If stats collection fails
        """
        try:
            start_time = time.time()
            count = self.collection.count()
            
            stats = {
                'name': self.collection_name,
                'count': count,
                'distance_metric': self.distance_metric,
                'persist_directory': self.persist_directory,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            duration = time.time() - start_time
            self.logger.debug(
                "Collection stats retrieved",
                extra={
                    'component': 'vector_store',
                    'operation': 'get_stats',
                    'duration_seconds': duration,
                    'document_count': count
                }
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(
                f"Error getting collection stats: {str(e)}",
                extra={
                    'component': 'vector_store',
                    'operation': 'get_stats',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
    def reset_collection(self) -> None:
        """Reset the current collection, removing all data.
        
        Raises:
            RuntimeError: If reset operation fails
        """
        try:
            start_time = time.time()
            
            self.logger.warning(
                "Resetting collection",
                extra={
                    'component': 'vector_store',
                    'operation': 'reset',
                    'collection': self.collection_name
                }
            )
            
            self.client.reset()
            self.collection = self._get_or_create_collection()
            
            duration = time.time() - start_time
            self.logger.info(
                "Collection reset complete",
                extra={
                    'component': 'vector_store',
                    'operation': 'reset',
                    'duration_seconds': duration
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error resetting collection: {str(e)}",
                extra={
                    'component': 'vector_store',
                    'operation': 'reset',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise 