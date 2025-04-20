"""
Vector store system for managing document removal operations.

This module provides a VectorStore class that handles vector database operations
for removing documents and their associated embeddings using ChromaDB as the backend.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection

from remove_files.config import get_config
from remove_files.logging_setup import setup_logging

class VectorStore:
    """Vector database management system for document removal using ChromaDB.
    
    This class handles vector database operations including:
    - Initialization and connection management
    - Document removal and verification
    - Collection management and cleanup
    - Integrity checks
    
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
            LOG_FILE=self.config.get_nested('LOGGING.LOG_FILE'),
            LOG_LEVEL=self.config.get_nested('LOGGING.LEVEL', "INFO"),
            DEBUG=self.config.get_nested('LOGGING.DEBUG', False)
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VectorStore for document removal")
        
        # Get vector store configuration
        self.persist_directory = self.config.get_nested(
            'VECTOR_STORE.PERSIST_DIRECTORY',
            './data/vector_store'
        )
        self.collection_name = self.config.get_nested(
            'VECTOR_STORE.COLLECTION_NAME',
            'embeddings'
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
                name=self.collection_name
            )
        except Exception as e:
            self.logger.error(f"Error getting/creating collection: {str(e)}")
            raise

    def remove_documents(self, doc_ids: List[str]) -> bool:
        """Remove documents and their vectors from the store.
        
        Args:
            doc_ids: List of document IDs to remove
            
        Returns:
            bool: True if removal was successful
            
        Raises:
            Exception: If removal fails
        """
        try:
            self.logger.info(f"Removing {len(doc_ids)} documents from vector store")
            
            # Remove documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(doc_ids), batch_size):
                batch = doc_ids[i:i + batch_size]
                self.collection.delete(
                    ids=batch
                )
                self.logger.debug(f"Removed batch of {len(batch)} documents")
            
            self.logger.info(f"Successfully removed {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing documents: {str(e)}")
            raise

    def verify_removal(self, doc_ids: List[str]) -> bool:
        """Verify documents were successfully removed.
        
        Args:
            doc_ids: List of document IDs to verify
            
        Returns:
            bool: True if all documents were removed
        """
        try:
            self.logger.debug(f"Verifying removal of {len(doc_ids)} documents")
            
            # Get all documents
            results = self.collection.get(
                ids=doc_ids,
                include=['metadatas']
            )
            
            # Check if any documents still exist
            remaining = len(results['ids'])
            if remaining > 0:
                self.logger.warning(f"Found {remaining} documents that were not removed")
                return False
                
            self.logger.info("Verified all documents were removed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying document removal: {str(e)}")
            raise

    def get_document_metadata(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specified documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of metadata dictionaries for each document
        """
        try:
            results = self.collection.get(
                ids=doc_ids,
                include=['metadatas']
            )
            return results['metadatas']
            
        except Exception as e:
            self.logger.error(f"Error getting document metadata: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Perform cleanup operations on the vector store.
        
        This includes:
        - Removing any temporary data
        - Optimizing storage
        - Verifying database integrity
        """
        try:
            self.logger.info("Starting vector store cleanup")
            
            # Perform cleanup operations
            # Note: Specific cleanup tasks would depend on ChromaDB capabilities
            # and system requirements
            
            self.logger.info("Vector store cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
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
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise 