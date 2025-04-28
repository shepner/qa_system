"""List flow module for retrieving document information.

This module handles listing documents and their metadata from the vector database,
with optional pattern filtering and statistics generation.
"""

import logging
from typing import Dict, Any, Optional, List
from qa_system.config import get_config
from qa_system.vector_system import VectorStore
from qa_system.exceptions import QASystemError, handle_exception

class ListFlow:
    """Handles listing documents from the vector store."""
    
    def __init__(self, config_path: str):
        """Initialize the list flow.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigError: If configuration is invalid
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize vector store
        self.vector_store = VectorStore(config_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def list_documents(self, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
        """List documents in the vector store.
        
        Args:
            filter_pattern: Optional pattern to filter results
            
        Returns:
            Dictionary containing:
            - documents: List of document metadata
            - stats: Collection statistics
            
        Raises:
            QASystemError: If listing fails
        """
        try:
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            # Get and filter documents
            documents = self.vector_store.list_documents(filter_pattern)
            
            return {
                'documents': documents,
                'stats': stats
            }
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to list documents",
                reraise=False
            )
            raise QASystemError(
                f"Failed to list documents: {error_details['message']}"
            ) from e 