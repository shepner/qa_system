"""Remove flow module for managing document deletion.

This module handles the safe deletion of files and their associated data from the system,
ensuring proper cleanup of vectors, metadata, and related information while maintaining
system consistency.
"""

import logging
from typing import Dict, Any, List, Optional
from qa_system.config import get_config
from qa_system.vector_system import VectorStore
from qa_system.exceptions import QASystemError, handle_exception

class RemoveFlow:
    """Handles the process of removing documents from the system."""
    
    def __init__(self, config_path: str):
        """Initialize the remove flow.
        
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
        
    def remove_documents(self, patterns: List[str]) -> Dict[str, Any]:
        """Remove documents matching the provided patterns.
        
        Args:
            patterns: List of glob patterns to match documents
            
        Returns:
            Dictionary containing:
            - removed: List of successfully removed documents
            - failed: List of documents that failed to remove
            - not_found: List of patterns with no matches
            - errors: List of error details
            
        Raises:
            QASystemError: If removal process fails
        """
        try:
            results = {
                'removed': [],
                'failed': [],
                'not_found': [],
                'errors': []
            }
            
            # Find matching documents
            for pattern in patterns:
                try:
                    matches = self.vector_store.find_documents(pattern)
                    
                    if not matches:
                        results['not_found'].append(pattern)
                        continue
                    
                    # Remove each matched document
                    for doc in matches:
                        try:
                            # Remove from vector store
                            self.vector_store.remove_document(doc['id'])
                            
                            # Verify removal
                            if self._verify_removal(doc['id']):
                                results['removed'].append(doc['path'])
                            else:
                                results['failed'].append(doc['path'])
                                
                        except Exception as e:
                            error_details = handle_exception(
                                e,
                                f"Failed to remove document: {doc['path']}",
                                reraise=False
                            )
                            results['failed'].append(doc['path'])
                            results['errors'].append(error_details)
                            
                except Exception as e:
                    error_details = handle_exception(
                        e,
                        f"Failed to process pattern: {pattern}",
                        reraise=False
                    )
                    results['errors'].append(error_details)
                    
            return results
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Failed to remove documents",
                reraise=False
            )
            raise QASystemError(
                f"Document removal failed: {error_details['message']}"
            ) from e
            
    def _verify_removal(self, document_id: str) -> bool:
        """Verify that a document was completely removed.
        
        Args:
            document_id: ID of document to verify
            
        Returns:
            True if document is completely removed, False otherwise
        """
        try:
            # Check if document still exists in vector store
            return not self.vector_store.document_exists(document_id)
        except Exception as e:
            self.logger.error(f"Error verifying document removal: {e}")
            return False 