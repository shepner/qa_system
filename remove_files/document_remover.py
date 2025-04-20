"""
Document removal system for the QA system.

This module handles the removal of documents from the vector store,
including validation and cleanup of associated data.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from remove_files.config import get_config
from remove_files.file_matcher import FileMatcher
from remove_files.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentRemover:
    """Handles removal of documents from the vector store.
    
    This class orchestrates the document removal process including:
    - Validation of removal requests
    - Safe removal of vectors and metadata
    - Verification of removal success
    - Cleanup of associated data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the document remover with configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize vector store
        self.vector_store = VectorStore(config_path)
        
        # Initialize file matcher with vector store for validation
        self.file_matcher = FileMatcher(self.vector_store)
        
        # Get removal validation settings
        self.require_confirmation = self.config.get_nested(
            'REMOVAL_VALIDATION.REQUIRE_CONFIRMATION',
            default=True
        )
        
        logger.info("Initialized DocumentRemover")
        
    def remove_documents(self, file_paths: List[str], force: bool = False) -> Dict[str, Any]:
        """Remove documents from the vector store.
        
        Args:
            file_paths: List of file paths to remove
            force: If True, skip confirmation even if configured
            
        Returns:
            Dictionary containing:
            - success: Boolean indicating if all operations succeeded
            - removed: List of successfully removed files
            - failed: List of files that failed to remove
            - errors: Dictionary mapping failed files to error messages
            
        Raises:
            ValueError: If no valid files are provided
        """
        if not file_paths:
            raise ValueError("No file paths provided for removal")
            
        logger.info(f"Starting removal of {len(file_paths)} documents")
        
        # Track results
        results = {
            'success': False,
            'removed': [],
            'failed': [],
            'errors': {}
        }
        
        try:
            # Validate files exist and are in vector store
            valid_files = []
            for file_path in file_paths:
                if self.file_matcher._validate_file(file_path):
                    valid_files.append(file_path)
                else:
                    results['failed'].append(file_path)
                    results['errors'][file_path] = "File not found or not in vector store"
            
            if not valid_files:
                logger.warning("No valid files found for removal")
                return results
                
            logger.debug(f"Found {len(valid_files)} valid files for removal")
            
            # Get confirmation if required
            if self.require_confirmation and not force:
                logger.info("Removal requires confirmation - skipping actual removal")
                results['success'] = True
                return results
            
            # Remove documents from vector store
            try:
                # Get document IDs from vector store
                doc_ids = self._get_document_ids(valid_files)
                
                if not doc_ids:
                    logger.error("No document IDs found for removal")
                    return results
                
                # Remove vectors and metadata
                self.vector_store.collection.delete(
                    ids=doc_ids
                )
                
                # Verify removal
                removed_files = self._verify_removal(valid_files)
                results['removed'].extend(removed_files)
                
                # Track any files that failed verification
                failed_files = set(valid_files) - set(removed_files)
                results['failed'].extend(failed_files)
                for file in failed_files:
                    results['errors'][file] = "Removal verification failed"
                    
                results['success'] = len(removed_files) == len(valid_files)
                
            except Exception as e:
                logger.error(f"Error removing documents from vector store: {str(e)}")
                results['failed'].extend(valid_files)
                for file in valid_files:
                    results['errors'][file] = f"Vector store error: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error during document removal process: {str(e)}")
            results['failed'].extend(file_paths)
            for file in file_paths:
                results['errors'][file] = f"Process error: {str(e)}"
            
        logger.info(
            f"Document removal completed - "
            f"Removed: {len(results['removed'])}, "
            f"Failed: {len(results['failed'])}"
        )
        return results
        
    def _get_document_ids(self, file_paths: List[str]) -> List[str]:
        """Get vector store document IDs for the given file paths.
        
        Args:
            file_paths: List of file paths to get IDs for
            
        Returns:
            List of document IDs
        """
        try:
            # Query vector store for documents matching file paths
            doc_ids = []
            for file_path in file_paths:
                # Search metadata for matching file path
                results = self.vector_store.collection.get(
                    where={"path": str(file_path)}
                )
                if results and results['ids']:
                    doc_ids.extend(results['ids'])
                    
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error getting document IDs: {str(e)}")
            return []
            
    def _verify_removal(self, file_paths: List[str]) -> List[str]:
        """Verify documents were successfully removed from vector store.
        
        Args:
            file_paths: List of file paths to verify
            
        Returns:
            List of successfully removed file paths
        """
        removed_files = []
        
        for file_path in file_paths:
            try:
                # Check if any documents still exist for this path
                results = self.vector_store.collection.get(
                    where={"path": str(file_path)}
                )
                
                if not results or not results['ids']:
                    removed_files.append(file_path)
                else:
                    logger.warning(f"Document still exists in vector store: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error verifying removal of {file_path}: {str(e)}")
                
        return removed_files 