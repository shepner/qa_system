"""Document storage and retrieval functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import fnmatch
import os
import json

class DocumentStore:
    """Handles storage and retrieval of documents."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the document store.
        
        Args:
            config: Configuration dictionary that may contain:
                - 'VECTOR_STORE.PERSIST_DIRECTORY': Directory for file storage
        """
        # File storage setup
        vector_store_config = config.get('VECTOR_STORE', {})
        data_dir = vector_store_config.get('PERSIST_DIRECTORY', './data/vector_store')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.data_dir / 'metadata'
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Get exclude patterns from config
        doc_config = config.get("DOCUMENT_PROCESSING", {})
        self.exclude_patterns = doc_config.get("EXCLUDE_PATTERNS", [
            "*.git/*",
            "*__pycache__/*",
            "*.pyc",
            "*.env",
            "*.DS_Store"
        ])
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DocumentStore at {self.data_dir}")

    def should_exclude(self, filepath: str) -> bool:
        """Check if a file should be excluded based on configured patterns.
        
        Args:
            filepath: Path to the file to check
            
        Returns:
            bool: True if the file should be excluded, False otherwise
        """
        try:
            # Get relative path if absolute path provided
            rel_path = os.path.relpath(filepath) if os.path.isabs(filepath) else filepath
            
            # Check if file matches any exclude pattern
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatch(rel_path, pattern):
                    self.logger.debug(f"Excluding file {rel_path} matching pattern {pattern}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking file exclusion for {filepath}: {str(e)}")
            # If there's an error checking, exclude the file to be safe
            return True

    async def store_document(self, content: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a document in the document store.
        
        Args:
            content: Document content to store
            doc_id: Unique identifier for the document
            metadata: Optional metadata to store with the document
            
        Returns:
            bool: True if storage was successful
        """
        try:
            # Store document content
            doc_path = self.data_dir / f"{doc_id}.txt"
            doc_path.write_text(content)
            
            # Store metadata separately
            if metadata is not None:
                metadata['doc_id'] = doc_id
                metadata_path = self.metadata_dir / f"{doc_id}.json"
                metadata_path.write_text(json.dumps(metadata))
            
            self.logger.debug(f"Stored document {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store document {doc_id}: {str(e)}")
            return False

    async def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document from the store.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Optional[str]: Document content if found, None otherwise
        """
        try:
            doc_path = self.data_dir / f"{doc_id}.txt"
            if doc_path.exists():
                content = doc_path.read_text()
                self.logger.debug(f"Retrieved document {doc_id}")
                return content
            self.logger.warning(f"Document {doc_id} not found")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve document {doc_id}: {str(e)}")
            return None

    async def list_documents(self) -> List[str]:
        """List all document IDs in the store.
        
        Returns:
            List[str]: List of document IDs
        """
        try:
            return [p.stem for p in self.data_dir.glob("*.txt")]
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            doc_path = self.data_dir / f"{doc_id}.txt"
            metadata_path = self.metadata_dir / f"{doc_id}.json"
            
            success = True
            if doc_path.exists():
                doc_path.unlink()
            else:
                success = False
                
            if metadata_path.exists():
                metadata_path.unlink()
                
            if success:
                self.logger.debug(f"Deleted document {doc_id}")
            else:
                self.logger.warning(f"Document {doc_id} not found for deletion")
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Optional[Dict[str, Any]]: Document metadata if found, None otherwise
        """
        try:
            metadata_path = self.metadata_dir / f"{doc_id}.json"
            if metadata_path.exists():
                return json.loads(metadata_path.read_text())
            return None
        except Exception as e:
            self.logger.error(f"Failed to get metadata for document {doc_id}: {str(e)}")
            return None

    async def cleanup(self) -> None:
        """Clean up resources and ensure all file handles are closed."""
        try:
            # Nothing to actively clean up for file-based storage
            # Just log completion for consistency with other components
            self.logger.info("Document store cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during document store cleanup: {str(e)}")
            # Don't re-raise the exception to allow other cleanup to continue
            # Just log it and continue 