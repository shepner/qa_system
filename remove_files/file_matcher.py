from pathlib import Path
import logging
from typing import List, Optional
import fnmatch
import os

from remove_files.config import get_config

logger = logging.getLogger(__name__)

class FileMatcher:
    """Identifies and validates files for removal based on input paths or patterns."""
    
    def __init__(self, vector_store):
        """Initialize the FileMatcher with configuration settings.
        
        Args:
            vector_store: Vector store instance for validation
        """
        self.config = get_config()
        self.case_sensitive = self.config.get_nested('FILE_MATCHER.CASE_SENSITIVE', default=False)
        self.vector_store = vector_store
        logger.debug(f"Initialized FileMatcher with case_sensitive={self.case_sensitive}")

    def find_matching_files(self, pattern: str) -> List[str]:
        """Find all files in the vector store matching the given pattern.
        
        Args:
            pattern: File path or pattern to match against
            
        Returns:
            List of matched file paths that exist in vector store
        """
        logger.info(f"Finding files matching pattern: {pattern}")
        
        try:
            # Get all documents from vector store
            all_docs = self.vector_store.get_all_documents()
            
            # If pattern is an exact path, just check if it exists
            if not self._is_pattern(pattern):
                logger.debug(f"Pattern is an exact file path: {pattern}")
                return [pattern] if pattern in all_docs else []

            # If pattern contains directory components, get the base pattern
            base_pattern = os.path.basename(pattern)
            dir_prefix = os.path.dirname(pattern)
            
            # Filter documents based on pattern
            matches = []
            for doc in all_docs:
                # If dir_prefix is specified, check if document is in that directory
                if dir_prefix and not doc.startswith(dir_prefix):
                    continue
                    
                # Check if base name matches pattern
                doc_base = os.path.basename(doc)
                if self._matches_pattern(doc_base, base_pattern):
                    matches.append(doc)
            
            logger.info(f"Found {len(matches)} files matching pattern: {pattern}")
            return matches

        except Exception as e:
            logger.error(f"Error finding matching files for pattern {pattern}: {str(e)}")
            raise

    def _is_pattern(self, path: str) -> bool:
        """Check if a path contains glob pattern characters.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path contains pattern characters
        """
        return any(c in path for c in '*?[]!')

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if a filename matches the given pattern.
        
        Args:
            filename: Name of the file to check
            pattern: Pattern to match against
            
        Returns:
            True if the filename matches the pattern
        """
        if not self.case_sensitive:
            filename = filename.lower()
            pattern = pattern.lower()
        
        return fnmatch.fnmatch(filename, pattern) 