import logging
from typing import List
import fnmatch
import os

from remove_files.vector_store import VectorStore
from remove_files.config import get_config, Config
from remove_files.logging_setup import get_logger

class FileMatcher:
    """Matches files based on configured patterns."""

    def __init__(self, config: Config):
        """Initialize file matcher with configuration.
        
        Args:
            config: Configuration objectvgfxv vd
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get file matching settings from config
        self.file_patterns = self.config.get_nested('FILE_MATCHER.PATTERNS', [])
        self.exclude_patterns = self.config.get_nested('FILE_MATCHER.EXCLUDE', [])
        self.min_file_size = self.config.get_nested('FILE_MATCHER.MIN_SIZE_BYTES', 0)
        self.max_file_size = self.config.get_nested('FILE_MATCHER.MAX_SIZE_BYTES', float('inf'))
        
        if not self.file_patterns:
            self.logger.warning("No file patterns configured - will match all files")
            self.file_patterns = ["*"]

    def matches_patterns(self, file_path: str) -> bool:
        """Check if file path matches any include pattern and no exclude patterns.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            bool: True if file matches patterns
        """
        # Check if file matches any include pattern
        matches_include = any(fnmatch.fnmatch(file_path, pattern) for pattern in self.file_patterns)
        
        # Check if file matches any exclude pattern
        matches_exclude = any(fnmatch.fnmatch(file_path, pattern) for pattern in self.exclude_patterns)
        
        return matches_include and not matches_exclude

    def matches_size(self, file_size: int) -> bool:
        """Check if file size is within configured limits.
        
        Args:
            file_size: Size of file in bytes
            
        Returns:
            bool: True if file size is within limits
        """
        return self.min_file_size <= file_size <= self.max_file_size

    def matches(self, file_path: str) -> bool:
        """Check if file matches all criteria.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            bool: True if file matches all criteria
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                return False
                
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Check patterns and size
            return self.matches_patterns(file_path) and self.matches_size(file_size)
            
        except Exception as e:
            self.logger.error(f"Error checking file {file_path}: {str(e)}")
            return False

    def find_matching_files(self, base_path: str) -> List[str]:
        """Find all files under base path that match configured criteria.
        
        Args:
            base_path: Base directory path to search
            
        Returns:
            List[str]: List of matching file paths
        """
        matching_files = []
        
        try:
            # Walk directory tree
            for root, _, files in os.walk(base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if file matches criteria
                    if self.matches(file_path):
                        matching_files.append(file_path)
                        self.logger.debug(f"Found matching file: {file_path}")
            
            self.logger.info(f"Found {len(matching_files)} matching files under {base_path}")
            return matching_files
            
        except Exception as e:
            self.logger.error(f"Error finding files under {base_path}: {str(e)}")
            return []

    def validate_file(self, file_path: str) -> bool:
        """Validate that a file exists in the vector store.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            True if file exists in vector store, False otherwise
        """
        try:
            results = self.vector_store.collection.get(
                where={"path": str(file_path)}
            )
            return bool(results and results['ids'])
        except Exception as e:
            self.logger.error(f"Error validating file {file_path}: {str(e)}")
            return False
        
    def _matches_pattern(self, doc_id: str, pattern: str) -> bool:
        """Check if a document ID matches the given pattern.
        
        Args:
            doc_id: Document ID to check
            pattern: Pattern to match against
            
        Returns:
            True if document matches pattern, False otherwise
        """
        # Apply case sensitivity based on config
        if not self.case_sensitive:
            doc_id = doc_id.lower()
            pattern = pattern.lower()
            
        # For now, implement simple wildcard matching
        # TODO: Implement more sophisticated pattern matching based on config
        if pattern == '*':
            return True
        elif pattern.startswith('*') and pattern.endswith('*'):
            return pattern[1:-1] in doc_id
        elif pattern.startswith('*'):
            return doc_id.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return doc_id.startswith(pattern[:-1])
        return doc_id == pattern 