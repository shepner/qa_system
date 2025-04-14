"""
Document processor for handling, validating, and preparing documents for embedding.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import hashlib
from datetime import datetime, timedelta
import magic
import re
import os


class DocumentProcessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize document processor with configuration."""
        self.config = config["DOCUMENT_PROCESSING"]
        self.mime = magic.Magic(mime=True)

    def is_valid_file(self, file_path: Path) -> bool:
        """
        Validate if a file should be processed based on configured rules.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if file is valid for processing, False otherwise
        """
        if not file_path.is_file():
            return False

        # Check file extension
        if file_path.suffix not in self.config["ALLOWED_EXTENSIONS"]:
            return False

        # Check include patterns (these override exclude patterns)
        included = False
        for pattern in self.config["INCLUDE_PATTERNS"]:
            if re.match(pattern, file_path.name):
                included = True
                break

        if not included:
            # Check exclude patterns
            for pattern in self.config["EXCLUDE_PATTERNS"]:
                if re.match(pattern, file_path.name):
                    return False

        # Check file size
        file_size = file_path.stat().st_size
        size_limits = self.config["FILE_SIZE_LIMITS"]
        if file_size < size_limits["MIN_BYTES"] or file_size > size_limits["MAX_BYTES"]:
            return False

        # Check file age
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        now = datetime.now()
        
        max_age = timedelta(days=self.config["DOCUMENT_AGE"]["MAX_DAYS"])
        min_age = timedelta(minutes=self.config["DOCUMENT_AGE"]["MIN_MINUTES"])
        
        if now - mtime > max_age or now - mtime < min_age:
            return False

        return True

    def get_document_hash(self, content: str) -> str:
        """Generate document hash using configured algorithm."""
        algo = self.config["UPDATE_HANDLING"]["HASH_ALGORITHM"].lower()
        hasher = hashlib.new(algo)
        hasher.update(content.encode())
        return hasher.hexdigest()

    def process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single document and return its metadata and content.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Dict containing document metadata and content, or None if invalid
        """
        if not self.is_valid_file(file_path):
            return None

        try:
            content = file_path.read_text()
            
            # Validate content length
            if len(content) < self.config["CONTENT_VALIDATION"]["MIN_CHARS"]:
                return None
            if len(content) > self.config["CONTENT_VALIDATION"]["MAX_CHARS"]:
                return None

            # Get MIME type
            mime_type = self.mime.from_file(str(file_path))
            
            # Skip binary files if configured
            if self.config["CONTENT_VALIDATION"]["SKIP_BINARY"] and not mime_type.startswith("text/"):
                return None

            # Calculate file hash for change detection
            file_hash = hashlib.sha256(content.encode()).hexdigest()

            return {
                "path": str(file_path),
                "content": content,
                "hash": file_hash,
                "mime_type": mime_type,
                "size": file_path.stat().st_size,
                "mtime": datetime.fromtimestamp(file_path.stat().st_mtime)
            }

        except Exception as e:
            # Log error and skip file
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_directory(self, dir_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Process all valid documents in a directory.
        
        Args:
            dir_path: Path to the directory to process
            
        Yields:
            Dict containing document metadata and content for each valid file
        """
        if not dir_path.is_dir():
            raise ValueError(f"Path {dir_path} is not a directory")

        for file_path in dir_path.rglob("*"):
            if result := self.process_document(file_path):
                yield result 