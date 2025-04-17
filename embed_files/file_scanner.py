import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import hashlib
import logging
import fnmatch
from embed_files.config import get_config

logger = logging.getLogger(__name__)

class FileScanner:
    """Scanner for discovering and processing files."""

    def __init__(self, config):
        """Initialize FileScanner with configuration."""
        self.config = config
        # Use get_nested as specified in architecture instead of get
        scanner_config = self.config.get_nested('FILE_SCANNER', {})
        self.allowed_extensions = scanner_config.get_nested('ALLOWED_EXTENSIONS', [])
        self.exclude_patterns = scanner_config.get_nested('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = scanner_config.get_nested('HASH_ALGORITHM', 'sha256')
        self.document_path = scanner_config.get_nested('DOCUMENT_PATH', './docs')
        self.processed_files: Set[str] = set()
        
        # Get configuration settings with defaults
        self.chunk_size = scanner_config.get_nested('chunk_size', 1024 * 1024)  # Default 1MB
        self.processor_map = scanner_config.get_nested('processor_map', {})

    def is_already_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            bool: True if the file has already been processed.
        """
        try:
            checksum = self.calculate_checksum(file_path)
            return checksum in self.processed_files
        except Exception as e:
            logger.error(f"Error checking if file is processed {file_path}: {str(e)}")
            return False

    def mark_as_processed(self, checksum: str) -> None:
        """Mark a file as processed using its checksum.
        
        Args:
            checksum: File's checksum to mark as processed.
        """
        self.processed_files.add(checksum)

    def should_process_file(self, file_path: str) -> bool:
        """Determine if a file should be processed based on configuration.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            bool: True if the file should be processed, False otherwise.
        """
        try:
            # Check if already processed
            if self.is_already_processed(file_path):
                logger.debug(f"Skipping already processed file: {file_path}")
                return False
                
            filename = os.path.basename(file_path)
            
            # Check allowed extensions
            if '*' not in self.allowed_extensions:
                file_ext = os.path.splitext(filename)[1].lstrip('.')
                if not any(fnmatch.fnmatch(file_ext, ext.lstrip('*').lstrip('.')) for ext in self.allowed_extensions):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {str(e)}")
            return False

    def calculate_checksum(self, file_path: str) -> str:
        """Calculate checksum for a file using configured hash algorithm.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            str: Hexadecimal checksum of the file.
        """
        try:
            hash_func = getattr(hashlib, self.hash_algorithm)()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {str(e)}")
            raise

    def get_file_metadata(self, file_path: str | Path, base_path: str | Path) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a file.

        Args:
            file_path (str | Path): Path to the file
            base_path (str | Path): Base path for calculating relative paths

        Returns:
            Dict[str, Any]: Dictionary containing file metadata including:
                - path: Absolute path to the file
                - relative_path: Path relative to base_path
                - directory: Directory containing the file
                - filename_full: Full filename with extension
                - filename_stem: Filename without extension
                - file_type: File extension
                - created_at: File creation timestamp
                - last_modified: Last modification timestamp
                - file_size: Size in bytes
                - checksum: SHA256 hash of file contents
                - processor_metadata: Additional metadata from type-specific processors
        """
        try:
            path_obj = Path(file_path).resolve()
            base_path_obj = Path(base_path).resolve()
            
            # Get file stats
            stats = path_obj.stat()
            
            # Calculate relative path
            try:
                relative_path = str(path_obj.relative_to(base_path_obj))
            except ValueError:
                # Fallback if path is not relative to base_path
                relative_path = str(path_obj)
            
            metadata = {
                "path": str(path_obj),
                "relative_path": relative_path,
                "directory": str(path_obj.parent),
                "filename_full": path_obj.name,
                "filename_stem": path_obj.stem,
                "file_type": path_obj.suffix.lstrip(".").lower(),
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "file_size": stats.st_size,
                "checksum": self.calculate_checksum(path_obj)
            }
            
            # Try to get additional metadata from type-specific processors
            try:
                processor_metadata = self._get_processor_metadata(path_obj)
                if processor_metadata:
                    metadata["processor_metadata"] = processor_metadata
            except Exception as e:
                logger.warning(f"Failed to get processor metadata for {path_obj}: {str(e)}")
                metadata["processor_metadata"] = {}
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            raise

    def _get_processor_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get additional metadata from type-specific processors.
        This is a placeholder method that should be implemented based on specific file type requirements.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            Optional[Dict[str, Any]]: Additional metadata from type-specific processor, or None if no processor exists
        """
        # TODO: Implement type-specific processors
        return None

    def get_document_processor_type(self, file_type: str) -> str:
        """Determine the appropriate document processor type based on file type.
        
        Args:
            file_type: File extension/type.
            
        Returns:
            str: Document processor type identifier.
        """
        return self.processor_map.get(file_type.lower(), 'unknown')

    def scan_files(self, directory: str) -> List[Dict[str, Any]]:
        """Scan directory for files and collect metadata.
        
        Args:
            directory: Base directory to scan for files.
            
        Returns:
            List of file metadata dictionaries.
        """
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return []

        base_path = Path(directory).resolve()
        file_info_list = []

        try:
            for root, _, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    try:
                        if not self.should_process_file(file_path):
                            logger.debug(f"Skipping file: {file_path}")
                            continue
                            
                        file_info = self.get_file_metadata(file_path, str(base_path))
                        
                        if file_info['checksum'] in self.processed_files:
                            logger.debug(f"Skipping duplicate file: {file_path}")
                            continue
                            
                        self.processed_files.add(file_info['checksum'])
                        file_info_list.append(file_info)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {str(e)}")
            
        return file_info_list 