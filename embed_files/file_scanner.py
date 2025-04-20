import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Type
import hashlib
import logging
import fnmatch
from embed_files.config import get_config
from embed_files.vector_system import VectorStore
from embed_files.document_processors import (
    BaseDocumentProcessor,
    TextDocumentProcessor,
    MarkdownDocumentProcessor,
    CSVDocumentProcessor,
    PDFDocumentProcessor,
    ImageDocumentProcessor
)

logger = logging.getLogger(__name__)

class FileScanner:
    """Scanner for discovering and processing files.
    
    Handles file discovery, validation, and metadata collection according to configuration.
    Integrates with VectorDB to determine which files need processing.
    """

    # Default mapping of file extensions to processor classes
    DEFAULT_PROCESSOR_MAP = {
        'txt': TextDocumentProcessor,
        'md': MarkdownDocumentProcessor,
        'csv': CSVDocumentProcessor,
        'pdf': PDFDocumentProcessor,
        # Image formats
        'jpg': ImageDocumentProcessor,
        'jpeg': ImageDocumentProcessor,
        'png': ImageDocumentProcessor,
        'gif': ImageDocumentProcessor,
        'bmp': ImageDocumentProcessor,
        'webp': ImageDocumentProcessor,
    }

    def __init__(self, config, vector_store: Optional[VectorStore] = None):
        """Initialize FileScanner with configuration and vector store.
        
        Args:
            config: Configuration object
            vector_store: Optional VectorStore instance for checking existing files
        """
        self.config = config
        self.vector_store = vector_store
        
        # Use get_nested to get the FILE_SCANNER section from config
        scanner_config = self.config.get_nested('FILE_SCANNER', {})
        self.allowed_extensions = scanner_config.get('ALLOWED_EXTENSIONS', ['*'])
        self.exclude_patterns = scanner_config.get('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = scanner_config.get('HASH_ALGORITHM', 'sha256')
        self.document_path = scanner_config.get('DOCUMENT_PATH', './docs')
        
        # Get configuration settings with defaults
        self.chunk_size = scanner_config.get('chunk_size', 1024 * 1024)  # Default 1MB
        
        # Initialize processor map with defaults and any custom mappings from config
        self.processor_map = dict(self.DEFAULT_PROCESSOR_MAP)
        custom_processor_map = scanner_config.get('processor_map', {})
        self.processor_map.update(custom_processor_map)
        
        logger.debug(f"FileScanner initialized with config: allowed_extensions={self.allowed_extensions}, "
                    f"exclude_patterns={self.exclude_patterns}, hash_algorithm={self.hash_algorithm}, "
                    f"document_path={self.document_path}, chunk_size={self.chunk_size}")

    def should_process_file(self, file_path: str) -> Tuple[bool, str]:
        """Determine if a file should be processed based on configuration.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Tuple[bool, str]: A tuple containing (should_process, reason)
            where reason explains why the file was skipped if should_process is False
        """
        try:
            logger.debug(f"Evaluating whether to process file: {file_path}")
            
            filename = os.path.basename(file_path)
            logger.debug(f"Checking extensions for file: {filename}")
            
            # Check allowed extensions
            if '*' not in self.allowed_extensions:
                file_ext = os.path.splitext(filename)[1].lstrip('.')
                logger.debug(f"Checking if extension '{file_ext}' matches allowed extensions: {self.allowed_extensions}")
                if not any(fnmatch.fnmatch(file_ext, ext.lstrip('*').lstrip('.')) for ext in self.allowed_extensions):
                    reason = f"extension '{file_ext}' not in allowed extensions {self.allowed_extensions}"
                    logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                    return False, reason
            
            # Check exclude patterns
            for pattern in self.exclude_patterns:
                # Handle explicit inclusion with ! prefix
                if pattern.startswith('!'):
                    if fnmatch.fnmatch(filename, pattern[1:]):
                        logger.debug(f"File {file_path} explicitly included by pattern {pattern}")
                        return True, "explicitly included"
                elif fnmatch.fnmatch(filename, pattern):
                    reason = f"matches exclude pattern '{pattern}'"
                    logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                    return False, reason
            
            logger.debug(f"File {file_path} approved for processing")
            return True, "approved for processing"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error checking file {file_path}: {error_msg}")
            return False, f"error during check: {error_msg}"

    def calculate_checksum(self, file_path: str) -> str:
        """Calculate checksum for a file using configured hash algorithm."""
        try:
            logger.debug(f"Calculating {self.hash_algorithm} checksum for: {file_path}")
            hash_func = getattr(hashlib, self.hash_algorithm)()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            checksum = hash_func.hexdigest()
            logger.debug(f"Checksum calculated for {file_path}: {checksum}")
            return checksum
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {str(e)}")
            raise

    def check_existing_file(self, file_path: str, checksum: str) -> bool:
        """Check if a file exists in the vector store based on its checksum.
        
        Args:
            file_path: Path to the file
            checksum: File's checksum
            
        Returns:
            bool: True if file needs processing (not in vector store), False otherwise
        """
        if not self.vector_store:
            logger.debug("No vector store provided - all files will be marked for processing")
            return True
            
        try:
            # Query vector store for matching checksum
            results = self.vector_store.collection.get(
                where={"checksum": checksum}
            )
            
            needs_processing = not bool(results and results['ids'])
            logger.debug(f"File {file_path} needs processing: {needs_processing}")
            return needs_processing
            
        except Exception as e:
            logger.error(f"Error checking file in vector store {file_path}: {str(e)}")
            return True

    def get_file_metadata(self, file_path: str | Path, base_path: str | Path) -> Dict[str, Any]:
        """Get comprehensive metadata for a file.
        
        Args:
            file_path: Path to the file
            base_path: Base path for calculating relative paths
            
        Returns:
            Dict containing file metadata including path, checksum, and needs_processing flag
        """
        try:
            logger.debug(f"Getting metadata for file: {file_path}")
            path_obj = Path(file_path).resolve()
            base_path_obj = Path(base_path).resolve()
            
            # Calculate relative path
            try:
                relative_path = str(path_obj.relative_to(base_path_obj))
            except ValueError:
                relative_path = str(path_obj)
            
            # Calculate checksum
            checksum = self.calculate_checksum(path_obj)
            
            # Check if file needs processing
            needs_processing = self.check_existing_file(str(path_obj), checksum)
            
            metadata = {
                "path": relative_path,
                "checksum": checksum,
                "needs_processing": needs_processing
            }
            
            logger.debug(f"Metadata collected for {path_obj}: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            raise

    def scan_files(self, path: str) -> List[Dict[str, Any]]:
        """Scan directory or individual file and collect metadata.
        
        Args:
            path: Path to scan (can be a directory or individual file)
            
        Returns:
            List of dictionaries containing file metadata including:
            - path: Relative path of the file
            - checksum: File's cryptographic hash
            - needs_processing: Boolean indicating if file needs processing
        """
        logger.debug(f"Starting file scan for path: {path}")
        
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return []

        file_info_list = []
        skipped_files = []
        base_path = Path(os.path.dirname(path) if os.path.isfile(path) else path).resolve()

        try:
            # Handle individual file
            if os.path.isfile(path):
                logger.debug(f"Processing individual file: {path}")
                try:
                    should_process, reason = self.should_process_file(path)
                    if should_process:
                        file_info = self.get_file_metadata(path, str(base_path))
                        file_info_list.append(file_info)
                        logger.debug(f"Successfully processed file: {path}")
                    else:
                        skipped_files.append({"path": path, "reason": reason})
                        logger.debug(f"Skipping file {path}: {reason}")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing file {path}: {error_msg}")
                    skipped_files.append({"path": path, "reason": f"error during processing: {error_msg}"})
            
            # Handle directory
            else:
                logger.debug(f"Beginning directory walk: {path}")
                for root, dirs, files in os.walk(path):
                    logger.debug(f"Scanning directory: {root}")
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        logger.debug(f"Processing file: {file_path}")
                        
                        try:
                            should_process, reason = self.should_process_file(file_path)
                            if should_process:
                                file_info = self.get_file_metadata(file_path, str(base_path))
                                file_info_list.append(file_info)
                                logger.debug(f"Successfully processed file: {file_path}")
                            else:
                                skipped_files.append({"path": file_path, "reason": reason})
                                logger.debug(f"Skipping file {file_path}: {reason}")
                                
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error processing file {file_path}: {error_msg}")
                            skipped_files.append({"path": file_path, "reason": f"error during processing: {error_msg}"})
                            continue

            # Log summary
            logger.info(f"File scan completed. Found {len(file_info_list)} files to process, "
                       f"skipped {len(skipped_files)} files")
            
            if skipped_files:
                logger.debug("Summary of skipped files:")
                for skip_info in skipped_files:
                    logger.debug(f"Skipped {skip_info['path']}: {skip_info['reason']}")
            
            return file_info_list
            
        except Exception as e:
            logger.error(f"Error scanning path {path}: {str(e)}")
            return []

class FileProcessingError(Exception):
    """Base exception for file processing errors."""
    pass

class FileAccessError(FileProcessingError):
    """Raised when file cannot be accessed."""
    pass

class FileValidationError(FileProcessingError):
    """Raised when file fails validation."""
    pass

class ProcessingError(FileProcessingError):
    """Raised when file processing fails."""
    pass

def process_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
    """Process a single file with appropriate error handling."""
    try:
        # Validate file access
        if not os.path.exists(file_path):
            raise FileAccessError(f"File does not exist: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise FileAccessError(f"File not readable: {file_path}")
            
        # Get file stats for context
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        
        logger.info(f"Processing file: {file_path} (size: {file_size} bytes)")
        
        # Validate file
        is_valid, validation_error = self._validate_file(file_path)
        if not is_valid:
            raise FileValidationError(f"File validation failed: {validation_error}")
            
        # Process file
        success, result = self._process_file_content(file_path)
        if not success:
            raise ProcessingError(f"File processing failed: {result}")
            
        return True, None
        
    except FileAccessError as e:
        logger.error(f"File access error for {file_path}: {str(e)}")
        return False, f"access error: {str(e)}"
    except FileValidationError as e:
        logger.error(f"File validation error for {file_path}: {str(e)}")
        return False, f"validation error: {str(e)}"
    except ProcessingError as e:
        logger.error(f"Processing error for {file_path}: {str(e)}")
        return False, f"processing error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {str(e)}", exc_info=True)
        return False, f"unexpected error: {str(e)}" 