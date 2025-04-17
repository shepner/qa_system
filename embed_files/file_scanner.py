import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Type
import hashlib
import logging
import fnmatch
from embed_files.config import get_config
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
    """Scanner for discovering and processing files."""

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

    def __init__(self, config):
        """Initialize FileScanner with configuration."""
        self.config = config
        # Use get_nested to get the FILE_SCANNER section from config
        scanner_config = self.config.get_nested('FILE_SCANNER', {})
        self.allowed_extensions = scanner_config.get('ALLOWED_EXTENSIONS', [])
        self.exclude_patterns = scanner_config.get('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = scanner_config.get('HASH_ALGORITHM', 'sha256')
        self.document_path = scanner_config.get('DOCUMENT_PATH', './docs')
        self.processed_files: Set[str] = set()
        
        # Get configuration settings with defaults
        self.chunk_size = scanner_config.get('chunk_size', 1024 * 1024)  # Default 1MB
        
        # Initialize processor map with defaults and any custom mappings from config
        self.processor_map = dict(self.DEFAULT_PROCESSOR_MAP)
        custom_processor_map = scanner_config.get('processor_map', {})
        self.processor_map.update(custom_processor_map)
        
        logger.debug(f"FileScanner initialized with config: allowed_extensions={self.allowed_extensions}, "
                    f"exclude_patterns={self.exclude_patterns}, hash_algorithm={self.hash_algorithm}, "
                    f"document_path={self.document_path}, chunk_size={self.chunk_size}")

    def is_already_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed."""
        try:
            logger.debug(f"Checking if file is already processed: {file_path}")
            checksum = self.calculate_checksum(file_path)
            is_processed = checksum in self.processed_files
            logger.debug(f"File {file_path} processed status: {is_processed} (checksum: {checksum})")
            return is_processed
        except Exception as e:
            logger.error(f"Error checking if file is processed {file_path}: {str(e)}")
            return False

    def mark_as_processed(self, checksum: str) -> None:
        """Mark a file as processed using its checksum."""
        logger.debug(f"Marking file as processed with checksum: {checksum}")
        self.processed_files.add(checksum)
        logger.debug(f"Total processed files count: {len(self.processed_files)}")

    def should_process_file(self, file_path: str) -> Tuple[bool, str]:
        """Determine if a file should be processed based on configuration.
        
        Returns:
            Tuple[bool, str]: A tuple containing (should_process, reason)
            where reason explains why the file was skipped if should_process is False
        """
        try:
            logger.debug(f"Evaluating whether to process file: {file_path}")
            
            # Check if already processed
            if self.is_already_processed(file_path):
                reason = "already processed"
                logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                return False, reason
                
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

    def get_file_metadata(self, file_path: str | Path, base_path: str | Path) -> Dict[str, Any]:
        """Get comprehensive metadata for a file."""
        try:
            logger.debug(f"Getting metadata for file: {file_path}")
            path_obj = Path(file_path).resolve()
            base_path_obj = Path(base_path).resolve()
            
            logger.debug(f"Resolved paths - file: {path_obj}, base: {base_path_obj}")
            
            # Get file stats
            stats = path_obj.stat()
            logger.debug(f"File stats retrieved for {path_obj}")
            
            # Calculate relative path
            try:
                relative_path = str(path_obj.relative_to(base_path_obj))
                logger.debug(f"Calculated relative path: {relative_path}")
            except ValueError:
                relative_path = str(path_obj)
                logger.debug(f"Using absolute path as relative path: {relative_path}")
            
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
            logger.debug(f"Basic metadata collected for {path_obj}: {metadata}")
            
            # Try to get additional metadata from type-specific processors
            try:
                logger.debug(f"Attempting to get processor metadata for {path_obj}")
                processor_metadata = self._get_processor_metadata(path_obj)
                if processor_metadata:
                    metadata["processor_metadata"] = processor_metadata
                    logger.debug(f"Processor metadata added for {path_obj}: {processor_metadata}")
                else:
                    logger.debug(f"No processor metadata available for {path_obj}")
            except Exception as e:
                logger.warning(f"Failed to get processor metadata for {path_obj}: {str(e)}")
                metadata["processor_metadata"] = {}
            
            logger.debug(f"Complete metadata collection finished for {path_obj}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            raise

    def _get_processor_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get additional metadata from type-specific processors.
        
        Args:
            file_path: Path object representing the file to process
            
        Returns:
            Optional[Dict[str, Any]]: Additional metadata from the processor, or None if no processor is found
        """
        try:
            # Get file extension without the dot and convert to lowercase
            file_ext = file_path.suffix.lstrip('.').lower()
            logger.debug(f"Getting processor for file extension: {file_ext}")
            
            # Get the appropriate processor class
            processor_class = self.processor_map.get(file_ext)
            if not processor_class:
                logger.debug(f"No processor found for extension: {file_ext}")
                return None
                
            logger.debug(f"Using processor class {processor_class.__name__} for {file_path}")
            
            # Initialize the processor
            processor = processor_class()
            
            # Get basic metadata to pass to the processor
            basic_metadata = {
                "path": str(file_path),
                "file_type": file_ext,
                "filename": file_path.name,
            }
            
            # Process the document
            logger.debug(f"Processing document with {processor_class.__name__}: {file_path}")
            processed_metadata = processor.process(str(file_path), basic_metadata)
            
            logger.debug(f"Successfully processed {file_path} with {processor_class.__name__}")
            return processed_metadata
            
        except Exception as e:
            logger.error(f"Error processing file {file_path} with document processor: {str(e)}")
            return None

    def get_document_processor_type(self, file_type: str) -> str:
        """Determine the appropriate document processor type based on file type."""
        processor_type = self.processor_map.get(file_type.lower(), 'unknown')
        logger.debug(f"Determined processor type for {file_type}: {processor_type}")
        return processor_type

    def scan_files(self, directory: str) -> List[Dict[str, Any]]:
        """Scan directory for files and collect metadata."""
        logger.debug(f"Starting file scan in directory: {directory}")
        
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return []

        base_path = Path(directory).resolve()
        logger.debug(f"Resolved base path: {base_path}")
        file_info_list = []
        skipped_files = []  # Track skipped files and reasons

        try:
            logger.debug(f"Beginning directory walk: {directory}")
            for root, dirs, files in os.walk(directory):
                logger.debug(f"Scanning directory: {root} (contains {len(files)} files, {len(dirs)} subdirectories)")
                for filename in files:
                    file_path = os.path.join(root, filename)
                    logger.debug(f"Processing file: {file_path}")
                    
                    try:
                        should_process, reason = self.should_process_file(file_path)
                        if not should_process:
                            skipped_files.append({"path": file_path, "reason": reason})
                            continue
                            
                        logger.debug(f"Getting metadata for file: {file_path}")
                        file_info = self.get_file_metadata(file_path, str(base_path))
                        
                        if file_info['checksum'] in self.processed_files:
                            reason = "duplicate file (checksum already processed)"
                            logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                            skipped_files.append({"path": file_path, "reason": reason})
                            continue
                            
                        logger.debug(f"Adding file to processed list: {file_path}")
                        self.processed_files.add(file_info['checksum'])
                        file_info_list.append(file_info)
                        logger.debug(f"Successfully processed file: {file_path}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error processing file {file_path}: {error_msg}")
                        skipped_files.append({"path": file_path, "reason": f"error during processing: {error_msg}"})
                        continue

            # Log summary of skipped files
            if skipped_files:
                logger.debug("Summary of skipped files:")
                for skip_info in skipped_files:
                    logger.debug(f"Skipped {skip_info['path']}: {skip_info['reason']}")

            logger.debug(f"File scan completed. Processed {len(file_info_list)} files, "
                        f"skipped {len(skipped_files)} files in {directory}")
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {str(e)}")
            
        return file_info_list 