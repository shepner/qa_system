import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Type, Union
import hashlib
import logging
import fnmatch
from qa_system.config import get_config
from qa_system.vector_system import VectorStore
from qa_system.document_processors import (
    BaseDocumentProcessor,
    TextDocumentProcessor,
    MarkdownDocumentProcessor,
    CSVDocumentProcessor,
    PDFDocumentProcessor,
    ImageDocumentProcessor
)
from .exceptions import (
    QASystemError,
    FileError,
    FileAccessError,
    FileProcessingError,
    ValidationError,
    handle_exception
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with debug level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

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

    def __init__(self, config):
        """Initialize FileScanner with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Use get_nested to get the FILE_SCANNER section from config
        scanner_config = self.config.get_nested('FILE_SCANNER', {})
        self.allowed_extensions = scanner_config.get('ALLOWED_EXTENSIONS', ['*'])
        self.exclude_patterns = scanner_config.get('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = scanner_config.get('HASH_ALGORITHM', 'sha256')
        self.document_path = scanner_config.get('DOCUMENT_PATH', './docs')
        
        # Get VECTOR_STORE configuration settings
        vector_config = self.config.get_nested('VECTOR_STORE', {})
        self.vector_collection = vector_config.get('COLLECTION_NAME', 'documents')
        self.vector_dimensions = vector_config.get('DIMENSIONS', 1536)  # Default for many embedding models
        self.vector_distance = vector_config.get('DISTANCE_METRIC', 'cosine')
        
        # Get document processing configuration settings
        doc_processing_config = self.config.get_nested('DOCUMENT_PROCESSING', {})
        self.max_chunk_size = doc_processing_config.get('MAX_CHUNK_SIZE', 3072)  # Default 3072 characters
        
        # Initialize processor map with defaults and any custom mappings from config
        self.processor_map = dict(self.DEFAULT_PROCESSOR_MAP)
        custom_processor_map = scanner_config.get('processor_map', {})
        self.processor_map.update(custom_processor_map)
        
        # Initialize vector store from config
        self.vector_store = VectorStore(None)  # Use the global config instance
        
        self.processed_files: Set[str] = set()
        self.file_checksums: Dict[str, str] = {}
        
        logger.debug(f"FileScanner initialized with config: allowed_extensions={self.allowed_extensions}, "
                    f"exclude_patterns={self.exclude_patterns}, hash_algorithm={self.hash_algorithm}, "
                    f"document_path={self.document_path}, max_chunk_size={self.max_chunk_size}")

    def should_process_file(self, file_path: str) -> Tuple[bool, str, Optional[Type[BaseDocumentProcessor]]]:
        """Determine if a file should be processed based on configuration.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            Tuple[bool, str, Optional[Type[BaseDocumentProcessor]]]: A tuple containing 
            (should_process, reason, processor_class) where:
            - should_process: indicates if the file should be processed
            - reason: explains why the file was skipped if should_process is False
            - processor_class: the appropriate document processor class if should_process is True
        """
        try:
            logger.debug(f"Evaluating whether to process file: {file_path}")
            
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lstrip('.')
            logger.debug(f"Checking extensions for file: {filename}")
            
            # Check allowed extensions
            if '*' not in self.allowed_extensions:
                logger.debug(f"Checking if extension '{file_ext}' matches allowed extensions: {self.allowed_extensions}")
                if not any(fnmatch.fnmatch(file_ext, ext.lstrip('*').lstrip('.')) for ext in self.allowed_extensions):
                    reason = f"extension '{file_ext}' not in allowed extensions {self.allowed_extensions}"
                    logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                    return False, reason, None
            
            # Check exclude patterns
            for pattern in self.exclude_patterns:
                # Handle explicit inclusion with ! prefix
                if pattern.startswith('!'):
                    if fnmatch.fnmatch(filename, pattern[1:]):
                        processor_class = self.processor_map.get(file_ext.lower())
                        logger.debug(f"File {file_path} explicitly included by pattern {pattern}")
                        return True, "explicitly included", processor_class
                elif fnmatch.fnmatch(filename, pattern):
                    reason = f"matches exclude pattern '{pattern}'"
                    logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                    return False, reason, None
            
            # Get the appropriate processor class for the file extension
            processor_class = self.processor_map.get(file_ext.lower())
            if not processor_class:
                reason = f"no processor available for extension '{file_ext}'"
                logger.debug(f"Skipping file: {file_path} (Reason: {reason})")
                return False, reason, None
            
            logger.debug(f"File {file_path} approved for processing with processor {processor_class.__name__}")
            return True, "approved for processing", processor_class
        except Exception as e:
            error_details = handle_exception(e, f"Error checking file {file_path}", reraise=False)
            return False, f"error during check: {error_details['message']}", None

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
        
        This is a key architectural decision point that determines whether a file needs processing:
        - If the checksum exists in the vector store, the file is skipped (return False)
        - If the checksum doesn't exist, the file needs processing (return True)
        
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
            # Query vector store for exact checksum match
            results = self.vector_store.collection.get(
                where={"checksum": checksum}
            )
            
            # If no results found, file needs processing
            needs_processing = len(results['ids']) == 0
            
            if needs_processing:
                logger.debug(f"File {file_path} not found in vector store (checksum: {checksum})")
            else:
                logger.debug(f"File {file_path} already exists in vector store (checksum: {checksum})")
                
            return needs_processing
                
        except Exception as e:
            logger.error(f"Error checking file in vector store {file_path}: {str(e)}")
            # If there's an error checking, assume we need to process the file
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
            
            # Get file stats
            file_stats = os.stat(path_obj)
            
            # Calculate relative path
            try:
                relative_path = str(path_obj.relative_to(base_path_obj))
            except ValueError:
                relative_path = str(path_obj.relative_to(Path.cwd()))
            
            # Calculate checksum
            checksum = self.calculate_checksum(str(path_obj))
            
            # Check if file needs processing
            needs_processing = self.check_existing_file(str(path_obj), checksum)
            
            metadata = {
                # File location information
                "path": str(path_obj),
                "relative_path": relative_path,
                "directory": str(path_obj.parent),
                "filename_full": path_obj.name,
                "filename_stem": path_obj.stem,
                "file_type": path_obj.suffix.lstrip('.'),
                
                # Timestamps
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                
                # Processing status
                "checksum": checksum,
                "needs_processing": needs_processing,
                "chunk_count": 0,  # Will be updated by document processor
                "total_tokens": 0  # Will be updated by document processor
            }
            
            logger.debug(f"Metadata collected for {path_obj}: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {str(e)}", exc_info=True)
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
            - processed: Boolean indicating if document was successfully processed
            - processor_output: Output from document processor if successful
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
                    should_process, reason, processor_class = self.should_process_file(path)
                    if should_process:
                        file_info = self.get_file_metadata(path, str(base_path))
                        if file_info['needs_processing']:
                            # Process the document using appropriate processor
                            success, error = self.process_document(path, processor_class)
                            file_info.update({
                                'processed': success,
                                'processor_error': error,
                                'processor_type': processor_class.__name__,
                                'processing_completed': datetime.utcnow().isoformat() if success else None
                            })
                        file_info_list.append(file_info)
                        if success:
                            logger.info(f"Successfully processed file: {path}")
                        else:
                            logger.warning(f"Failed to process file: {path} - {error}")
                    else:
                        skipped_files.append({"path": path, "reason": reason})
                        logger.debug(f"Skipping file {path}: {reason}")
                except Exception as e:
                    error_details = handle_exception(
                        e,
                        f"Error processing file {path}",
                        reraise=False
                    )
                    skipped_files.append({
                        "path": path,
                        "reason": error_details['message']
                    })
            
            # Handle directory
            else:
                logger.debug(f"Beginning directory walk: {path}")
                for root, dirs, files in os.walk(path):
                    logger.debug(f"Scanning directory: {root}")
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        logger.debug(f"Processing file: {file_path}")
                        
                        try:
                            should_process, reason, processor_class = self.should_process_file(file_path)
                            if should_process:
                                file_info = self.get_file_metadata(file_path, str(base_path))
                                if file_info['needs_processing']:
                                    # Process the document using appropriate processor
                                    success, error = self.process_document(file_path, processor_class)
                                    file_info.update({
                                        'processed': success,
                                        'processor_error': error,
                                        'processor_type': processor_class.__name__,
                                        'processing_completed': datetime.utcnow().isoformat() if success else None
                                    })
                                file_info_list.append(file_info)
                                if success:
                                    logger.info(f"Successfully processed file: {file_path}")
                                else:
                                    logger.warning(f"Failed to process file: {file_path} - {error}")
                            else:
                                skipped_files.append({"path": file_path, "reason": reason})
                                logger.debug(f"Skipping file {file_path}: {reason}")
                                
                        except Exception as e:
                            error_details = handle_exception(
                                e,
                                f"Error processing file {file_path}",
                                reraise=False
                            )
                            skipped_files.append({
                                "path": file_path,
                                "reason": error_details['message']
                            })
                            continue

            # Log summary with more details
            processed_count = sum(1 for f in file_info_list if f.get('processed', False))
            failed_count = sum(1 for f in file_info_list if not f.get('processed', True))
            logger.info(
                f"File scan completed. Successfully processed {processed_count} files, "
                f"failed to process {failed_count} files, skipped {len(skipped_files)} files"
            )
            
            if skipped_files:
                logger.debug("Summary of skipped files:")
                for skip_info in skipped_files:
                    logger.debug(f"Skipped {skip_info['path']}: {skip_info['reason']}")
            
            return file_info_list
            
        except Exception as e:
            error_details = handle_exception(
                e,
                f"Error scanning path {path}",
                reraise=False
            )
            logger.error(f"File processing failed: {error_details['message']}")
            return []

    def process_document(self, file_path: str, processor_class: Type[BaseDocumentProcessor]) -> Tuple[bool, Optional[str]]:
        """Process a document using the appropriate document processor.
        
        Args:
            file_path: Path to the file to process
            processor_class: The document processor class to use
            
        Returns:
            Tuple[bool, Optional[str]]: A tuple containing (success, error_message)
            where error_message is None if processing was successful
        """
        try:
            logger.debug(f"Processing document {file_path} with {processor_class.__name__}")
            processor = processor_class(self.config)
            
            # Get file stats
            file_stats = os.stat(file_path)
            path_obj = Path(file_path).resolve()
            
            # Create comprehensive initial metadata as per architecture spec
            initial_metadata = {
                # File location information
                'path': str(path_obj),
                'relative_path': str(path_obj.relative_to(Path.cwd())),
                'directory': str(path_obj.parent),
                'filename_full': path_obj.name,
                'filename_stem': path_obj.stem,
                'file_type': path_obj.suffix.lstrip('.'),
                
                # Timestamps
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                
                # Processing information
                'processor_type': processor_class.__name__,
                'processing_started': datetime.utcnow().isoformat(),
                'checksum': self.calculate_checksum(str(path_obj)),
                
                # These will be updated by the document processor
                'chunk_count': 0,
                'total_tokens': 0
            }
            
            # Process the document
            try:
                processor.process(str(path_obj), initial_metadata)
                logger.info(f"Successfully processed document: {file_path}")
                return True, None
                
            except Exception as proc_error:
                error_details = handle_exception(
                    proc_error, 
                    f"Document processor error for {file_path}",
                    reraise=False
                )
                return False, error_details['message']
            
        except FileAccessError as e:
            error_details = handle_exception(e, context="File access error", reraise=False)
            return False, error_details['message']
        except Exception as e:
            error_details = handle_exception(
                e, 
                f"Error in process_document for {file_path}",
                reraise=False
            )
            return False, error_details['message']

    def process_files(self, base_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Process all files in the given path.
        
        Args:
            base_path: Base path to start scanning from
            
        Returns:
            List[Dict[str, Any]]: List of processed file information dictionaries
        """
        logger.info(f"Starting file processing from base path: {base_path}")
        processed_files = []
        skipped_files = []
        
        try:
            base_path = Path(base_path)
            if not base_path.exists():
                raise FileAccessError(f"Base path does not exist: {base_path}")
                
            if base_path.is_file():
                logger.debug(f"Processing individual file: {base_path}")
                try:
                    should_process, reason, processor_class = self.should_process_file(str(base_path))
                    if should_process:
                        file_info = self.get_file_metadata(str(base_path), str(base_path))
                        if file_info:
                            # Process the document
                            success, error = self.process_document(str(base_path), processor_class)
                            if success:
                                processed_files.append(file_info)
                            else:
                                skipped_files.append({
                                    "path": str(base_path),
                                    "reason": error
                                })
                    else:
                        skipped_files.append({
                            "path": str(base_path),
                            "reason": reason
                        })
                except Exception as e:
                    error_details = handle_exception(
                        e,
                        f"Error processing file {base_path}",
                        reraise=False
                    )
                    skipped_files.append({
                        "path": str(base_path),
                        "reason": error_details['message']
                    })
            else:
                logger.debug(f"Processing directory: {base_path}")
                for file_path in base_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            should_process, reason, processor_class = self.should_process_file(str(file_path))
                            if should_process:
                                file_info = self.get_file_metadata(str(file_path), str(base_path))
                                if file_info:
                                    # Process the document
                                    success, error = self.process_document(str(file_path), processor_class)
                                    if success:
                                        processed_files.append(file_info)
                                    else:
                                        skipped_files.append({
                                            "path": str(file_path),
                                            "reason": error
                                        })
                            else:
                                skipped_files.append({
                                    "path": str(file_path),
                                    "reason": reason
                                })
                        except Exception as e:
                            error_details = handle_exception(
                                e,
                                f"Error processing file {file_path}",
                                reraise=False
                            )
                            skipped_files.append({
                                "path": str(file_path),
                                "reason": error_details['message']
                            })
            
            # Log summary
            logger.info(
                f"File processing complete. "
                f"Processed: {len(processed_files)}, "
                f"Skipped: {len(skipped_files)}"
            )
            
            if skipped_files:
                logger.debug("Summary of skipped files:")
                for skip_info in skipped_files:
                    logger.debug(f"Skipped {skip_info['path']}: {skip_info['reason']}")
            
            return processed_files
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error during file processing",
                reraise=False
            )
            logger.error(f"File processing failed: {error_details['message']}")
            return []

    def process_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Process a single file with proper error handling."""
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
                raise ValidationError(f"File validation failed: {validation_error}")
            
            # Process file
            success, result = self._process_file_content(file_path)
            if not success:
                raise FileProcessingError(f"File processing failed: {result}")
            
            return True, None
            
        except (FileAccessError, ValidationError, FileProcessingError) as e:
            error_details = handle_exception(e, context="File processing error", reraise=False)
            return False, error_details['message']
        except Exception as e:
            error_details = handle_exception(
                e,
                f"Unexpected error processing {file_path}",
                reraise=False
            )
            return False, error_details['message']

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