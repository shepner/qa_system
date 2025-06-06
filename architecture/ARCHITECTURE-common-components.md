---
source: internal
tags:
  - architecture
  - documentation
  - system-design
  - components
version: 1.0
last_updated: 2024-03-20
---

# Common Components

## Table of Contents
1. [Overview](#1-overview)
2. [Main Module](#2-main-module)
   - [2.1 Purpose](#21-purpose)
   - [2.2 Interface](#22-interface)
   - [2.3 Usage](#23-usage)
3. [Configuration Module](#3-configuration-module)
   - [3.1 Purpose](#31-purpose)
   - [3.2 Interface](#32-interface)
   - [3.3 Usage](#33-usage)
4. [Logging Setup](#4-logging-setup)
   - [4.1 Purpose](#41-purpose)
   - [4.2 Interface](#42-interface)
   - [4.3 Usage](#43-usage)
5. [Vector Database](#5-vector-database)
   - [5.1 Core Components](#51-core-components)
   - [5.2 Data Model](#52-data-model)
   - [5.3 Operations](#53-operations)
   - [5.4 Configuration](#54-configuration)
   - [5.5 Integration](#55-integration)
6. [Exceptions Module](#6-exceptions-module)
   - [6.1 Purpose](#61-purpose)
   - [6.2 Core Exceptions](#62-core-exceptions)
   - [6.3 Usage](#63-usage)

## 1. Overview
This document details the common components used across all flows in the File Embedding System. Each component is designed to be modular, reusable, and follows consistent configuration patterns.

## 2. Main Module (__main__.py)

### 2.1 Purpose
Command-line interface and entry point for the File Embedding System.

### 2.2 Interface
- **Input Arguments**:
  - `--add`: Path to process (can be a directory or individual file) (See: [ARCHITECTURE-add-flow](ARCHITECTURE-add-flow.md))
  - `--list`: List the contents of the vector data store (See: [ARCHITECTURE-list-flow](ARCHITECTURE-list-flow.md))
  - `--remove`: Remove data from the vector data store (See: [ARCHITECTURE-remove-flow](ARCHITECTURE-remove-flow.md))
  - `--query`: Enter interactive chat mode (See: [ARCHITECTURE-query-flow](ARCHITECTURE-query-flow.md))
  - `--config`: Path to configuration file (default: './config/config.yaml')
  - `--debug`: Flag to enable debug logging
- **Output**:
  - Exit code 0: Successful execution
  - Exit code 1: Error occurred
  - Console output for progress and errors
  - Log file entries

### 2.3 Usage
```bash
# Process files or directories (add flow)
python -m qa_system --add /path/to/docs                     # Process entire directory
python -m qa_system --add /path/to/docs/specific_file.md    # Process single file
python -m qa_system --add file1.md --add file2.pdf         # Process multiple files

# List contents of vector store (list flow)
python -m qa_system --list                                  # List all contents
python -m qa_system --list --filter "*.md"                  # List only markdown files

# Remove data from vector store (remove flow)
python -m qa_system --remove /path/to/docs/old_file.md      # Remove specific file
python -m qa_system --remove /path/to/old_docs/             # Remove entire directory
python -m qa_system --remove --filter "*.pdf"               # Remove all PDF files

# Interactive chat mode (query flow)
python -m qa_system --query                                 # Start interactive chat
python -m qa_system --query "What is the project about?"    # Single query mode

# Configuration and debugging
python -m qa_system --add /path/to/docs --config custom_config.yaml    # Use custom config
python -m qa_system --add /path/to/docs --debug                        # Enable debug logging

# Combined usage
python -m qa_system --add /path/to/docs --config custom_config.yaml --debug    # Add with custom config and debug
python -m qa_system --list --config custom_config.yaml                         # List with custom config
python -m qa_system --query --debug                                           # Query with debug logging
```

## 3. Configuration Module (config.py)

### 3.1 Purpose
Manages system configuration loading and access with support for file-based and environment variable configuration.

### 3.2 Interface
- **Input**:
  - Configuration file path (optional, defaults to './config/config.yaml')
  - Environment variables with 'QA_' prefix
- **Output**: Config object with sections:
  - `LOGGING`: Logging settings
    - `LEVEL`: Log level setting (default: "INFO")
    - `LOG_FILE`: Path to log file (default: "logs/qa_system.log")
  - `SECURITY`: Security-related settings (loaded from environment variables)
    - `GOOGLE_APPLICATION_CREDENTIALS`: Google Cloud credentials path
    - `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
    - `GOOGLE_VISION_API_KEY`: Vision API key
  - `FILE_SCANNER`: File scanning settings
    - `DOCUMENT_PATH`: Default path for documents (default: "./docs")
    - `ALLOWED_EXTENSIONS`: List of file extensions to process
    - `EXCLUDE_PATTERNS`: List of patterns to exclude
    - `HASH_ALGORITHM`: Hash algorithm for checksums (default: "sha256")
    - `SKIP_EXISTING`: Whether to skip files already in vector db (default: true)
  - `DATA_REMOVER`: File matcher configuration
    - `RECURSIVE`: Enable recursive matching (default: true)
    - `CASE_SENSITIVE`: Case sensitivity in matching (default: false)
    - `REQUIRE_CONFIRMATION`: Require confirmation before removing (default: true)
  - `DOCUMENT_PROCESSING`: Document processing settings
    - `MAX_CHUNK_SIZE`: Maximum size of text chunks (default: 3072)
    - `MIN_CHUNK_SIZE`: Minimum chunk size (default: 1024)
    - `CHUNK_OVERLAP`: Overlap between chunks (default: 768)
    - `CONCURRENT_TASKS`: Number of parallel tasks (default: 6)
    - `BATCH_SIZE`: Documents per batch (default: 50)
    - `PRESERVE_SENTENCES`: Ensure chunks don't break sentences (default: true)
    - `PDF_HEADER_RECOGNITION`: Header pattern recognition settings
      - `ENABLED`: Enable header recognition (default: true)
      - `MIN_FONT_SIZE`: Minimum font size for headers (default: 12)
      - `PATTERNS`: Regular expressions for header detection
      - `MAX_HEADER_LENGTH`: Maximum header line length (default: 100)
    - `VISION_API`: Image processing settings
      - `ENABLED`: Enable Vision API processing (default: true)
      - `FEATURES`: List of enabled Vision API features
      - `MAX_RESULTS`: Maximum results per feature (default: 50)
  - `EMBEDDING_MODEL`: Model configuration
    - `MODEL_NAME`: Name of Gemini model (default: "models/embedding-001")
    - `BATCH_SIZE`: Embedding batch size (default: 15)
    - `MAX_LENGTH`: Maximum text length (default: 3072)
    - `DIMENSIONS`: Output dimensions (default: 768)
  - `VECTOR_STORE`: Vector database configuration
    - `TYPE`: Vector store implementation (default: "chroma")
    - `PERSIST_DIRECTORY`: Data storage location (default: "./data/vector_store")
    - `COLLECTION_NAME`: Name of collection (default: "qa_documents")
    - `DISTANCE_METRIC`: Similarity metric (default: "cosine")
    - `TOP_K`: Number of results to retrieve (default: 40)

### 3.3 Usage
```python
from qa_system.config import get_config

# Load with default config path
config = get_config()

# Load with specific config path
config = get_config("./my_config.yaml")

# Access configuration sections
logging_config = config.get_nested('LOGGING')
print(f"Log level: {logging_config['LEVEL']}")  # "INFO"
print(f"Log file: {logging_config['LOG_FILE']}")  # "logs/qa_system.log"

# Access nested configuration
doc_processing = config.get_nested('DOCUMENT_PROCESSING')
print(f"Max chunk size: {doc_processing['MAX_CHUNK_SIZE']}")  # 3072
print(f"Batch size: {doc_processing['BATCH_SIZE']}")  # 50

# Access deeply nested configuration
vision_config = config.get_nested('DOCUMENT_PROCESSING.VISION_API')
print(f"Vision API enabled: {vision_config['ENABLED']}")  # true
print(f"Max results: {vision_config['MAX_RESULTS']}")  # 50

# Access with default values
custom_batch_size = config.get_nested('DOCUMENT_PROCESSING.BATCH_SIZE', default=10)
custom_log_level = config.get_nested('LOGGING.LEVEL', default="WARNING")
```

## 4. Logging Setup (logging_setup.py)

### 4.1 Purpose
Provides centralized logging configuration with support for file and console output.

### 4.2 Interface
- **Input Configuration**:
  - From `config.LOGGING` section:
    - `LOG_FILE`: Path to the log file (default: "logs/qa_system.log")

- **Output Configuration**:
  - Format: `%(asctime)s - %(name)s - %(level) - %(message)s`
  - Location: Specified by `LOG_FILE` setting

### 4.3 Usage
- **Initialize logging** at application startup using configuration values from the `LOGGING` section.
- **Obtain module-specific loggers** using `logging.getLogger(__name__)` for consistent, contextual logging.
- **Log messages** at appropriate levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) throughout the codebase.


Example setup and usage:
```python
from qa_system.config import get_config
from qa_system.logging_setup import setup_logging
import logging

# Initialize logging from config
env_config = get_config()
setup_logging(
    LOG_FILE=env_config.get_nested('LOGGING.LOG_FILE'),
    LEVEL=env_config.get_nested('LOGGING.LEVEL', default="DEBUG")
)

# Get a logger for your module
logger = logging.getLogger(__name__)

# Log messages at various levels
logger.info("System started")
logger.warning("Low disk space")
logger.error("Failed to process file")

# Log with extra context
# (If you want extra fields to appear in the log output, include them in the message string)
doc_path = "/path/to/doc.pdf"
logger.info("Processing document: document_path=%s, document_type=%s", doc_path, "PDF")

# Note: The 'extra' parameter can be used to add custom fields to the log record, but these will only appear in the output if your log formatter includes them (e.g., %(document_path)s). For most use cases, include context directly in the message as shown above.

# Log exceptions with stack trace
try:
    process_document(doc_path)
except Exception as e:
    logger.exception("Error processing document")

# Change log level at runtime
logging.getLogger().setLevel(logging.DEBUG)
logger.debug("Debug logging enabled")
```

Example log output:
```
2024-03-20 15:30:45 - qa_system.processor - INFO - System started
2024-03-20 15:30:45 - qa_system.processor - WARNING - Low disk space
2024-03-20 15:30:46 - qa_system.processor - INFO - Processing document: document_path=/path/to/doc.pdf, document_type=PDF
2024-03-20 15:30:47 - qa_system.processor - ERROR - Error processing document
Traceback (most recent call last):
  ...
```

## 5. Vector Database
- **Purpose**: Stores vector embeddings and metadata for efficient retrieval and querying
- **Key Functions**:
  - Add embeddings to the database
  - Retrieve embeddings based on query
  - Manage database operations
- **Technologies**:
  - ChromaDB for vector storage
  - Python for database operations

#### 5.1 Core Components

##### Vector Store Interface
- **Module**: `vector_store.py`
- **Responsibilities**:
  - Defines common interface for vector database operations
  - Manages database connections and lifecycle
  - Implements retry logic and error handling
  - Provides transaction management
  - Handles batch operations efficiently

##### ChromaDB Implementation
- **Purpose**: Primary vector database implementation using ChromaDB
- **Features**:
  - Persistent storage of embeddings
  - Efficient similarity search
  - Metadata filtering and querying
  - Collection management
  - Automatic index optimization
  - Concurrent access handling

#### 5.2 Data Model

##### Document Records
- **Embedding Data**:
  - Vector values (768 or 1024 or more dimensions)
  - Chunk text content
  - Distance metric (cosine similarity)
  - Embedding model information
  
- **Metadata**:
  - Document identifiers
  - File information
  - Chunk information
  - Processing timestamps
  - Document relationships
  - Custom attributes

##### Collection Structure
- **Organization**:
  - Multiple collections support
  - Namespace isolation
  - Version tracking
  - Index management
  - Backup/restore points

#### 5.3 Operations

##### Adding Documents
- **Process**:
  - Validate input data
  - Generate unique identifiers
  - Batch processing for efficiency
  - Update existing records if needed
  - Maintain consistency with retries
  - Log operations for tracking

##### Querying
- **Capabilities**:
  - Similarity search with configurable k
  - Metadata filtering
  - Range queries
  - Aggregations
  - Faceted search
  - Combined queries (vector + metadata)


#### 5.4 Configuration
```yaml
VECTOR_STORE:
  # Database Settings
  TYPE: "chroma"
  PERSIST_DIRECTORY: "./vector_store"
  COLLECTION_NAME: "documents"
  
  # Search Configuration
  DISTANCE_METRIC: "cosine"
  TOP_K: 40
```

#### 5.5 Integration

##### Usage Example
```python
from embed_files.vector_store import ChromaVectorStore
from embed_files.config import get_config
from datetime import datetime

# Initialize store
config = get_config()
vector_store = ChromaVectorStore(config)

try:
    # Example metadata incorporating all fields from section 3.2.1
    metadata = {
        # File location information
        "path": "/Users/username/documents/research/paper.pdf",
        "relative_path": "research/paper.pdf",
        "directory": "research",
        "filename_full": "paper.pdf",
        "filename_stem": "paper",
        "file_type": "pdf",
        
        # Timestamps
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        
        # Processing information
        "chunk_count": 15,
        "total_tokens": 4500,
        "checksum": "abc123def456...",
        "chunk_index": 0  # Index of current chunk
    }

    # Add embeddings with complete metadata
    vector_store.add_embeddings(
        embeddings=[embedding_vector],
        texts=["chunk text content"],
        metadatas=[metadata]
    )

    # Query similar documents with metadata filtering
    results = vector_store.query(
        query_vector=query_embedding,
        top_k=5,
        filter_criteria={
            "file_type": "pdf",
            "directory": "research",
            "chunk_count": {"$gt": 10}  # Example of numeric filtering
        }
    )

    # Delete documents by various criteria
    try:
        # Delete by file path
        vector_store.delete(
            filter_criteria={"path": "/Users/username/documents/research/paper.pdf"}
        )

        # Delete by directory
        vector_store.delete(
            filter_criteria={"directory": "research"}
        )

        # Delete by multiple criteria
        vector_store.delete(
            filter_criteria={
                "file_type": "pdf",
                "created_at": {"$lt": "2024-01-01T00:00:00"}  # Delete PDFs created before 2024
            }
        )

        # Delete specific chunks
        vector_store.delete(
            filter_criteria={
                "path": "/Users/username/documents/research/paper.pdf",
                "chunk_index": {"$in": [0, 1, 2]}  # Delete specific chunks
            }
        )

        # Batch delete with confirmation
        deleted_count = vector_store.delete(
            filter_criteria={"directory": "old_docs"},
            require_confirmation=True  # Prompts for confirmation before deletion
        )
        logger.info(f"Deleted {deleted_count} documents")

    except VectorStoreError as e:
        logger.error(f"Deletion failed: {str(e)}")
        raise

except VectorStoreError as e:
    logger.error(f"Vector store operation failed: {str(e)}")
    raise

##### Error Handling
```python
class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass

class ConnectionError(VectorStoreError):
    """Database connection errors."""
    pass

class QueryError(VectorStoreError):
    """Query execution errors."""
    pass

class ValidationError(VectorStoreError):
    """Data validation errors."""
    pass
```

## 6. Exceptions Module (exceptions.py)

### 6.1 Purpose
Provides a centralized set of exception classes for consistent error handling across the system. All custom exceptions inherit from a base system exception to ensure consistent error handling patterns.

### 6.2 Core Exceptions
```python
class QASystemError(Exception):
    """Base exception for all system errors."""
    pass

class ValidationError(QASystemError):
    """Raised when input validation fails."""
    pass

class ProcessingError(QASystemError):
    """Raised when document processing fails."""
    pass

class ConfigurationError(QASystemError):
    """Raised when configuration is invalid or missing."""
    pass

class StorageError(QASystemError):
    """Raised when storage operations fail."""
    pass

class EmbeddingError(QASystemError):
    """Raised when embedding generation fails."""
    pass

class APIError(QASystemError):
    """Raised when external API calls fail."""
    pass
```

### 6.3 Usage
```python
from qa_system.exceptions import ValidationError, ProcessingError

def process_document(file_path: str) -> Dict:
    """Process a document with proper error handling.
    
    Args:
        file_path: Path to document to process
        
    Returns:
        Processed document data
        
    Raises:
        ValidationError: If document validation fails
        ProcessingError: If processing fails
    """
    try:
        # Validate input
        if not os.path.exists(file_path):
            raise ValidationError(f"File not found: {file_path}")
            
        # Process document
        result = perform_processing(file_path)
        if not result:
            raise ProcessingError(f"Failed to process {file_path}")
            
        return result
        
    except Exception as e:
        # Convert unknown errors to ProcessingError
        if not isinstance(e, (ValidationError, ProcessingError)):
            raise ProcessingError(f"Unexpected error: {str(e)}") from e
        raise
```

Example error handling pattern:
```python
try:
    result = process_document(file_path)
except ValidationError as e:
    logger.error(f"Validation failed: {str(e)}")
    # Handle validation error (e.g., skip file)
except ProcessingError as e:
    logger.error(f"Processing failed: {str(e)}")
    # Handle processing error (e.g., retry or log)
except QASystemError as e:
    logger.error(f"System error: {str(e)}")
    # Handle any other system error
except Exception as e:
    logger.critical(f"Unexpected error: {str(e)}")
    # Handle unexpected errors
```
