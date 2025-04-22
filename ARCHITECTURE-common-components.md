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

This document details the common components used across all flows in the File Embedding System.

## Main Module (__main__.py)

- **Purpose**: Command-line interface and entry point
- **Input Arguments**:
  - `--add`: Path to process (can be a directory or individual file)
  - `--config`: Path to configuration file (default: './config/config.yaml')
  - `--debug`: Flag to enable debug logging
- **Output**:
  - Exit code 0: Successful execution
  - Exit code 1: Error occurred
  - Console output for progress and errors
  - Log file entries
- **Usage**:
```bash
# Process a directory with default config
python -m embed_files --add /path/to/docs

# Process a single file with custom config
python -m embed_files --add /path/to/docs/specific_file.md --config custom_config.yaml

# Process multiple inputs with debug enabled
python -m embed_files --add /path/to/docs/file1.md --add /path/to/docs/file2.pdf --debug
```

## Configuration Module (config.py)

- **Purpose**: Manages system configuration loading and access
- **Input**:
  - Configuration file path (optional, defaults to './config/config.yaml')
  - Environment variables with 'QA_' prefix
- **Output**: Config object with sections:
  - `LOGGING`: Logging settings
    - `LEVEL`: Log level setting
    - `LOG_FILE`: Path to log file
  - `SECURITY`: Security-related settings
  - `FILE_SCANNER`: File scanning settings
    - `ALLOWED_EXTENSIONS`: List of file extensions to process
    - `EXCLUDE_PATTERNS`: List of patterns to exclude
    - `HASH_ALGORITHM`: Hash algorithm for checksums (default: 'sha256')
    - `DOCUMENT_PATH`: Default path for documents to be processed
  - `DOCUMENT_PROCESSING`: Document processing settings
    - `MAX_CHUNK_SIZE`: Maximum size of text chunks
    - `CHUNK_OVERLAP`: Overlap between chunks
    - `CONCURRENT_TASKS`: Number of parallel tasks
    - `BATCH_SIZE`: Documents per batch
  - `EMBEDDING_MODEL`: Model configuration
    - `MODEL_NAME`: Name of Gemini model
    - `BATCH_SIZE`: Embedding batch size
    - `MAX_LENGTH`: Maximum text length
    - `DIMENSIONS`: Output dimensions
  - `VECTOR_STORE`: Vector database configuration
    - `TYPE`: Vector store implementation
    - `PERSIST_DIRECTORY`: Data storage location
    - `COLLECTION_NAME`: Name of collection
    - `DISTANCE_METRIC`: Similarity metric
    - `TOP_K`: Number of results to retrieve
- **Usage**:
```python
from embed_files.config import get_config

# Load with default config path
config = get_config()

# Load with specific config path
config = get_config("./my_config.yaml")

# Access configuration sections
vector_store_config = config.get_nested('VECTOR_STORE')
```

## Logging Setup (logging_setup.py)
- **Purpose**: Provides centralized logging configuration with support for file and console output, log rotation, and debug level control
- **Input**:
  - `LOG_FILE`: Path to the log file (from config.LOGGING.LOG_FILE)
  - `LOG_LEVEL`: Logging level (from config.LOGGING.LEVEL, default: "INFO")
  - `DEBUG`: Flag to enable debug logging (default: False)
- **Output**:
  - Configured root logger with both file and console handlers
  - Rotating log files with specified size limits and backup counts
  - Formatted log messages with timestamp, logger name, level, and message
- **Usage**:
```python
from embed_files.config import get_config
from embed_files.logging_setup import setup_logging

# Load configuration
config = get_config()

# Setup logging using configuration values
setup_logging(
    LOG_FILE=config.get_nested('LOGGING.LOG_FILE'),
    LOG_LEVEL=config.get_nested('LOGGING.LEVEL', default="INFO"),
    DEBUG=config.get_nested('LOGGING.DEBUG', default=False)
)
```

Key Features:
- Configuration-driven setup
- Automatic log directory creation
- Log rotation to manage file sizes
- Consistent log formatting across handlers
- Debug mode support
- Console output for immediate feedback
- Thread-safe logging implementation

Log Format:
```
YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - Message
```

Example Log Entry:
```
2024-03-20 14:30:45 - embed_files.document_processor - INFO - Processing document: example.pdf
```