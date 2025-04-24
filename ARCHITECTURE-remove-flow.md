---
source: internal
tags:
  - architecture
  - documentation
  - system-design
  - remove-flow
version: 1.0
last_updated: 2024-03-20
---

# Table of Contents
1. [Overview](#1-overview)
2. [System Flow](#2-system-flow)
3. [Components](#3-components)
   3.1. [Remove Module](#31-remove-module)
      3.1.1. [Remove Module (remove_handler.py)](#311-remove-module-remove_handlerpy)
   3.2. [Vector Database](#32-vector-database)
4. [Error Handling](#4-error-handling)
   4.1. [Common Error Scenarios](#41-common-error-scenarios)
   4.2. [Recovery Procedures](#42-recovery-procedures)
5. [Configuration](#5-configuration)
6. [Usage Examples](#6-usage-examples)
   6.1. [Command Line Interface](#command-line-interface)
   6.2. [Python API](#python-api)

# 1. Overview
The remove flow handles the safe deletion of files and their associated data from the system. It ensures that all vectors, metadata, and related information are properly cleaned up while maintaining system consistency.

# 2. System Flow

```mermaid
sequenceDiagram
    actor User
    participant Main
    participant Remove as Remove Module
    participant VDB as Vector Database

    User->>Main: Execute remove command
    Main->>Remove: Call with file/pattern
    Remove->>VDB: Query matching documents
    VDB-->>Remove: Return matching documents
    
    alt Matches found
        Remove->>VDB: Delete matched documents
        VDB->>VDB: Remove vectors & metadata
        VDB-->>Remove: Confirm deletion
        Remove-->>Main: Report success
        Main-->>User: Show success message
    else No matches found
        Remove-->>Main: Report no matches
        Main-->>User: Show no matches message
    end
```

# 3. Components

## 3.1. Remove Module
- **Purpose**: Handles pattern matching and removal of documents from the vector database
- **Dependencies**:
  - [Configuration Module](ARCHITECTURE-common-components.md#3-configuration-module) for removal settings
  - [Logging Setup](ARCHITECTURE-common-components.md#4-logging-setup) for operation tracking
  - [Vector Database](ARCHITECTURE-common-components.md#5-vector-database) for document removal
- **Key Functions**:
  - Pattern matching for file selection
    - Supports exact file paths
    - Supports glob patterns (*.pdf, docs/*.txt, etc)
    - Directory path handling
  - Document querying
    - Searches vector database for matching documents
    - Returns list of matches with metadata
  - Document removal
    - Batch removal of matched documents
    - Cleanup of associated vectors and metadata
    - Consistency verification
- **Technologies**:
  - Python pathlib for path handling
  - glob pattern matching
  - Vector database operations

### 3.1.1. Remove Module (remove_handler.py)
- **Purpose**: Implements document removal logic including pattern matching, document querying, and vector store cleanup
- **Integration with Main**:
  - Receives paths/patterns from main's --remove argument
  - Receives configuration either from default or main's --config argument
  - Loads VECTOR_STORE configuration from config.py for Vector DB integration
- **Input**:
  - `pattern`: File or pattern to match (from main's --remove argument)
  - `vector_db`: Instance of Vector DB for document operations
  - Configuration settings (from config.REMOVE_HANDLER):
    - `BATCH_SIZE`: Number of documents to remove in each batch
    - `VERIFY_REMOVAL`: Whether to verify complete removal
    - `RECURSIVE`: Whether to match patterns recursively
- **Output**:
  - Dictionary containing:
    - `removed`: List of successfully removed documents
    - `failed`: List of documents that failed to remove
    - `not_found`: List of patterns with no matches
- **Key Functions**:
  - `find_matches`: Queries vector store for documents matching pattern
  - `remove_documents`: Removes matched documents from vector store
  - `verify_removal`: Confirms complete removal of documents and metadata

## 3.2. Vector Database
The remove flow utilizes the vector database component described in [ARCHITECTURE-common-components.md](ARCHITECTURE-common-components.md#5-vector-database) for all document operations including:
- Pattern matching against stored documents
- Removal of document vectors
- Cleanup of associated metadata
- Consistency verification

For configuration and usage details, see [Vector Database Configuration](ARCHITECTURE-common-components.md#52-configuration).

# 4. Error Handling

## 4.1 Common Error Scenarios
1. **Document Not Found**
   - System reports file not present in database
   - No changes made to system state

2. **Partial Removal**
   - System detects incomplete removal
   - Triggers additional cleanup
   - Logs warning for manual review

3. **Dependency Conflicts**
   - System identifies dependent documents
   - Provides option for cascade removal
   - Warns user of implications

## 4.2 Recovery Procedures
1. **Transaction Rollback**
   - All operations are transactional
   - System state preserved on failure
   - Detailed error logging for debugging

2. **Integrity Verification**
   - Post-removal consistency check
   - Automatic repair of minor issues
   - Notification for major problems

# 5. Configuration

The remove flow uses configuration settings from the following sections of the [common configuration](ARCHITECTURE-common-components.md#3-configuration-module):

- `DATA_REMOVER`: Remove-specific settings
  ```yaml
  DATA_REMOVER:
    # Enable recursive matching for directory patterns
    RECURSIVE: true
    # Case sensitivity in matching
    CASE_SENSITIVE: false
    # Whether to require confirmation before removing files
    REQUIRE_CONFIRMATION: true
  ```

- `VECTOR_STORE`: Vector database settings (see [Vector Database Configuration](ARCHITECTURE-common-components.md#54-configuration))

For detailed information about shared configuration settings, including logging and security, refer to the [Configuration Module documentation](ARCHITECTURE-common-components.md#3-configuration-module).

# 6. Usage Examples

## Command Line Interface
```bash
# Remove a single file
python -m qa_system --remove /path/to/document.pdf

# Remove multiple files
python -m qa_system --remove file1.md --remove file2.pdf

# Remove with pattern matching
python -m qa_system --remove "*.pdf"  # Remove all PDF files
python -m qa_system --remove "/docs/*.md"  # Remove markdown files in docs directory

# Remove with custom configuration
python -m qa_system --remove document.pdf --config custom_config.yaml

# Remove with debug logging
python -m qa_system --remove document.pdf --debug

# Remove directory recursively
python -m qa_system --remove /path/to/old_docs/

# Remove with confirmation prompt
python -m qa_system --remove document.pdf --confirm
```

## Python API
```python
from qa_system.remove_handler import RemoveHandler
from qa_system.config import get_config

# Initialize components
config = get_config()
handler = RemoveHandler(config)

# Remove using exact path
result = handler.remove_documents("/path/to/document.pdf")

# Remove using pattern
result = handler.remove_documents("*.pdf")

# Remove multiple patterns
result = handler.remove_documents([
    "/path/to/doc1.pdf",
    "docs/*.md"
])

# Remove with custom options
result = handler.remove_documents(
    "/path/to/document.pdf",
    recursive=True,
    verify_removal=True,
    require_confirmation=True
)

# Handle results
if result['removed']:
    print(f"Successfully removed: {result['removed']}")
if result['failed']:
    print(f"Failed to remove: {result['failed']}")
if result['not_found']:
    print(f"No matches found for: {result['not_found']}") 