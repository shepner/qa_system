"""
Custom exception hierarchy for the QA system.

This module defines all custom exceptions used throughout the QA system, organized by logical error domains:

- QASystemError: Base for all system-level errors
- ValidationError: For input or data validation failures
- ProcessingError: For document processing errors
- ConfigurationError: For configuration issues
- StorageError: For storage-related errors
- EmbeddingError: For embedding generation failures
- APIError: For external API call failures

Vector store-specific errors inherit from VectorStoreError:
- VectorStoreError: Base for vector store errors
- ConnectionError: For database connection issues
- QueryError: For query execution problems
- DocumentNotFoundError: When a document is missing in the vector store
- RemovalError: For failures during removal operations (with cleanup info)
"""

class QASystemError(Exception):
    """Base exception for all QA system errors."""
    pass

class ValidationError(QASystemError):
    """Raised when input or data validation fails anywhere in the system."""
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

class VectorStoreError(QASystemError):
    """Base exception for vector store operations."""
    pass

class ConnectionError(VectorStoreError):
    """Raised when a database connection error occurs in the vector store."""
    pass

class QueryError(VectorStoreError):
    """Raised when a query execution error occurs in the vector store."""
    pass

class DocumentNotFoundError(VectorStoreError):
    """Raised when a document is not found in the vector store."""
    pass

class RemovalError(VectorStoreError):
    """Raised when a removal operation fails in the vector store.

    Attributes:
        requires_cleanup (bool): Whether additional cleanup is required.
        document_id (str, optional): The ID of the document related to the failure.
    """
    def __init__(self, message, requires_cleanup=False, document_id=None):
        super().__init__(message)
        self.requires_cleanup = requires_cleanup
        self.document_id = document_id

class RateLimitError(APIError):
    """Raised when an API call fails due to rate limiting (HTTP 429 or RESOURCE_EXHAUSTED)."""
    pass 