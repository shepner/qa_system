"""Exceptions module for the QA system."""

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

class VectorStoreError(QASystemError):
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