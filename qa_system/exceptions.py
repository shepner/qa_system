"""
Exception classes for the QA system.

This module provides a centralized set of exception classes for consistent error handling
across the system. All custom exceptions inherit from a base QASystemError to ensure
consistent error handling patterns.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QASystemError(Exception):
    """Base exception for all system errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.code = code or self.__class__.__name__
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for logging/serialization."""
        return {
            'error': self.code,
            'message': str(self),
            'details': self.details
        }

class ConfigurationError(QASystemError):
    """Raised when configuration is invalid or missing."""
    pass

class ValidationError(QASystemError):
    """Raised when input validation fails."""
    pass

class FileError(QASystemError):
    """Base class for file-related errors."""
    pass

class FileAccessError(FileError):
    """Raised when file cannot be accessed."""
    pass

class FileProcessingError(FileError):
    """Raised when file processing fails."""
    pass

class StorageError(QASystemError):
    """Base class for storage-related errors."""
    pass

class VectorStoreError(StorageError):
    """Raised when vector store operations fail."""
    pass

class DatabaseError(StorageError):
    """Raised when database operations fail."""
    pass

class QueryError(QASystemError):
    """Raised when query processing fails."""
    pass

class EmbeddingError(QASystemError):
    """Raised when embedding generation fails."""
    pass

class APIError(QASystemError):
    """Raised when external API calls fail."""
    pass

class SecurityError(QASystemError):
    """Raised when security-related operations fail."""
    pass

class ResourceError(QASystemError):
    """Raised when resource limits are exceeded or resources are unavailable."""
    pass

# Utility functions for exception handling
def handle_exception(e: Exception, context: str = "", reraise: bool = True) -> Optional[Dict[str, Any]]:
    """Handle exceptions in a consistent way across the system.
    
    Args:
        e: The exception to handle
        context: Additional context about where the error occurred
        reraise: Whether to re-raise the exception after handling
        
    Returns:
        Optional dictionary with error details if not re-raising
        
    Raises:
        QASystemError: If reraise is True and the exception needs to be propagated
    """
    error_details = {
        'error_type': type(e).__name__,
        'message': str(e),
        'context': context
    }
    
    if isinstance(e, QASystemError):
        error_details.update(e.to_dict())
        logger.error(
            f"{context}: {str(e)}",
            extra=error_details,
            exc_info=True
        )
        if reraise:
            raise
    else:
        # Convert unknown exceptions to QASystemError
        error = QASystemError(
            f"Unexpected error in {context}: {str(e)}",
            code="UNEXPECTED_ERROR",
            details=error_details
        )
        logger.error(
            str(error),
            extra=error.to_dict(),
            exc_info=True
        )
        if reraise:
            raise error from e
            
    return error_details if not reraise else None 