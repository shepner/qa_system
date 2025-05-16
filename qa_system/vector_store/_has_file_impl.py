"""
@file: _has_file_impl.py
Provides an implementation for checking if a file with a given hash exists in the vector store collection.

This module defines a helper function for verifying file existence by checksum, with logging for traceability and error handling.
"""
import logging

logger = logging.getLogger(__name__)

def _has_file_impl(self, file_hash: str) -> bool:
    """
    Check if a file with the given hash exists in the vector store collection.

    Args:
        self: The instance containing the vector store collection. Must have a 'collection' attribute with a 'get' method.
        file_hash (str): The checksum/hash of the file to check for existence.

    Returns:
        bool: True if a file with the given hash exists, False otherwise.

    Logs:
        - Debug information about the query and its result.
        - Errors encountered during the check.
    """
    try:
        logger.debug(f"Querying vector store for checksum: {file_hash}")
        results = self.collection.get(where={"checksum": file_hash}, limit=1)
        logger.debug(f"Vector store query result for checksum {file_hash}: {results}")
        return bool(results and results.get('ids'))
    except Exception as e:
        logger.error(f"Failed to check file existence in vector store: {e}")
        return False 