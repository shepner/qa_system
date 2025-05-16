"""
@file: __init__.py
ChromaVectorStore: High-level interface for managing document embeddings and metadata in a ChromaDB-backed vector store.

This module provides a unified API for adding, querying, deleting, and searching document embeddings and their metadata.
It delegates implementation details to internal modules for modularity and maintainability.
"""

from typing import List, Dict, Any, Optional
from ._init_impl import _init_impl
from ._add_embeddings_impl import _add_embeddings_impl
from ._query_impl import _query_impl
from ._delete_impl import _delete_impl
from ._has_file_impl import _has_file_impl
from ._list_documents_impl import _list_metadata_impl, _search_metadata_impl
from ._get_all_tags_impl import _get_all_tags_impl

class ChromaVectorStore:
    """
    High-level interface for managing document embeddings and metadata in a ChromaDB-backed vector store.
    """
    def __init__(self, config: Any):
        """
        Initialize the vector store with the given configuration.
        Args:
            config: Configuration object for the vector store backend.
        """
        _init_impl(self, config)
        self._all_tags_cache = None

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add new embeddings, texts, and their associated metadata to the vector store.
        Args:
            embeddings: List of embedding vectors.
            texts: List of corresponding text strings.
            metadatas: List of metadata dictionaries for each document.
        """
        _add_embeddings_impl(self, embeddings, texts, metadatas)

    def query(self, query_vector: List[float], top_k: Optional[int] = None, filter_criteria: Optional[Dict[str, Any]] = None) -> Any:
        """
        Query the vector store for the most similar documents to the given query vector.
        Args:
            query_vector: The embedding vector to search for.
            top_k: Maximum number of results to return.
            filter_criteria: Optional metadata filter for narrowing results.
        Returns:
            Query results from the backend implementation.
        """
        return _query_impl(self, query_vector, top_k, filter_criteria)

    def delete(self, filter_criteria: Optional[Dict[str, Any]] = None, ids: Optional[list] = None, require_confirmation: bool = False):
        """
        Delete documents from the vector store by filter or IDs.
        Args:
            filter_criteria: Metadata filter for deletion.
            ids: List of document IDs to delete.
            require_confirmation: If True, require confirmation before deletion.
        """
        _delete_impl(self, filter_criteria, ids, require_confirmation)

    def has_file(self, file_hash: str) -> bool:
        """
        Check if a file with the given hash exists in the vector store.
        Args:
            file_hash: Hash of the file to check.
        Returns:
            True if the file exists, False otherwise.
        """
        return _has_file_impl(self, file_hash)

    def list_metadata(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all document metadata in the collection, optionally filtered by pattern (glob on path).
        Args:
            pattern: Optional glob pattern to filter by path.
        Returns:
            List of document metadata dictionaries.
        """
        return _list_metadata_impl(self, pattern)

    def get_all_tags(self) -> List[str]:
        """
        Retrieve all unique tags from the documents in the vector store.
        Returns:
            Sorted list of unique, lowercased tags across all documents.
        """
        return _get_all_tags_impl(self)

    def search_metadata(self, search_string: str, metadata_keys: Optional[List[str]] = None) -> List[str]:
        """
        Search for a string in specified metadata keys (or all keys if none provided),
        and return the contents of the 'path' metadata key for matches.

        Args:
            search_string: The string to search for (case-insensitive substring match).
            metadata_keys: List of metadata keys to search in. If None, search all keys.

        Returns:
            List of 'path' values for matching documents.
        """
        return _search_metadata_impl(self, search_string, metadata_keys)

__all__ = ["ChromaVectorStore"]
