"""
@file: list.py
ListModule: Utilities for listing and summarizing document metadata in the vector store.

This module provides a high-level interface for listing document metadata, retrieving collection statistics,
and counting documents in a ChromaDB-backed vector store. It is intended for use in QA/document search systems.

Classes:
    ListModule: Main interface for listing and summarizing document metadata.

Functions:
    get_list_module(config=None): Factory for ListModule.
"""

from qa_system.vector_store import ChromaVectorStore
from qa_system.config import get_config
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ListModule:
    """
    Provides methods to list document metadata, get collection statistics, and count documents
    in the vector store. Optionally supports filtering by glob pattern on document path.
    """
    def __init__(self, config=None):
        """
        Initialize the ListModule with the given configuration.

        Args:
            config: Optional configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.store = ChromaVectorStore(self.config)
        # No debug/info logging at init

    def list_metadata(self, pattern: Optional[str] = None) -> List[Dict]:
        """
        Return a list of document metadata, optionally filtered by a glob pattern on the path.

        Args:
            pattern: Optional glob pattern (e.g., '*.md') to filter document paths. If None, returns all.

        Returns:
            List of unique document metadata dictionaries.
        """
        docs = self.store.list_metadata(pattern=pattern)
        seen = set()
        unique_docs = []
        for doc in docs:
            key = (doc.get('path'), doc.get('checksum'))
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        logger.debug(f"Listed {len(unique_docs)} unique document metadata entries (pattern={pattern!r})")
        return unique_docs

    def get_collection_stats(self) -> Dict:
        """
        Return statistics about the document collection, including total count and document types.

        Returns:
            Dictionary with 'total_documents' and 'document_types' (extension counts).
        """
        docs = self.list_metadata()
        types = {}
        for doc in docs:
            ext = doc.get('path', '').split('.')[-1] if 'path' in doc else 'unknown'
            types[ext] = types.get(ext, 0) + 1
        stats = {
            'total_documents': len(docs),
            'document_types': types
        }
        # No info/debug logging
        logger.debug(f"Collection stats: {stats}")
        return stats

    def get_document_count(self) -> int:
        """
        Return the total number of unique documents in the collection.

        Returns:
            Integer count of unique documents.
        """
        count = len(self.list_metadata())
        # No info/debug logging
        logger.debug(f"Document count: {count}")
        return count

def get_list_module(config=None) -> ListModule:
    """
    Factory function to create a ListModule instance with the given configuration.

    Args:
        config: Optional configuration object.

    Returns:
        ListModule instance.
    """
    return ListModule(config=config) 