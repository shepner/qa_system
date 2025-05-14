from qa_system.exceptions import VectorStoreError
import logging
from typing import Optional
import fnmatch

logger = logging.getLogger(__name__)

def _list_metadata_impl(self, pattern: Optional[str] = None) -> list[dict]:
    """List all document metadata in the collection, optionally filtered by pattern (glob on path)."""
    try:
        logger.debug(f"Listing documents in collection '{self.collection_name}' with pattern: {pattern}")
        # ChromaDB get() with no filter returns all
        results = self.collection.get()
        metadatas = results.get('metadatas', [])
        if pattern:
            filtered = [m for m in metadatas if 'path' in m and fnmatch.fnmatch(m['path'], pattern)]
            logger.debug(f"Filtered {len(filtered)} document metadatas matching pattern '{pattern}' out of {len(metadatas)} total.")
            return filtered
        logger.debug(f"Returning {len(metadatas)} document metadatas.")
        return metadatas
    except Exception as e:
        logger.error(f"Failed to list document metadatas: {e}")
        raise VectorStoreError(f"Failed to list document metadatas: {e}") 