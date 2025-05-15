from qa_system.exceptions import VectorStoreError
import logging
from typing import Optional
import fnmatch
import inspect

logger = logging.getLogger(__name__)

def _list_metadata_impl(self, pattern: Optional[str] = None) -> list[dict]:
    """List all document metadata in the collection, optionally filtered by pattern (glob on path)."""
    try:
        caller = inspect.stack()[1].function
        logger.debug(f"Listing documents in collection '{self.collection_name}' with path filter pattern: {pattern} (called from: {caller})")
        # Optionally, log a short stack trace for deeper debugging
        # logger.debug('Call stack:\n%s', ''.join(traceback.format_stack(limit=5)))
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

def _search_metadata_impl(self, search_string: str, metadata_keys: Optional[list[str]] = None) -> list[str]:
    """
    Search for a string in specified metadata keys (or all keys if none provided),
    and return the document IDs (from the 'id' metadata key) for matches.

    Args:
        search_string (str): The string to search for (case-insensitive substring match, except for tags which are exact).
        metadata_keys (Optional[list[str]]): List of metadata keys to search in. If None, search all keys.

    Returns:
        list[str]: List of document IDs ('id' values) for matching documents. Only includes documents with an 'id'.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.debug(f"Searching metadata for string '{search_string}' in keys: {metadata_keys}")
        all_metas = self.list_metadata()
        search_lower = search_string.strip().lower()
        results = []
        for meta in all_metas:
            keys_to_search = metadata_keys if metadata_keys is not None else list(meta.keys())
            for key in keys_to_search:
                value = meta.get(key)
                if key == 'tags' and isinstance(value, str):
                    tags = [t.strip().lower() for t in value.split(',') if t.strip()]
                    if search_lower in tags:
                        doc_id = meta.get('id')
                        if doc_id is not None:
                            results.append(doc_id)
                        break  # Only add each document once
                elif isinstance(value, str) and search_lower in value.lower():
                    doc_id = meta.get('id')
                    if doc_id is not None:
                        results.append(doc_id)
                    break  # Only add each document once
        logger.info(f"Search for '{search_string}' in keys {metadata_keys} matched {len(results)} documents.")
        return results
    except Exception as e:
        logger.error(f"Failed to search metadata: {e}")
        raise VectorStoreError(f"Failed to search metadata: {e}") 