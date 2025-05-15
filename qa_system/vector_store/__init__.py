from typing import List, Dict, Any, Optional
from ._init_impl import _init_impl
from ._add_embeddings_impl import _add_embeddings_impl
from ._query_impl import _query_impl
from ._delete_impl import _delete_impl
from ._has_file_impl import _has_file_impl
from ._list_documents_impl import _list_metadata_impl, _search_metadata_impl
from ._get_all_tags_impl import _get_all_tags_impl

class ChromaVectorStore:
    def __init__(self, config):
        _init_impl(self, config)
        self._all_tags_cache = None

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]):
        _add_embeddings_impl(self, embeddings, texts, metadatas)

    def query(self, query_vector: List[float], top_k: Optional[int] = None, filter_criteria: Optional[Dict[str, Any]] = None):
        return _query_impl(self, query_vector, top_k, filter_criteria)

    def delete(self, filter_criteria: Dict[str, Any] = None, ids: list = None, require_confirmation: bool = False):
        _delete_impl(self, filter_criteria, ids, require_confirmation)

    def has_file(self, file_hash: str) -> bool:
        return _has_file_impl(self, file_hash)

    def list_metadata(self, pattern: Optional[str] = None) -> list[dict]:
        """List all document metadata in the collection, optionally filtered by pattern (glob on path)."""
        return _list_metadata_impl(self, pattern)

    def get_all_tags(self):
        return _get_all_tags_impl(self)

    def list_metadata_by_tag_or_keyword(self, value: str) -> list[dict]:
        """
        List all document metadata where the tag, path, filename_stem, or url matches the given value (case-insensitive).
        Args:
            value (str): Tag or keyword to match (case-insensitive).
        Returns:
            list[dict]: List of document metadata dicts matching the tag or keyword.
        """
        import logging
        logger = logging.getLogger(__name__)
        value_norm = value.strip().lower()
        # Try to use backend filtering if possible
        try:
            # Attempt to use the vector store's filtering capabilities
            filter_criteria = {"tags": value_norm}
            logger.debug(f"Attempting fast tag filter with filter_criteria={filter_criteria}")
            results = self.collection.get(where=filter_criteria)
            metadatas = results.get('metadatas', [])
            logger.info(f"[FAST TAG FILTER] Tag '{value_norm}' matched {len(metadatas)} documents using backend filter.")
            return metadatas
        except Exception as e:
            logger.warning(f"[FALLBACK] Backend tag filter failed for tag '{value_norm}': {e}. Falling back to in-memory filtering.")
        # Fallback: in-memory filtering
        matches = []
        all_metas = self.list_metadata()
        for meta in all_metas:
            tags = meta.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',') if t.strip()]
            if any(t.lower() == value_norm for t in tags):
                matches.append(meta)
                continue
            # Path, filename_stem, url matching (substring, case-insensitive)
            for field in ('path', 'filename_stem', 'url'):
                field_val = meta.get(field, '')
                if isinstance(field_val, str) and value_norm in field_val.lower():
                    matches.append(meta)
                    break
        logger.info(f"[FALLBACK] Tag/keyword '{value_norm}' matched {len(matches)} documents using in-memory filtering (total scanned: {len(all_metas)}).");
        return matches

    def search_metadata(self, search_string: str, metadata_keys: Optional[list[str]] = None) -> list[str]:
        """
        Search for a string in specified metadata keys (or all keys if none provided),
        and return the contents of the 'path' metadata key for matches.

        Args:
            search_string (str): The string to search for (case-insensitive substring match).
            metadata_keys (Optional[list[str]]): List of metadata keys to search in. If None, search all keys.

        Returns:
            list[str]: List of 'path' values for matching documents.
        """
        return _search_metadata_impl(self, search_string, metadata_keys)

__all__ = ["ChromaVectorStore"]
