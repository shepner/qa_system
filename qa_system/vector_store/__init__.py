from typing import List, Dict, Any, Optional
from ._init_impl import _init_impl
from ._add_embeddings_impl import _add_embeddings_impl
from ._query_impl import _query_impl
from ._delete_impl import _delete_impl
from ._has_file_impl import _has_file_impl
from ._list_documents_impl import _list_metadata_impl
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

__all__ = ["ChromaVectorStore"]
