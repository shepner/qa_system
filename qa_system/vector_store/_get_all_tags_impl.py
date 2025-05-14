"""
@file: _get_all_tags_impl.py
Implementation for retrieving all unique tags from the vector store documents.

This module provides an internal method to extract, deduplicate, and normalize tags
across all documents managed by the vector store. Tags are returned as a sorted list
of unique, lowercased values for consistency and ease of use.
"""

def _get_all_tags_impl(self):
    """
    Retrieve all unique tags from the documents in the vector store.

    This method collects tags from all documents, normalizes them to lowercase,
    deduplicates them, and returns a sorted list. Tags can be provided as a list or
    a comma-separated string in the document metadata. The result is cached for
    performance.

    Returns:
        list[str]: Sorted list of unique, lowercased tags across all documents.
    """
    if self._all_tags_cache is not None:
        return self._all_tags_cache
    all_tags = set()
    for meta in self.list_metadata():
        tags = meta.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        all_tags.update(t.lower() for t in tags)
    result = sorted(all_tags)
    self._all_tags_cache = result
    return result 