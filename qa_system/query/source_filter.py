"""
source_filter.py
Utilities for filtering sources by similarity and tag-matching.
"""
from typing import List, Set, Any
from .models import Source

def filter_sources(
    sources: List[Source],
    tag_matching_keywords: Set[str],
    min_similarity: float,
    tag_min_similarity: float,
    logger: Any = None
) -> (List[Source], List[Source]):
    """
    Filter sources by similarity and tag-matching.
    Args:
        sources: List of Source objects
        tag_matching_keywords: Set of keywords for tag matching
        min_similarity: Minimum similarity threshold
        tag_min_similarity: Minimum similarity for tag-matched sources
        logger: Optional logger for debug output
    Returns:
        (filtered_sources, tag_matched_sources): Tuple of lists
    """
    filtered_sources = []
    tag_matched_sources = []
    for src in sources:
        tags = src.metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        tag_match = any(t.lower() in tag_matching_keywords for t in tags)
        if src.similarity >= min_similarity:
            filtered_sources.append(src)
        elif tag_match and src.similarity >= tag_min_similarity:
            tag_matched_sources.append(src)
    if logger:
        logger.info(f"Sources included due to tag-matching: {[src.document for src in tag_matched_sources]}")
    # Ensure tag-matched sources are included in filtered_sources
    for src in tag_matched_sources:
        if src not in filtered_sources:
            filtered_sources.append(src)
    return filtered_sources, tag_matched_sources 