"""
@file: source_filter.py
Utilities for filtering sources by similarity and tag-matching in the QA system.

This module provides a function to filter a list of Source objects based on similarity thresholds and tag-matching criteria. It is used to select the most relevant sources (document chunks) for a given query, supporting both direct similarity and tag-based inclusion.

Functions:
    filter_sources: Filter sources by similarity and tag-matching keywords.
"""
from typing import List, Set, Tuple, Any
from .models import Source

def filter_sources(
    sources: List[Source],
    tag_matching_keywords: Set[str],
    min_similarity: float,
    tag_min_similarity: float,
    logger: Any = None
) -> Tuple[List[Source], List[Source]]:
    """
    Filter a list of Source objects by similarity and tag-matching.

    Args:
        sources (List[Source]): List of Source objects to filter.
        tag_matching_keywords (Set[str]): Set of keywords for tag matching (case-insensitive).
        min_similarity (float): Minimum similarity threshold for direct inclusion.
        tag_min_similarity (float): Minimum similarity for tag-matched sources.
        logger (Any, optional): Optional logger for debug output. Should support .info().

    Returns:
        Tuple[List[Source], List[Source]]: A tuple containing:
            - filtered_sources: List of sources meeting the similarity threshold or included via tag-matching.
            - tag_matched_sources: List of sources included specifically due to tag-matching.
    """
    filtered_sources: List[Source] = []
    tag_matched_sources: List[Source] = []
    for src in sources:
        tags = src.metadata.get('tags', [])
        # Support both comma-separated string and list for tags
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        tag_match = any(t.lower() in tag_matching_keywords for t in tags)
        if src.similarity >= min_similarity:
            filtered_sources.append(src)
        elif tag_match and src.similarity >= tag_min_similarity:
            tag_matched_sources.append(src)
    if logger and tag_matched_sources:
        logger.info(
            f"Sources included due to tag-matching: {[src.document for src in tag_matched_sources]}"
        )
    # Ensure tag-matched sources are included in filtered_sources
    for src in tag_matched_sources:
        if src not in filtered_sources:
            filtered_sources.append(src)
    return filtered_sources, tag_matched_sources 