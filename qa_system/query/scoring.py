"""
@file: scoring.py
Scoring and deduplication utilities for query sources in the QA system.

This module provides functions to:
    - Deduplicate sources by document, keeping the one with the highest similarity.
    - Apply hybrid scoring to sources using semantic similarity and metadata-based boosts (recency, tag, and preferred source boosts).

All functions are designed to work with the Source model and are used in the query pipeline to rank and filter candidate sources for LLM context construction.
"""

import time
import fnmatch
from typing import List, Optional, Any
from .models import Source


def deduplicate_sources(sources: List[Source], logger: Optional[Any] = None) -> List[Source]:
    """
    Deduplicate sources by document, keeping only the one with the highest similarity for each document.

    Args:
        sources (List[Source]): List of Source objects to deduplicate.
        logger (Optional[Any]): Optional logger for debug output.

    Returns:
        List[Source]: Deduplicated list of Source objects (one per document).
    """
    seen = {}
    for src in sources:
        doc = src.document
        if doc not in seen or src.similarity > seen[doc].similarity:
            seen[doc] = src
    if logger:
        logger.debug(f"Deduplicated sources: {[s.document for s in seen.values()]}")
    return list(seen.values())


def apply_scoring(processor, sources: List[Source]) -> List[Source]:
    """
    Apply hybrid scoring to sources using semantic similarity and metadata-based boosts.

    Boosts applied:
        - Recency boost: If the source is less than 1 year old (based on 'date' in metadata).
        - Tag boost: If any tag in the source matches the last query keywords.
        - Source boost: If the document matches any preferred source pattern.

    Args:
        processor: QueryProcessor instance (must have .config and .logger attributes).
        sources (List[Source]): List of Source objects to score.

    Returns:
        List[Source]: Scored and sorted sources (highest similarity first).
    """
    recency_boost = float(processor.config.get_nested('QUERY.RECENCY_BOOST', default=1.0))
    tag_boost = float(processor.config.get_nested('QUERY.TAG_BOOST', default=1.5))
    source_boost = float(processor.config.get_nested('QUERY.SOURCE_BOOST', default=1.0))
    now = time.time()
    preferred_sources = processor.config.get_nested('QUERY.PREFERRED_SOURCES', default=[])

    for src in sources:
        original_similarity = src.similarity
        # Clamp similarity to [0, 1]
        src.similarity = max(0.0, min(1.0, src.similarity))
        boost = 1.0

        # Recency boost: if 'date' metadata is present and less than 1 year old
        date = src.metadata.get('date')
        if date:
            try:
                if isinstance(date, (int, float)):
                    age_days = (now - float(date)) / 86400
                else:
                    from dateutil.parser import parse
                    dt = parse(date)
                    age_days = (now - dt.timestamp()) / 86400
                if age_days < 365:
                    boost *= recency_boost
            except Exception:
                pass  # Ignore date parsing errors

        # Tag boost: if any tag matches the last query keywords
        tags = src.metadata.get('tags', [])
        if tags and hasattr(processor, '_last_query_keywords'):
            if any(tag.lower() in processor._last_query_keywords for tag in tags):
                boost *= tag_boost

        # Source boost: if document matches any preferred source pattern
        matched = any(fnmatch.fnmatch(src.document, pref) for pref in preferred_sources)
        processor.logger.debug(f"Scoring: src.document={src.document}, preferred_sources={preferred_sources}, matched={matched}")
        if matched:
            processor.logger.debug(f"SOURCE BOOST APPLIED: {src.document} matched {preferred_sources}, boost={source_boost}")
            boost *= source_boost

        final_similarity = src.similarity * boost
        processor.logger.debug(
            f"Scoring: {src.document} | original={original_similarity:.4f} | clamped={src.similarity:.4f} | boost={boost:.2f} | final={final_similarity:.4f}"
        )
        src.original_similarity = original_similarity
        src.boost = boost
        src.similarity = final_similarity

    return sorted(sources, key=lambda s: s.similarity, reverse=True) 