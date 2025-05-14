import time
import fnmatch
from typing import List
from .models import Source

def apply_hybrid_scoring(processor, sources: List[Source]) -> List[Source]:
    """
    Applies hybrid scoring to sources using semantic similarity and metadata boosts.
    Args:
        processor: QueryProcessor instance (for config and logger)
        sources: List of Source objects
    Returns:
        List[Source]: Scored and sorted sources
    """
    recency_boost = float(processor.config.get_nested('QUERY.RECENCY_BOOST', default=1.0))
    tag_boost = float(processor.config.get_nested('QUERY.TAG_BOOST', default=1.5))
    source_boost = float(processor.config.get_nested('QUERY.SOURCE_BOOST', default=1.0))
    now = time.time()
    preferred_sources = processor.config.get_nested('QUERY.PREFERRED_SOURCES', default=[])
    for src in sources:
        original_similarity = src.similarity
        src.similarity = max(0.0, min(1.0, src.similarity))
        boost = 1.0
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
                pass
        tags = src.metadata.get('tags', [])
        if tags and hasattr(processor, '_last_query_keywords'):
            if any(tag.lower() in processor._last_query_keywords for tag in tags):
                boost *= tag_boost
        matched = False
        for pref in preferred_sources:
            if fnmatch.fnmatch(src.document, pref):
                matched = True
                break
        processor.logger.debug(f"Hybrid scoring: src.document={src.document}, preferred_sources={preferred_sources}, matched={matched}")
        if matched:
            processor.logger.debug(f"SOURCE BOOST APPLIED: {src.document} matched {preferred_sources}, boost={source_boost}")
            boost *= source_boost
        final_similarity = src.similarity * boost
        processor.logger.debug(f"Scoring: {src.document} | original={original_similarity:.4f} | clamped={src.similarity:.4f} | boost={boost:.2f} | final={final_similarity:.4f}")
        src.original_similarity = original_similarity
        src.boost = boost
        src.similarity = final_similarity
    return sorted(sources, key=lambda s: s.similarity, reverse=True) 