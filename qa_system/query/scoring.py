import time
import fnmatch
import re
import difflib
from typing import List, Set, Optional, Any
from .models import Source


def deduplicate_sources(sources: List[Source], logger: Optional[Any] = None) -> List[Source]:
    """
    Deduplicate sources by document, keeping the one with the highest similarity.
    """
    seen = {}
    for src in sources:
        doc = src.document
        if doc not in seen or src.similarity > seen[doc].similarity:
            seen[doc] = src
    if logger:
        logger.debug(f"Deduplicated sources: {[s.document for s in seen.values()]}")
    return list(seen.values())


def extract_tag_matching_keywords(query: str, all_tags: List[str], stopwords: Set[str], logger: Optional[Any] = None, fuzzy_cutoff: float = 0.75) -> Set[str]:
    """
    Extract keywords from query and match them to tags using exact and fuzzy matching.
    """
    all_words = re.findall(r"\w+", query.lower())
    concept_keywords = [w for w in all_words if w not in stopwords]
    concept_to_tags = {}
    matched_tags = set()
    for concept in concept_keywords:
        exact_matches = [tag for tag in all_tags if tag == concept]
        fuzzy_matches = difflib.get_close_matches(concept, all_tags, n=5, cutoff=fuzzy_cutoff)
        all_matches = set(exact_matches + fuzzy_matches)
        if all_matches:
            concept_to_tags[concept] = sorted(all_matches)
            matched_tags.update(all_matches)
    if logger:
        logger.info(f"Concept keywords from query: {concept_keywords}")
        for concept, tags in concept_to_tags.items():
            logger.info(f"Tag matches for '{concept}': {tags}")
        logger.info(f"Final tag-matching keywords: {sorted(matched_tags)}")
    return matched_tags


def apply_scoring(processor, sources: List[Source]) -> List[Source]:
    """
    Applies scoring to sources using semantic similarity and metadata boosts.
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
        processor.logger.debug(f"Scoring: src.document={src.document}, preferred_sources={preferred_sources}, matched={matched}")
        if matched:
            processor.logger.debug(f"SOURCE BOOST APPLIED: {src.document} matched {preferred_sources}, boost={source_boost}")
            boost *= source_boost
        final_similarity = src.similarity * boost
        processor.logger.debug(f"Scoring: {src.document} | original={original_similarity:.4f} | clamped={src.similarity:.4f} | boost={boost:.2f} | final={final_similarity:.4f}")
        src.original_similarity = original_similarity
        src.boost = boost
        src.similarity = final_similarity
    return sorted(sources, key=lambda s: s.similarity, reverse=True) 