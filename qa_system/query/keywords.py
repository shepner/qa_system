"""
keywords.py
Utilities for extracting keywords from queries, with stopword removal.
"""
import re
from typing import Set, List

def extract_keywords(query: str, stopwords: Set[str]) -> Set[str]:
    """
    Extract keywords from a query string, removing stopwords.
    Args:
        query: The input query string
        stopwords: Set of stopwords to remove
    Returns:
        Set[str]: Set of keywords (lowercased, alphanumeric)
    """
    all_words = re.findall(r"\w+", query.lower())
    return set(w for w in all_words if w not in stopwords) 