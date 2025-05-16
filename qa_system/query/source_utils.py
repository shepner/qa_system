"""
@file: source_utils.py
Utilities for constructing and normalizing Source objects from vector store results.

This module provides functions to transform raw vector store query results into standardized
Source objects for downstream use in the QA system. It ensures consistent path handling,
similarity normalization, and context extraction for each source.
"""
from typing import List, Dict, Any
import os
import logging
from .models import Source


def build_sources_from_vector_results(
    ids: List[str],
    docs: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[float],
    docs_root: str,
    context_length: int = 200
) -> List[Source]:
    """
    Construct Source objects from vector store query results.

    Args:
        ids (List[str]): List of document IDs.
        docs (List[str]): List of document chunks (text excerpts).
        metadatas (List[Dict[str, Any]]): List of metadata dictionaries for each chunk.
        distances (List[float]): List of distance/similarity floats (lower is more similar).
        docs_root (str): Root directory for document paths (used for relative path calculation).
        context_length (int, optional): Number of characters to use for context. Defaults to 200.

    Returns:
        List[Source]: List of constructed Source objects, each representing a document chunk.
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        "build_sources_from_vector_results: lengths - ids: %d, docs: %d, metadatas: %d, distances: %d",
        len(ids), len(docs), len(metadatas), len(distances)
    )
    logger.debug("Sample ids: %s", ids[:5])
    logger.debug("Sample docs: %s", docs[:1])
    logger.debug("Sample metadatas: %s", metadatas[:1])
    logger.debug("Sample distances: %s", distances[:5])

    sources = []
    docs_root = os.path.abspath(docs_root)

    for i, doc_id in enumerate(ids):
        doc = docs[i] if i < len(docs) else ''
        meta = metadatas[i] if i < len(metadatas) else {}
        sim = 1.0 - distances[i] if i < len(distances) else 0.0
        context = doc[:context_length]
        abs_path = os.path.abspath(meta.get('path', doc_id))
        if abs_path.startswith(docs_root):
            rel_path = os.path.relpath(abs_path, docs_root)
        else:
            rel_path = abs_path
        clamped_sim = max(0.0, min(1.0, sim))
        sources.append(Source(
            document=rel_path,
            chunk=doc,
            similarity=clamped_sim,
            context=context,
            metadata=meta,
            original_similarity=sim
        ))
    return sources 