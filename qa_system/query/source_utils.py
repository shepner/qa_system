"""
source_utils.py
Utilities for constructing and normalizing Source objects from vector store results.
"""
from typing import List, Dict, Any
import os
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
        ids: List of document IDs
        docs: List of document chunks
        metadatas: List of metadata dicts
        distances: List of distance floats
        docs_root: Root directory for document paths
        context_length: Number of characters to use for context
    Returns:
        List[Source]: List of constructed Source objects
    """
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