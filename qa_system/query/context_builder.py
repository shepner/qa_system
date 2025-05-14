"""
context_builder.py
Utilities for building the context window for LLM queries from filtered sources.
"""
from typing import List, Set
from .models import Source

def build_context_window(
    sources: List[Source],
    context_window: int = 4096
) -> (str, int, Set[str]):
    """
    Build the context text for the LLM, respecting the context window (token/word limit).
    Args:
        sources: List of Source objects (filtered and deduped)
        context_window: Maximum number of tokens/words to include
    Returns:
        context_text: The constructed context string
        tokens_used: Number of tokens/words used
        context_docs: Set of document names included
    """
    context_text = ""
    tokens_used = 0
    used_docs = set()
    context_chunks = []
    for src in sources:
        if src.document in used_docs:
            continue
        chunk = src.chunk
        tokens_used += len(chunk.split())
        if tokens_used > context_window:
            break
        context_text += f"\n---\n{chunk}"
        context_chunks.append(src.document)
        used_docs.add(src.document)
    return context_text, tokens_used, set(context_chunks) 