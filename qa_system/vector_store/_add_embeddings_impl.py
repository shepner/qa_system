"""
@file: _add_embeddings_impl.py
Implementation for adding embeddings to a vector store collection.

This module provides the internal implementation for adding embedding vectors, their associated texts, and metadata to a vector store collection. It includes duplicate ID detection and robust error handling.
"""
from qa_system.exceptions import VectorStoreError
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _add_embeddings_impl(
    self,
    embeddings: List[List[float]],
    texts: List[str],
    metadatas: List[Dict[str, Any]]
) -> None:
    """
    Add embedding vectors, texts, and metadata to the vector store collection.

    Args:
        self: The instance of the vector store (expects self.collection and self.collection_name).
        embeddings: List of embedding vectors (each a list of floats).
        texts: List of text documents corresponding to the embeddings.
        metadatas: List of metadata dictionaries for each embedding/text.

    Raises:
        VectorStoreError: If adding embeddings to the collection fails.

    Logging:
        - Logs the number of embeddings, texts, and metadatas being added.
        - Warns if duplicate IDs are detected in the input.
        - Logs success or failure of the operation.
    """
    logger.debug(
        f"Called ChromaVectorStore.add_embeddings(embeddings=<len {len(embeddings)}>, "
        f"texts=<len {len(texts)}>, metadatas=<len {len(metadatas)}>)"
    )
    try:
        # Generate IDs for each embedding, preferring 'id', then 'path', then index as fallback
        ids = [meta.get('id') or meta.get('path') or str(i) for i, meta in enumerate(metadatas)]
        logger.debug(f"Embedding IDs to add: {ids}")

        # Check for duplicate IDs
        seen = set()
        duplicates = set()
        for id_ in ids:
            if id_ in seen:
                duplicates.add(id_)
            seen.add(id_)
        if duplicates:
            logger.warning(f"Duplicate IDs detected in embeddings: {duplicates}")

        # Add embeddings, texts, and metadata to the collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(embeddings)} embeddings to collection '{self.collection_name}'")
    except Exception as e:
        logger.error(f"Failed to add embeddings: {e}")
        raise VectorStoreError(f"Failed to add embeddings: {e}") 