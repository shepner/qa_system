from qa_system.exceptions import VectorStoreError
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _add_embeddings_impl(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]):
    logger.debug(f"Called ChromaVectorStore.add_embeddings(embeddings=<len {len(embeddings)}>, texts=<len {len(texts)}>, metadatas=<len {len(metadatas)}>)")
    try:
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