"""
@file: _query_impl.py
Implementation of the vector store query logic for ChromaVectorStore.

This module provides the internal query implementation used to retrieve the most similar vectors from a Chroma collection, with support for filtering and error handling.
"""

from qa_system.exceptions import QueryError
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def _query_impl(
    self,
    query_vector: List[float],
    top_k: Optional[int] = None,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query the vector store for the most similar vectors to the given query vector.

    Args:
        self: The ChromaVectorStore instance (expects self.collection and self.top_k attributes).
        query_vector: The embedding vector to query against the collection.
        top_k: The maximum number of results to return. If None, uses self.top_k.
        filter_criteria: Optional dictionary specifying filter conditions for the query.

    Returns:
        Dictionary containing query results as returned by the underlying collection.

    Raises:
        QueryError: If the query fails for any reason.
    """
    logger.debug(
        "Called ChromaVectorStore.query(query_vector=<len %d>, top_k=%s, filter_criteria=%s)",
        len(query_vector), top_k, filter_criteria
    )
    try:
        k = top_k or self.top_k
        query_args = {
            'query_embeddings': [query_vector],
            'n_results': k
        }
        if filter_criteria is not None:
            query_args['where'] = filter_criteria
        results = self.collection.query(**query_args)
        logger.info(
            "Query returned %d results from collection '%s'",
            len(results['ids'][0]), self.collection_name
        )
        logger.debug("Query results: %r", results)
        return results
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise QueryError(f"Query failed: {e}") 