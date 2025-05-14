from qa_system.exceptions import QueryError
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def _query_impl(self, query_vector: List[float], top_k: Optional[int] = None, filter_criteria: Optional[Dict[str, Any]] = None):
    logger.debug(f"Called ChromaVectorStore.query(query_vector=<len {len(query_vector)}>, top_k={top_k}, filter_criteria={filter_criteria})")
    try:
        k = top_k or self.top_k
        query_args = {
            'query_embeddings': [query_vector],
            'n_results': k
        }
        if filter_criteria is not None:
            query_args['where'] = filter_criteria
        results = self.collection.query(**query_args)
        logger.info(f"Query returned {len(results['ids'][0])} results from collection '{self.collection_name}'")
        logger.debug(f"Query results: {results}")
        return results
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise QueryError(f"Query failed: {e}") 