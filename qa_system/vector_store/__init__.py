import chromadb
from chromadb.config import Settings
from qa_system.exceptions import VectorStoreError, ConnectionError, QueryError, ValidationError
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, config):
        try:
            vector_config = config.get_nested('VECTOR_STORE')
            self.persist_directory = vector_config.get('PERSIST_DIRECTORY', './data/vector_store')
            self.collection_name = vector_config.get('COLLECTION_NAME', 'qa_documents')
            self.distance_metric = vector_config.get('DISTANCE_METRIC', 'cosine')
            self.top_k = vector_config.get('TOP_K', 40)
            self.client = chromadb.Client(Settings(
                persist_directory=self.persist_directory
            ))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaVectorStore: {e}")
            raise ConnectionError(f"Failed to initialize vector store: {e}")

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]):
        try:
            ids = [meta.get('id') or meta.get('path') or str(i) for i, meta in enumerate(metadatas)]
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

    def query(self, query_vector: List[float], top_k: Optional[int] = None, filter_criteria: Optional[Dict[str, Any]] = None):
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
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(f"Query failed: {e}")

    def delete(self, filter_criteria: Dict[str, Any], require_confirmation: bool = False):
        try:
            # ChromaDB supports deletion by ids or metadata filter
            if require_confirmation:
                # In a real CLI, prompt user; here, just log
                logger.info(f"Confirmation required for deletion: {filter_criteria}")
            self.collection.delete(where=filter_criteria)
            logger.info(f"Deleted documents matching: {filter_criteria}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise VectorStoreError(f"Failed to delete documents: {e}")
