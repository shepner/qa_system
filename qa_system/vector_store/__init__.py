import chromadb
from chromadb.config import Settings
from qa_system.exceptions import VectorStoreError, ConnectionError, QueryError, ValidationError
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, config):
        logger.debug(f"Called ChromaVectorStore.__init__(config={config})")
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

    def query(self, query_vector: List[float], top_k: Optional[int] = None, filter_criteria: Optional[Dict[str, Any]] = None):
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
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(f"Query failed: {e}")

    def delete(self, filter_criteria: Dict[str, Any], require_confirmation: bool = False):
        logger.debug(f"Called ChromaVectorStore.delete(filter_criteria={filter_criteria}, require_confirmation={require_confirmation})")
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

    def has_file(self, file_hash: str) -> bool:
        """Check if a file with the given hash exists in the collection."""
        try:
            # Query for any document with this hash in metadata
            results = self.collection.get(where={"hash": file_hash}, limit=1)
            # Chroma returns a dict with 'ids' key
            return bool(results and results.get('ids'))
        except Exception as e:
            logger.error(f"Failed to check file existence in vector store: {e}")
            return False
