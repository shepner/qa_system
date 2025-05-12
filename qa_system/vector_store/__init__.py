import chromadb
from qa_system.exceptions import VectorStoreError, ConnectionError, QueryError, ValidationError
import logging
from typing import List, Dict, Any, Optional
import fnmatch

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, config):
        logger.info(f"Called ChromaVectorStore.__init__(config={config})")
        try:
            vector_config = config.get_nested('VECTOR_STORE')
            self.persist_directory = vector_config.get('PERSIST_DIRECTORY', './data/vector_store')
            self.collection_name = vector_config.get('COLLECTION_NAME', 'qa_documents')
            self.distance_metric = vector_config.get('DISTANCE_METRIC', 'cosine')
            self.top_k = vector_config.get('TOP_K', 40)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Try to get or create the collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name, metadata={"hnsw:space": self.distance_metric})
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
            logger.debug(f"Query results: {results}")
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(f"Query failed: {e}")

    def delete(self, filter_criteria: Dict[str, Any] = None, ids: list = None, require_confirmation: bool = False):
        logger.info(f"Called ChromaVectorStore.delete(filter_criteria={filter_criteria}, ids={ids}, require_confirmation={require_confirmation})")
        logger.debug(f"Called ChromaVectorStore.delete(filter_criteria={filter_criteria}, ids={ids}, require_confirmation={require_confirmation})")
        try:
            if require_confirmation:
                logger.info(f"Confirmation required for deletion: ids={ids}, filter_criteria={filter_criteria}")
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted documents with ids: {ids}")
            elif filter_criteria:
                self.collection.delete(where=filter_criteria)
                logger.info(f"Deleted documents matching: {filter_criteria}")
            else:
                raise ValueError("Must provide either ids or filter_criteria for deletion.")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise VectorStoreError(f"Failed to delete documents: {e}")

    def has_file(self, file_hash: str) -> bool:
        """Check if a file with the given hash exists in the collection."""
        try:
            logger.debug(f"Querying vector store for checksum: {file_hash}")
            results = self.collection.get(where={"checksum": file_hash}, limit=1)
            logger.debug(f"Vector store query result for checksum {file_hash}: {results}")
            return bool(results and results.get('ids'))
        except Exception as e:
            logger.error(f"Failed to check file existence in vector store: {e}")
            return False

    def list_documents(self, pattern: Optional[str] = None) -> list[dict]:
        """List all document metadata in the collection, optionally filtered by pattern (glob on path)."""
        try:
            logger.debug(f"Listing documents in collection '{self.collection_name}' with pattern: {pattern}")
            # ChromaDB get() with no filter returns all
            results = self.collection.get()
            metadatas = results.get('metadatas', [])
            if pattern:
                filtered = [m for m in metadatas if 'path' in m and fnmatch.fnmatch(m['path'], pattern)]
                logger.debug(f"Filtered {len(filtered)} documents matching pattern '{pattern}' out of {len(metadatas)} total.")
                return filtered
            logger.debug(f"Returning {len(metadatas)} document metadatas.")
            return metadatas
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise VectorStoreError(f"Failed to list documents: {e}")
