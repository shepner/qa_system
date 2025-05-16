"""
@file: _init_impl.py
Internal implementation for initializing the Chroma vector store backend.

This module provides the _init_impl method, which sets up the persistent ChromaDB client and collection
for storing and retrieving document embeddings. It is intended for use as part of the ChromaVectorStore class.

Configuration:
    - VECTOR_STORE.PERSIST_DIRECTORY: Directory for persistent storage (default: './data/vector_store')
    - VECTOR_STORE.COLLECTION_NAME: Name of the collection (default: 'qa_documents')
    - VECTOR_STORE.DISTANCE_METRIC: Distance metric for similarity search (default: 'cosine')
    - VECTOR_STORE.TOP_K: Default number of top results to return (default: 40)

Raises:
    ConnectionError: If initialization fails (e.g., cannot connect to or create the collection)
"""

import chromadb
import logging
from qa_system.exceptions import ConnectionError

logger = logging.getLogger(__name__)

def _init_impl(self, config):
    """
    Initialize the Chroma vector store client and collection.

    Args:
        self: The ChromaVectorStore instance.
        config: Configuration object with get_nested method for retrieving settings.

    Sets the following attributes on self:
        - persist_directory: Directory for persistent storage
        - collection_name: Name of the collection
        - distance_metric: Distance metric for similarity search
        - top_k: Default number of top results to return
        - client: ChromaDB PersistentClient instance
        - collection: ChromaDB collection instance

    Raises:
        ConnectionError: If initialization fails (e.g., cannot connect to or create the collection)
    """
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
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaVectorStore: {e}")
        raise ConnectionError(f"Failed to initialize vector store: {e}") 