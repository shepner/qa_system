import chromadb
from qa_system.exceptions import ConnectionError
import logging

logger = logging.getLogger(__name__)

def _init_impl(self, config):
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