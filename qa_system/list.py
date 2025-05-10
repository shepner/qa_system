from qa_system.vector_store import ChromaVectorStore
from qa_system.config import get_config
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ListModule:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.store = ChromaVectorStore(self.config)
        logger.debug(f"Initialized ListModule with config: {self.config}")

    def list_documents(self, pattern: Optional[str] = None) -> List[Dict]:
        """Return list of document metadata, optionally filtered by glob pattern. Also print full metadata for each document."""
        docs = self.store.list_documents(pattern=pattern)
        logger.info(f"Listing {len(docs)} documents (pattern={pattern})")
        for i, doc in enumerate(docs, 1):
            print(f"Document {i} metadata: {doc}")
        return docs

    def get_collection_stats(self) -> Dict:
        """Return statistics about the document collection."""
        docs = self.store.list_documents()
        types = {}
        for doc in docs:
            ext = doc.get('path', '').split('.')[-1] if 'path' in doc else 'unknown'
            types[ext] = types.get(ext, 0) + 1
        stats = {
            'total_documents': len(docs),
            'document_types': types
        }
        logger.info(f"Collection stats: {stats}")
        return stats

    def get_document_count(self) -> int:
        """Return total number of documents in the collection."""
        count = len(self.store.list_documents())
        logger.info(f"Document count: {count}")
        return count

def get_list_module(config=None):
    return ListModule(config=config) 