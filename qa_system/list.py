from qa_system.vector_store import ChromaVectorStore
from qa_system.config import get_config
import logging
from typing import Optional, List, Dict
from pprint import pprint

logger = logging.getLogger(__name__)

class ListModule:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.store = ChromaVectorStore(self.config)
        # No debug/info logging at init

    def list_documents(self, pattern: Optional[str] = None) -> List[Dict]:
        """Return list of document metadata, optionally filtered by glob pattern. Print each unique metadata dict nicely formatted."""
        docs = self.store.list_documents(pattern=pattern)
        seen = set()
        for doc in docs:
            key = (doc.get('path'), doc.get('hash'))
            if key in seen:
                continue
            seen.add(key)
            pprint(doc)
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
        # No info/debug logging
        return stats

    def get_document_count(self) -> int:
        """Return total number of documents in the collection."""
        count = len(self.store.list_documents())
        # No info/debug logging
        return count

def get_list_module(config=None):
    return ListModule(config=config) 