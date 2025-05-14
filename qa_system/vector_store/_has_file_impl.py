import logging

logger = logging.getLogger(__name__)

def _has_file_impl(self, file_hash: str) -> bool:
    """Check if a file with the given hash exists in the collection."""
    try:
        logger.debug(f"Querying vector store for checksum: {file_hash}")
        results = self.collection.get(where={"checksum": file_hash}, limit=1)
        logger.debug(f"Vector store query result for checksum {file_hash}: {results}")
        return bool(results and results.get('ids'))
    except Exception as e:
        logger.error(f"Failed to check file existence in vector store: {e}")
        return False 