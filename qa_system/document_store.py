"""Document storage and retrieval functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
import fnmatch
import os

class DocumentStore:
    """Handles storage and retrieval of documents."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the document store.
        
        Args:
            config: Configuration dictionary that may contain:
                - 'VECTOR_STORE.PERSIST_DIRECTORY': Directory for file storage
                - 'mongodb_uri': MongoDB connection URI
                - 'mongodb_db': MongoDB database name
                - 'mongodb_collection': MongoDB collection name
        """
        # File storage setup
        vector_store_config = config.get('VECTOR_STORE', {})
        data_dir = vector_store_config.get('PERSIST_DIRECTORY', './data/vector_store')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # MongoDB setup
        mongodb_uri = config.get('mongodb_uri', 'mongodb://localhost:27017')
        mongodb_db = config.get('mongodb_db', 'qa_system')
        mongodb_collection = config.get('mongodb_collection', 'documents')
        
        # Get exclude patterns from config
        doc_config = config.get("DOCUMENT_PROCESSING", {})
        self.exclude_patterns = doc_config.get("EXCLUDE_PATTERNS", [
            "*.git/*",
            "*__pycache__/*",
            "*.pyc",
            "*.env",
            "*.DS_Store"
        ])
        
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client[mongodb_db]
        self.collection: AsyncIOMotorCollection = self.db[mongodb_collection]
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DocumentStore at {self.data_dir} with MongoDB collection {mongodb_collection}")

    def should_exclude(self, filepath: str) -> bool:
        """Check if a file should be excluded based on configured patterns.
        
        Args:
            filepath: Path to the file to check
            
        Returns:
            bool: True if the file should be excluded, False otherwise
        """
        try:
            # Get relative path if absolute path provided
            rel_path = os.path.relpath(filepath) if os.path.isabs(filepath) else filepath
            
            # Check if file matches any exclude pattern
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatch(rel_path, pattern):
                    self.logger.debug(f"Excluding file {rel_path} matching pattern {pattern}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking file exclusion for {filepath}: {str(e)}")
            # If there's an error checking, exclude the file to be safe
            return True

    async def store_document(self, content: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a document in the document store.
        
        Args:
            content: Document content to store
            doc_id: Unique identifier for the document
            metadata: Optional metadata to store with the document
            
        Returns:
            bool: True if storage was successful
        """
        try:
            # Store in file system
            doc_path = self.data_dir / f"{doc_id}.txt"
            doc_path.write_text(content)
            
            # Store in MongoDB
            if metadata is None:
                metadata = {}
            metadata['doc_id'] = doc_id
            await self.collection.update_one(
                {'metadata.doc_id': doc_id},
                {'$set': {'content': content, 'metadata': metadata}},
                upsert=True
            )
            
            self.logger.debug(f"Stored document {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store document {doc_id}: {str(e)}")
            return False

    async def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document from the store.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Optional[str]: Document content if found, None otherwise
        """
        try:
            doc_path = self.data_dir / f"{doc_id}.txt"
            if doc_path.exists():
                content = doc_path.read_text()
                self.logger.debug(f"Retrieved document {doc_id}")
                return content
            self.logger.warning(f"Document {doc_id} not found")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve document {doc_id}: {str(e)}")
            return None

    async def list_documents(self) -> List[str]:
        """List all document IDs in the store.
        
        Returns:
            List[str]: List of document IDs
        """
        try:
            return [p.stem for p in self.data_dir.glob("*.txt")]
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            doc_path = self.data_dir / f"{doc_id}.txt"
            if doc_path.exists():
                doc_path.unlink()
                self.logger.debug(f"Deleted document {doc_id}")
                return True
            self.logger.warning(f"Document {doc_id} not found for deletion")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    async def remove_excluded_documents(self) -> None:
        """Remove documents that have been marked as excluded from the index."""
        excluded_docs = await self.get_excluded_documents()
        for doc_id in excluded_docs:
            await self.remove_document(doc_id)

    async def get_excluded_documents(self) -> List[str]:
        """Get a list of excluded document IDs.
        
        Returns:
            List[str]: List of document IDs that are marked as excluded
        """
        try:
            excluded_docs = []
            cursor = self.collection.find({'metadata.excluded': True})
            async for doc in cursor:
                if doc.get('metadata', {}).get('doc_id'):
                    excluded_docs.append(doc['metadata']['doc_id'])
            return excluded_docs
        except Exception as e:
            self.logger.error(f"Failed to get excluded documents: {str(e)}")
            return []

    async def mark_document_excluded(self, doc_id: str) -> bool:
        """Mark a document as excluded.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            bool: True if marking was successful
        """
        try:
            result = await self.collection.update_one(
                {'metadata.doc_id': doc_id},
                {'$set': {'metadata.excluded': True}}
            )
            success = result.modified_count > 0 or result.upserted_id is not None
            if success:
                self.logger.debug(f"Marked document {doc_id} as excluded")
            else:
                self.logger.warning(f"Document {doc_id} not found for marking as excluded")
            return success
        except Exception as e:
            self.logger.error(f"Failed to mark document {doc_id} as excluded: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if hasattr(self, 'client'):
                await self.client.close()
                self.logger.info("Closed MongoDB connection")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 