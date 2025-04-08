#!/usr/bin/env python3

import fnmatch
import logging
import os
from pathlib import Path
from qa_system.core import QASystem
from qa_system.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def remove_excluded_documents(config_path: str = None):
    """Remove documents from vector store that match exclude patterns in config.
    
    Args:
        config_path: Optional path to config file
    """
    # Use config.yaml from root directory by default
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    
    # Initialize QA system and get config
    qa = QASystem(config_path)
    
    # Get list of all documents in vector store
    docs = qa.list_documents()
    logger.info(f"Found {len(docs)} documents in vector store")
    
    # Get exclude patterns from config
    exclude_patterns = qa.config.get("EXCLUDE_PATTERNS", [])
    base_dir = Path(qa.config.get("BASE_DIR", "."))
    logger.info(f"Using base directory: {base_dir}")
    logger.info(f"Exclude patterns: {exclude_patterns}")
    
    removed = 0
    for doc in docs:
        file_path = Path(doc["filename"])
        logger.debug(f"Processing document: {file_path}")
        
        # Get relative path for pattern matching
        try:
            relative_path = str(file_path.resolve().relative_to(base_dir.resolve()))
            logger.debug(f"Relative path: {relative_path}")
        except ValueError as e:
            logger.warning(f"Could not make path relative to base dir: {e}")
            relative_path = str(file_path)
            logger.debug(f"Using absolute path: {relative_path}")
            
        # Check if path matches any exclude pattern
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                logger.info(f"Removing {file_path} - matches exclude pattern '{pattern}'")
                try:
                    success = await qa.remove_document(doc["id"])
                    if success:
                        removed += 1
                        logger.info(f"Successfully removed document {doc['id']}")
                    else:
                        logger.error(f"Failed to remove document {doc['id']}")
                except Exception as e:
                    logger.error(f"Error removing document {doc['id']}: {str(e)}")
                break
                
    logger.info(f"Removed {removed} documents matching exclude patterns")

if __name__ == "__main__":
    import asyncio
    asyncio.run(remove_excluded_documents()) 