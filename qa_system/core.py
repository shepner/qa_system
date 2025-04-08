"""
Core QA System implementation
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .document_processor import DocumentProcessor
from .query_engine import QueryEngine
from .vector_store import VectorStore
from .config import load_config

def setup_logging():
    """Configure logging for the entire QA system."""
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging format and level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a file handler for persistent logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "qa_system.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

# Set up logging when module is imported
setup_logging()
logger = logging.getLogger(__name__)

class QASystem:
    """Main QA System class that orchestrates document processing and querying."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the QA System.
        
        Args:
            config_path: Optional path to config file. If not provided, uses environment variables.
        """
        self.config = load_config(config_path)
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.query_engine = QueryEngine(self.config, self.vector_store)
        
    async def add_documents(self, path: str) -> Dict[str, int]:
        """Process and add documents to the system.
        
        Args:
            path: Path to document or directory of documents
        
        Returns:
            Dict with processing statistics
        """
        path_obj = Path(path)
        if path_obj.is_file():
            files = [path_obj]
            logger.info(f"Adding single file: {path_obj}")
        else:
            # Get all files but filter out excluded ones
            files = [f for f in path_obj.glob("**/*") 
                    if f.is_file() and not self.doc_processor.should_exclude(f)]
            logger.info(f"Found {len(files)} valid files in directory: {path_obj}")
            
        stats = {
            "processed": 0, 
            "failed": 0, 
            "skipped": 0,  # Track skipped files
            "unchanged": 0,  # Track files that haven't changed
            "excluded": 0,  # Track excluded files
            "chunks": 0
        }
        total_files = len(files)
        
        for i, file in enumerate(files, 1):
            try:
                logger.info(f"Processing file {i}/{total_files}: {file}")
                
                # Check if file exists in vector store and hasn't been modified
                existing_doc = self.vector_store.get_document_by_path(str(file))
                was_successful = existing_doc and existing_doc.get("processing_status") == "success"
                
                # Add detailed status logging
                if existing_doc:
                    status = existing_doc.get("processing_status", "unknown")
                    doc_id = existing_doc.get("id", "N/A")
                    logger.info(f"Found file {file} in vector store: status='{status}', doc_id='{doc_id}'")
                else:
                    logger.info(f"File {file} not found in vector store")
                
                if existing_doc:
                    # Only skip if file exists AND hasn't been modified AND was successfully processed
                    if not self.doc_processor._needs_reprocessing(file, existing_doc["id"], was_successful):
                        logger.info(f"Skipping unchanged file: {file}")
                        stats["unchanged"] += 1
                        continue
                    else:
                        if not was_successful:
                            logger.info(f"File {file} had failed processing previously, retrying")
                        else:
                            logger.info(f"File {file} has changed, reprocessing")
                
                doc_metadata = await self.doc_processor.process_document(str(file))
                embeddings = await self.doc_processor.generate_embeddings(doc_metadata)
                
                logger.info(f"Storing {len(embeddings)} embeddings for {file}")
                await self.vector_store.store_embeddings(embeddings, doc_metadata)
                
                stats["processed"] += 1
                stats["chunks"] += len(embeddings)
                
            except ValueError as e:
                # Expected errors (unsupported files, etc)
                stats["skipped"] += 1
                logger.info(f"Skipped {file}: {str(e)}")
                
            except Exception as e:
                # Unexpected errors
                stats["failed"] += 1
                logger.error(f"Failed to process {file}: {str(e)}", exc_info=True)
            
            # Log progress after each file
            logger.info(f"Progress: {i}/{total_files} files processed ({(i/total_files)*100:.1f}%)")
            logger.info(f"Current stats - Processed: {stats['processed']}, "
                       f"Failed: {stats['failed']}, Skipped: {stats['skipped']}, "
                       f"Excluded: {stats['excluded']}, "
                       f"Unchanged: {stats['unchanged']}, "
                       f"Total Chunks: {stats['chunks']}")
                
        logger.info("Document processing complete!")
        logger.info(f"Final stats - Processed: {stats['processed']}, "
                   f"Failed: {stats['failed']}, Skipped: {stats['skipped']}, "
                   f"Excluded: {stats['excluded']}, "
                   f"Unchanged: {stats['unchanged']}, "
                   f"Total Chunks: {stats['chunks']}")
        return stats
    
    async def ask(self, question: str) -> Dict:
        """Ask a question and get an answer with sources.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing answer and source documents
        """
        return await self.query_engine.process_query(question)
    
    def list_documents(self) -> List[Dict]:
        """List all indexed documents.
        
        Returns:
            List of document metadata
        """
        return self.vector_store.list_documents()
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the system.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if successful
        """
        return await self.vector_store.remove_document(doc_id) 