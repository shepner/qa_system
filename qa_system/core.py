"""
Core QA System implementation
"""
import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from .document_processor import DocumentProcessor
from .query_engine import QueryEngine
from .vector_store import VectorStore
from .config import load_config
from .qa_engine import QAEngine

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
    """Main QA System class that orchestrates document processing and querying.
    
    This class now delegates core functionality to QAEngine while maintaining
    backward compatibility with existing CLI interfaces.
    """
    
    def __init__(self, config_path: Optional[str | Dict[str, Any]] = None):
        """Initialize the QA System.
        
        Args:
            config_path: Optional path to config file or config dictionary. If not provided, uses environment variables.
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            self.config = load_config(config_path)
        self.qa_engine = QAEngine(self.config)
        
    async def initialize(self):
        """Initialize the QA system components."""
        await self.qa_engine.initialize()
        
    async def add_documents(self, path: str) -> Dict[str, int]:
        """Process and add documents to the system.
        
        Args:
            path: Path to document or directory of documents
        
        Returns:
            Dict with processing statistics
        """
        await self.initialize()  # Ensure system is initialized
        
        path_obj = Path(path)
        if path_obj.is_file():
            files = [path_obj]
            logger.info(f"Adding single file: {path_obj}")
        else:
            # Get all files but filter out excluded ones
            files = [f for f in path_obj.glob("**/*") 
                    if f.is_file() and not self.qa_engine.document_store.should_exclude(f)]
            logger.info(f"Found {len(files)} valid files in directory: {path_obj}")
            
        stats = {
            "processed": 0, 
            "failed": 0, 
            "skipped": 0,
            "unchanged": 0,
            "excluded": 0,
            "chunks": 0
        }
        
        total_files = len(files)
        
        for i, file in enumerate(files, 1):
            try:
                logger.info(f"Processing file {i}/{total_files}: {file}")
                result = await self.qa_engine.add_document(str(file))
                
                if result["status"] == "success":
                    stats["processed"] += 1
                    stats["chunks"] += result["chunks"]
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                elif result["status"] == "unchanged":
                    stats["unchanged"] += 1
                elif result["status"] == "excluded":
                    stats["excluded"] += 1
                
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"Failed to process {file}: {str(e)}", exc_info=True)
            
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
        await self.initialize()  # Ensure system is initialized
        return await self.qa_engine.get_answer(question)
    
    async def list_documents(self) -> List[Dict]:
        """List all documents in the vector store."""
        return await self.qa_engine.list_documents()
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the system.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if successful
        """
        await self.initialize()  # Ensure system is initialized
        return await self.qa_engine.remove_document(doc_id)
    
    async def ask_question(self, question: str) -> str:
        """Process a question and return an answer.
        
        Args:
            question: The question to answer
            
        Returns:
            The answer as a string
        """
        await self.initialize()  # Ensure system is initialized
        
        try:
            response = await self.qa_engine.get_answer(question)
            return response["answer"]
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise RuntimeError(f"Failed to process question: {str(e)}")
    
    async def cleanup(self) -> None:
        """Clean up resources used by the QA System."""
        logger.info("Cleaning up QA System resources...")
        try:
            if self.qa_engine:
                await self.qa_engine.cleanup()
                self.qa_engine = None
            logger.info("QA System cleanup complete")
        except Exception as e:
            logger.error(f"Error during QA System cleanup: {str(e)}", exc_info=True)
            raise RuntimeError(f"QA System cleanup failed: {str(e)}") 