"""Add flow module for processing and embedding files.

This module orchestrates the process of scanning files, processing documents,
generating embeddings, and storing them in the vector database.
"""

import logging
from typing import Dict, Any, List
from qa_system.config import get_config
from qa_system.file_scanner import FileScanner
from qa_system.vector_system import VectorStore
from qa_system.embedding_system import EmbeddingGenerator
from qa_system.exceptions import QASystemError, handle_exception

class AddFlow:
    """Handles the process of adding files to the system."""
    
    def __init__(self, config_path: str):
        """Initialize the add flow.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigError: If configuration is invalid
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize components
        self.scanner = FileScanner(self.config)
        self.vector_store = VectorStore(config_path)
        self.logger = logging.getLogger(__name__)
        self.embedding_generator = EmbeddingGenerator(self.logger)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def process_path(self, path: str) -> Dict[str, Any]:
        """Process a file or directory path.
        
        Args:
            path: Path to process
            
        Returns:
            Dictionary containing processing statistics
            
        Raises:
            FileNotFoundError: If path does not exist
            ProcessingError: If processing fails
        """
        try:
            # 1. File Scanner: Get list of files to process
            files = self.scanner.scan_files(path)
            
            successful = 0
            failed = 0
            
            for file_info in files:
                try:
                    # Get appropriate document processor for file type
                    processor = self.scanner.get_processor_for_file(file_info['path'])
                    
                    # 2. Document Processor: Process file and get chunks
                    processed_data = processor.process(
                        file_info['path'],
                        file_info
                    )
                    
                    if processed_data and 'chunks' in processed_data:
                        # Extract text content from chunks
                        chunk_texts = [chunk['content'] for chunk in processed_data['chunks']]
                        
                        # Create metadata list for each chunk
                        chunk_metadata = []
                        base_metadata = processed_data.get('metadata', {})
                        for i, chunk in enumerate(processed_data['chunks']):
                            # Combine base metadata with chunk-specific metadata
                            chunk_meta = base_metadata.copy()
                            chunk_meta.update({
                                'chunk_index': i,
                                'total_chunks': len(processed_data['chunks'])
                            })
                            # Add any chunk-specific metadata if it exists
                            if isinstance(chunk, dict):
                                chunk_meta.update({k: v for k, v in chunk.items() if k != 'content'})
                            chunk_metadata.append(chunk_meta)
                        
                        # 3. Embedding Generator: Generate embeddings for chunks
                        embedding_results = self.embedding_generator.generate_embeddings_batch(
                            texts=chunk_texts,
                            metadata=chunk_metadata
                        )
                        
                        if embedding_results and 'embeddings' in embedding_results:
                            # 4. Vector Store: Store embeddings and metadata
                            self.vector_store.add_embeddings(
                                embeddings=embedding_results['embeddings'],
                                metadata=embedding_results['processed_metadata']
                            )
                            successful += 1
                        else:
                            failed += 1
                            self.logger.error(
                                f"No embeddings generated for file: {file_info['path']}"
                            )
                    else:
                        failed += 1
                        self.logger.error(
                            f"File not properly processed: {file_info['path']}"
                        )
                        
                except Exception as e:
                    failed += 1
                    self.logger.error(
                        f"Failed to process file: {file_info['path']}: {str(e)}",
                        exc_info=True
                    )
            
            return {
                'total_files': len(files),
                'successful': successful,
                'failed': failed
            }
            
        except Exception as e:
            self.logger.error(f"Error processing path {path}: {str(e)}", exc_info=True)
            raise 