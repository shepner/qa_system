"""Add flow module for processing and embedding files.

This module orchestrates the process of scanning files, processing documents,
generating embeddings, and storing them in the vector database.
"""

import logging
import time
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
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initializing AddFlow",
            extra={
                'component': 'add_flow',
                'operation': 'init',
                'config_path': config_path
            }
        )
        
        try:
            # Load configuration
            self.config = get_config(config_path)
            self.logger.debug(
                "Configuration loaded",
                extra={
                    'component': 'add_flow',
                    'operation': 'init',
                    'config_sections': list(self.config.keys()) if hasattr(self.config, 'keys') else None
                }
            )
            
            # Initialize components in correct order
            self.vector_store = VectorStore(config_path)
            self.scanner = FileScanner(self.config, vector_store=self.vector_store)
            self.embedding_generator = EmbeddingGenerator(self.logger)
            
            self.logger.info(
                "AddFlow initialized successfully",
                extra={
                    'component': 'add_flow',
                    'operation': 'init',
                    'scanner_type': type(self.scanner).__name__,
                    'vector_store_type': type(self.vector_store).__name__,
                    'embedding_generator_type': type(self.embedding_generator).__name__
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize AddFlow: {str(e)}",
                extra={
                    'component': 'add_flow',
                    'operation': 'init',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
        
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
        start_time = time.time()
        self.logger.info(
            "Starting path processing",
            extra={
                'component': 'add_flow',
                'operation': 'process_path',
                'path': path
            }
        )
        
        try:
            # 1. File Scanner: Get list of files to process
            self.logger.debug(
                "Scanning for files",
                extra={
                    'component': 'add_flow',
                    'operation': 'scan_files',
                    'path': path
                }
            )
            files = self.scanner.scan_files(path)
            self.logger.info(
                "File scanning complete",
                extra={
                    'component': 'add_flow',
                    'operation': 'scan_files',
                    'files_found': len(files),
                    'path': path
                }
            )
            
            successful = 0
            failed = 0
            
            for file_info in files:
                file_start_time = time.time()
                self.logger.debug(
                    "Processing file",
                    extra={
                        'component': 'add_flow',
                        'operation': 'process_file',
                        'file_path': file_info['path'],
                        'file_type': file_info.get('file_type'),
                        'file_size': file_info.get('size')
                    }
                )
                
                try:
                    # Get appropriate document processor for file type
                    processor = self.scanner.get_processor_for_file(file_info['path'])
                    self.logger.debug(
                        "Document processor selected",
                        extra={
                            'component': 'add_flow',
                            'operation': 'get_processor',
                            'processor_type': type(processor).__name__,
                            'file_path': file_info['path']
                        }
                    )
                    
                    # 2. Document Processor: Process file and get chunks
                    processed_data = processor.process(
                        file_info['path'],
                        file_info
                    )
                    
                    if processed_data and 'chunks' in processed_data:
                        chunk_count = len(processed_data['chunks'])
                        self.logger.debug(
                            "Document processing complete",
                            extra={
                                'component': 'add_flow',
                                'operation': 'process_document',
                                'file_path': file_info['path'],
                                'chunks_generated': chunk_count,
                                'metadata_keys': list(processed_data.get('metadata', {}).keys())
                            }
                        )
                        
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
                                'total_chunks': chunk_count
                            })
                            # Add any chunk-specific metadata if it exists
                            if isinstance(chunk, dict):
                                chunk_meta.update({k: v for k, v in chunk.items() if k != 'content'})
                            chunk_metadata.append(chunk_meta)
                        
                        self.logger.debug(
                            "Generating embeddings",
                            extra={
                                'component': 'add_flow',
                                'operation': 'generate_embeddings',
                                'file_path': file_info['path'],
                                'chunk_count': chunk_count,
                                'metadata_fields': list(chunk_metadata[0].keys()) if chunk_metadata else None
                            }
                        )
                        
                        # 3. Embedding Generator: Generate embeddings for chunks
                        embedding_results = self.embedding_generator.generate_embeddings_batch(
                            texts=chunk_texts,
                            metadata=chunk_metadata
                        )
                        
                        if embedding_results and 'embeddings' in embedding_results:
                            self.logger.debug(
                                "Storing embeddings",
                                extra={
                                    'component': 'add_flow',
                                    'operation': 'store_embeddings',
                                    'file_path': file_info['path'],
                                    'embedding_count': len(embedding_results['embeddings']),
                                    'metadata_count': len(embedding_results['processed_metadata'])
                                }
                            )
                            
                            # 4. Vector Store: Store embeddings and metadata
                            self.vector_store.add_embeddings(
                                embeddings=embedding_results['embeddings'],
                                metadata=embedding_results['processed_metadata']
                            )
                            successful += 1
                            
                            file_duration = time.time() - file_start_time
                            self.logger.info(
                                "File processed successfully",
                                extra={
                                    'component': 'add_flow',
                                    'operation': 'process_file',
                                    'file_path': file_info['path'],
                                    'duration': f"{file_duration:.2f}s",
                                    'chunks_processed': chunk_count,
                                    'embeddings_stored': len(embedding_results['embeddings'])
                                }
                            )
                        else:
                            failed += 1
                            self.logger.error(
                                "No embeddings generated",
                                extra={
                                    'component': 'add_flow',
                                    'operation': 'generate_embeddings',
                                    'file_path': file_info['path'],
                                    'error': 'No embeddings in results'
                                }
                            )
                    else:
                        failed += 1
                        self.logger.error(
                            "Document processing failed",
                            extra={
                                'component': 'add_flow',
                                'operation': 'process_document',
                                'file_path': file_info['path'],
                                'error': 'No chunks generated',
                                'processed_data_keys': list(processed_data.keys()) if processed_data else None
                            }
                        )
                        
                except Exception as e:
                    failed += 1
                    self.logger.error(
                        "File processing failed",
                        extra={
                            'component': 'add_flow',
                            'operation': 'process_file',
                            'file_path': file_info['path'],
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        },
                        exc_info=True
                    )
            
            total_duration = time.time() - start_time
            self.logger.info(
                "Path processing complete",
                extra={
                    'component': 'add_flow',
                    'operation': 'process_path',
                    'path': path,
                    'total_files': len(files),
                    'successful': successful,
                    'failed': failed,
                    'duration': f"{total_duration:.2f}s"
                }
            )
            
            return {
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'duration': total_duration
            }
            
        except Exception as e:
            self.logger.error(
                "Path processing failed",
                extra={
                    'component': 'add_flow',
                    'operation': 'process_path',
                    'path': path,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            raise 