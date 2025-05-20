"""
@file: __main__.py
Main entry point for the QA System CLI.

This module provides a command-line interface for document processing and question-answering operations, including:
- Adding documents to the vector store
- Listing documents (with optional filtering)
- Removing documents
- Querying the system (single or interactive mode)

It handles argument parsing, configuration loading, logging setup, and delegates to the appropriate handlers for each operation.
"""

#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from qa_system.config import get_config
from qa_system.exceptions import QASystemError
from qa_system.logging_setup import setup_logging
from qa_system.list import get_list_module

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the QA System CLI.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    logger.info("Called parse_args()")
    parser = argparse.ArgumentParser(
        description="QA System - Document processing and question-answering system"
    )
    
    # Mutually exclusive operation group
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--add",
        action="append",
        help="Add document(s) to the system. Can be specified multiple times."
    )
    operation_group.add_argument(
        "--list",
        nargs="?",
        const="*",
        help="List documents in the system (optionally filter by glob pattern)"
    )
    operation_group.add_argument(
        "--remove",
        action="append",
        help="Remove document(s) from the system. Can be specified multiple times."
    )
    operation_group.add_argument(
        "--query",
        nargs="?",
        const="",
        help="Query the system. If no query is provided, enters interactive mode."
    )
    
    # Optional filter for list/remove
    parser.add_argument(
        "--filter",
        help="Filter pattern for list operation (e.g., '*.pdf')"
    )
    
    # Debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Config file option
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    return parser.parse_args()

def _serialize_metadata(metadata):
    """
    Convert all list values in metadata to comma-separated strings.
    """
    for k, v in metadata.items():
        if isinstance(v, list):
            metadata[k] = ','.join(str(x) for x in v)
    return metadata

def process_add_files(files: List[str], config: dict) -> int:
    """Process and add files to the system.
    
    Args:
        files: List of file paths to process
        config: Configuration dictionary
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Called process_add_files(files={files}, config={{...}})")
    try:
        from qa_system.document_processors import FileScanner, get_processor_for_file_type
        from qa_system.embedding import EmbeddingGenerator
        from qa_system.vector_store import ChromaVectorStore
        from qa_system.query import QueryProcessor
        
        scanner = FileScanner(config)
        store = ChromaVectorStore(config)
        generator = EmbeddingGenerator(config)
        query_processor = QueryProcessor(config)
        
        for file_path in files:
            print(f"\nScanning: {file_path}")
            logger.info(f"Processing file: {file_path}")
            
            # Scan files and check if they need processing
            scan_results = scanner.scan_files(file_path)
            
            for result in scan_results:
                logger.info(f"Checking if file checksum exists in vector DB: {result['checksum']} for {result['path']}")
                hash_exists = store.has_file(result['checksum'])
                logger.info(f"Checksum check for {result['path']} (checksum={result['checksum']}): {'FOUND' if hash_exists else 'NOT FOUND'} in vector DB")
                result['needs_processing'] = not hash_exists
                if not result['needs_processing']:
                    print('.', end='', flush=True)
                else:
                    print(f"\n{result['path']}")
            
            for result in scan_results:
                if not result['needs_processing']:
                    logger.info(f"Skipping file (already exists in vector DB by checksum): {result['path']} (checksum={result['checksum']})")
                    continue
                logger.info(f"File needs processing (not found in vector DB): {result['path']} (checksum={result['checksum']})")
                # Get appropriate processor for file type
                processor = get_processor_for_file_type(result['path'], config, query_processor=query_processor)
                
                # Process file into chunks
                processed = processor.process(result['path'])
                
                # If the file was skipped (e.g., encrypted PDF), log and continue
                if processed['metadata'].get('skipped'):
                    print(f"Skipping (processing issue): {result['path']} (reason: {processed['metadata'].get('skip_reason')})")
                    logger.warning(f"Skipping file due to processing issue: {result['path']} (reason: {processed['metadata'].get('skip_reason')})")
                    continue
                
                # --- NEW: Handle files with zero chunks (metadata-only entry) ---
                if not processed['chunks']:
                    print(f"No chunks generated for: {result['path']}. Adding metadata-only entry to vector store.")
                    logger.info(f"No chunks generated for {result['path']}. Adding metadata-only entry to vector store.")
                    zero_vector = [0.0] * generator.dimensions
                    meta = _serialize_metadata(dict(processed['metadata']))
                    meta['id'] = f"{meta['path']}:0"
                    meta['checksum'] = result['checksum']
                    store.add_embeddings(
                        embeddings=[zero_vector],
                        texts=[""],
                        metadatas=[meta]
                    )
                    print(f"Added metadata-only entry to vector store: {result['path']}")
                    logger.info(f"Added metadata-only entry to vector store: {result['path']}")
                    continue
                
                # Assign unique IDs to each chunk's metadata
                chunk_metadatas = []
                for idx, chunk in enumerate(processed['chunks']):
                    meta = _serialize_metadata(dict(processed['metadata']))
                    meta['id'] = f"{meta['path']}:{idx}"
                    meta['checksum'] = result['checksum']  # Ensure checksum is present in every chunk's metadata
                    chunk_metadatas.append(meta)
                
                print(f"Generating embeddings for: {result['path']}")
                # Generate embeddings
                file_type = processed['metadata'].get('file_type', '').lower()
                if file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
                    embeddings = generator.generate_image_embeddings(
                        image_path=processed['metadata']['path'],
                        metadata=processed['metadata']
                    )
                else:
                    embeddings = generator.generate_embeddings(
                        texts=[chunk['text'] for chunk in processed['chunks']],
                        metadata=processed['metadata']
                    )
                # Overwrite embeddings['metadata'] with chunk_metadatas
                embeddings['metadata'] = chunk_metadatas
                
                # Defensive check before adding to vector store
                nonempty_texts = [t for t in embeddings['texts'] if t and t.strip()]
                if not embeddings['vectors'] or not embeddings['texts'] or not embeddings['metadata']:
                    if nonempty_texts:
                        print(f"No embeddings generated for: {result['path']}, but chunk texts were non-empty. Likely rejected by embedding model (all filetypes). Indexing metadata only.")
                        logger.warning(f"No embeddings generated for {result['path']}, but chunk texts were non-empty. Likely rejected by embedding model (all filetypes). Indexing metadata only.")
                        zero_vector = [0.0] * generator.dimensions
                        store.add_embeddings(
                            embeddings=[zero_vector for _ in nonempty_texts],
                            texts=nonempty_texts,
                            metadatas=[_serialize_metadata(m) for t, m in zip(embeddings['texts'], embeddings['metadata']) if t and t.strip()]
                        )
                        print(f"Added to vector store (metadata only): {result['path']}")
                        logger.info(f"Added to vector store (metadata only): {result['path']}")
                        continue
                    else:
                        print(f"No embeddings generated for: {result['path']}, skipping add to vector store. All chunk texts empty or whitespace.")
                        logger.warning(f"No embeddings generated for {result['path']}, skipping add to vector store. All chunk texts empty or whitespace.")
                        continue
                if not (len(embeddings['vectors']) == len(embeddings['texts']) == len(embeddings['metadata'])):
                    print(f"Mismatch in number of vectors, texts, and metadatas for: {result['path']}, skipping.")
                    logger.error(f"Mismatch in number of vectors, texts, and metadatas for {result['path']}, skipping.")
                    continue

                # Add to vector store
                store.add_embeddings(
                    embeddings=embeddings['vectors'],
                    texts=embeddings['texts'],
                    metadatas=[_serialize_metadata(m) for m in embeddings['metadata']]
                )
                
                print(f"Added to vector store: {result['path']}")
                logger.info(f"Successfully processed and added: {result['path']}")
                
        return 0
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        logger.error(f"Error processing files: {str(e)}")
        return 1

def process_list(filter_pattern: Optional[str], config: dict) -> int:
    """List documents in the system.
    
    Args:
        filter_pattern: Optional pattern to filter documents
        config: Configuration dictionary
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Called process_list(filter_pattern={filter_pattern}, config={{...}})")
    try:
        from qa_system.document_processors import ListHandler
        
        handler = ListHandler(config)
        documents = handler.list_metadata(filter_pattern)
        
        if not documents:
            print("No documents found")
            return 0
            
        # Print document list
        print("\nDocuments in system:")
        print("-" * 80)
        print(f"{'Path':<50} {'Type':<10} {'Chunks':<8} {'Last Modified':<20}")
        print("-" * 80)
        
        for doc in documents:
            print(
                f"{doc['path']:<50} "
                f"{doc['metadata']['file_type']:<10} "
                f"{doc['metadata'].get('chunk_count', '-'):<8} "
                f"{doc['metadata'].get('last_modified', '-'):<20}"
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return 1

def process_remove(paths: List[str], filter_pattern: Optional[str], config: dict) -> int:
    """Remove documents from the system.
    
    Args:
        paths: List of paths to remove
        filter_pattern: Optional pattern to filter documents
        config: Configuration dictionary
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Called process_remove(paths={paths}, filter_pattern={filter_pattern}, config={{...}})")
    if not paths and not filter_pattern:
        logger.error("No paths or filter pattern provided for removal")
        return 1
        
    try:
        from qa_system.remove_handler import RemoveHandler
        
        handler = RemoveHandler(config)
        result = handler.remove_documents(paths, filter_pattern)
        
        # Print results
        if result['removed']:
            print("\nSuccessfully removed:")
            for path in result['removed']:
                print(f"  - {path}")
                
        if result['failed']:
            print("\nFailed to remove:")
            for path, error in result['failed'].items():
                print(f"  - {path}: {error}")
                
        if result['not_found']:
            print("\nNo matches found for:")
            for path in result['not_found']:
                print(f"  - {path}")
                
        return 0 if not result['failed'] else 1
        
    except Exception as e:
        logger.error(f"Error removing documents: {str(e)}")
        return 1

def process_query(query: Optional[str], config: dict) -> int:
    """Process a query or enter interactive query mode.
    
    Args:
        query: Query string or None for interactive mode
        config: Configuration dictionary
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Called process_query(query={query}, config={{...}})")
    try:
        from qa_system.query import QueryProcessor, print_response
        
        processor = QueryProcessor(config)
        
        if query:
            # Single query mode
            response = processor.process_query(query)
            print_response(response)
        else:
            # Interactive mode
            print("Enter your questions (type 'exit' to quit):")
            while True:
                try:
                    query = input("\nQuestion: ").strip()
                    if query.lower() in ('exit', 'quit'):
                        break
                    if not query:
                        continue
                        
                    response = processor.process_query(query)
                    print_response(response)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                    
        return 0
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return 1

def main() -> int:
    """Main entry point for the QA System CLI.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info("Called main()")
    try:
        args = parse_args()
        
        # Load configuration
        config = get_config(args.config)
        
        # Setup logging
        log_level = "DEBUG" if args.debug else config.get_nested('LOGGING.LEVEL', 'INFO')
        setup_logging(
            LOG_FILE=config.get_nested('LOGGING.LOG_FILE', 'logs/qa_system.log'),
            LEVEL=log_level
        )
        # Ensure all loggers inherit the root logger's level
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.NOTSET)
        
        # Process command
        if args.add:
            return process_add_files(args.add, config)
        elif args.list is not None:
            list_module = get_list_module(config)
            pattern = args.list if args.list != '*' else None
            docs = list_module.list_metadata(pattern=pattern)
            print(f"\nDocuments in vector store ({len(docs)} found):")
            for doc in docs:
                print(f"- {doc.get('path', '[no path]')} (checksum={doc.get('checksum', '[no checksum]')})")
            stats = list_module.get_collection_stats()
            print(f"\nTotal documents: {stats['total_documents']}")
            print(f"Document types: {stats['document_types']}")
            return 0
        elif args.remove:
            return process_remove(args.remove, args.filter, config)
        elif args.query is not None:  # Empty string is valid for interactive mode
            return process_query(args.query, config)
            
        return 0
        
    except QASystemError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 