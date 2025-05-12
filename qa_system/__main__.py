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

logging.basicConfig(
    level=logging.INFO,  # Default to INFO; can be overridden by setup_logging()
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    logger.info("Called parse_args()")
    parser = argparse.ArgumentParser(
        description="QA System - Document processing and question-answering system"
    )
    
    # Add operation group
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
    
    # Add filter option for list operation
    parser.add_argument(
        "--filter",
        help="Filter pattern for list operation (e.g., '*.pdf')"
    )
    
    # Add debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Add config file option
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    return parser.parse_args()

def process_add_files(files: List[str], config: dict) -> int:
    """Process and add files to the system.
    
    Args:
        files: List of file paths to process
        config: Configuration dictionary
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Called process_add_files(files={files}, config={config})")
    try:
        from qa_system.document_processors import FileScanner, get_processor_for_file_type
        from qa_system.embedding import EmbeddingGenerator
        from qa_system.vector_store import ChromaVectorStore
        
        scanner = FileScanner(config)
        store = ChromaVectorStore(config)
        generator = EmbeddingGenerator(config)
        
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            
            # Scan files and check if they need processing
            scan_results = scanner.scan_files(file_path)
            
            for result in scan_results:
                logger.info(f"Checking if file checksum exists in vector DB: {result['checksum']} for {result['path']}")
                hash_exists = store.has_file(result['checksum'])
                logger.info(f"Checksum check for {result['path']} (checksum={result['checksum']}): {'FOUND' if hash_exists else 'NOT FOUND'} in vector DB")
                result['needs_processing'] = not hash_exists
            
            for result in scan_results:
                if not result['needs_processing']:
                    logger.info(f"Skipping file (already exists in vector DB by checksum): {result['path']} (checksum={result['checksum']})")
                    continue
                logger.info(f"File needs processing (not found in vector DB): {result['path']} (checksum={result['checksum']})")
                # Get appropriate processor for file type
                processor = get_processor_for_file_type(result['path'], config)
                
                # Process file into chunks
                processed = processor.process(result['path'])
                
                # If the file was skipped (e.g., encrypted PDF), log and continue
                if processed['metadata'].get('skipped'):
                    logger.warning(f"Skipping file due to processing issue: {result['path']} (reason: {processed['metadata'].get('skip_reason')})")
                    continue
                
                # Assign unique IDs to each chunk's metadata
                chunk_metadatas = []
                for idx, chunk in enumerate(processed['chunks']):
                    meta = dict(processed['metadata'])
                    meta['id'] = f"{meta['path']}:{idx}"
                    meta['checksum'] = result['checksum']  # Ensure checksum is present in every chunk's metadata
                    chunk_metadatas.append(meta)
                
                # Generate embeddings
                embeddings = generator.generate_embeddings(
                    texts=[chunk['text'] for chunk in processed['chunks']],
                    metadata=processed['metadata']
                )
                # Overwrite embeddings['metadata'] with chunk_metadatas
                embeddings['metadata'] = chunk_metadatas
                
                # Defensive check before adding to vector store
                if not embeddings['vectors'] or not embeddings['texts'] or not embeddings['metadata']:
                    logger.warning(f"No embeddings generated for {result['path']}, skipping add to vector store.")
                    continue
                if not (len(embeddings['vectors']) == len(embeddings['texts']) == len(embeddings['metadata'])):
                    logger.error(f"Mismatch in number of vectors, texts, and metadatas for {result['path']}, skipping.")
                    continue

                # Add to vector store
                store.add_embeddings(
                    embeddings=embeddings['vectors'],
                    texts=embeddings['texts'],
                    metadatas=embeddings['metadata']
                )
                
                logger.info(f"Successfully processed and added: {result['path']}")
                
        return 0
        
    except Exception as e:
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
    logger.info(f"Called process_list(filter_pattern={filter_pattern}, config={config})")
    try:
        from qa_system.document_processors import ListHandler
        
        handler = ListHandler(config)
        documents = handler.list_documents(filter_pattern)
        
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
    logger.info(f"Called process_remove(paths={paths}, filter_pattern={filter_pattern}, config={config})")
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
    logger.info(f"Called process_query(query={query}, config={config})")
    try:
        from qa_system.query import QueryProcessor
        
        processor = QueryProcessor(config)
        
        def print_response(response):
            print("\nAnswer:")
            print("-" * 80)
            print(response.text)
            print("\nSources:")
            print("-" * 80)
            for source in response.sources:
                print(f"- {source.document} (similarity: {source.similarity:.2f})")
        
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
    """Main entry point.
    
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
        
        # Process command
        if args.add:
            return process_add_files(args.add, config)
        elif args.list is not None:
            list_module = get_list_module(config)
            pattern = args.list if args.list != '*' else None
            docs = list_module.list_documents(pattern=pattern)
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