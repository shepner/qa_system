#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from remove_files.config import get_config
from remove_files.logging_setup import setup_logging
from remove_files.file_matcher import FileMatcher
from remove_files.document_remover import DocumentRemover

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Remove files and their vector embeddings from the vector database."
    )
    parser.add_argument(
        "--remove",
        action="append",
        required=True,
        help="Path or pattern to remove. Can be specified multiple times."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.yaml",
        help="Path to configuration file (default: ./config/config.yaml)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate removal without actual deletion"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    return parser.parse_args()

def confirm_removal(files: List[str], force: bool = False) -> bool:
    """Ask for user confirmation before removing files.
    
    Args:
        files: List of files to be removed
        force: Skip confirmation if True
        
    Returns:
        bool: True if user confirms or force is True, False otherwise
    """
    if force:
        return True
        
    print("\nThe following files will be removed:")
    for file in files:
        print(f"  - {file}")
    
    response = input("\nDo you want to proceed with removal? [y/N]: ").lower()
    return response in ['y', 'yes']

def main() -> int:
    """Main entry point for the file removal system.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = get_config(args.config)
        
        # Setup logging
        setup_logging(
            LOG_FILE=config.get_nested('LOGGING.LOG_FILE', default="logs/remove_files.log"),
            LOG_LEVEL="DEBUG" if args.debug else config.get_nested('LOGGING.LEVEL', default="INFO"),
            DEBUG=args.debug
        )
        
        logger.info("Starting file removal process")
        logger.debug(f"Arguments: {args}")
        
        # Initialize components
        file_matcher = FileMatcher(config)
        document_remover = DocumentRemover(args.config)
        
        # Find matching files
        matching_files = []
        for pattern in args.remove:
            files = file_matcher.find_matching_files(pattern)
            matching_files.extend(files)
            
        if not matching_files:
            logger.warning("No matching files found")
            return 0
            
        logger.info(f"Found {len(matching_files)} matching files")
        
        # Handle dry run
        if args.dry_run:
            print("\nDry run - the following files would be removed:")
            for file in matching_files:
                print(f"  - {file}")
            return 0
            
        # Get user confirmation if required
        require_confirmation = config.get_nested(
            'REMOVAL_VALIDATION.REQUIRE_CONFIRMATION',
            default=True
        )
        
        if require_confirmation and not confirm_removal(matching_files, args.force):
            logger.info("Removal cancelled by user")
            return 0
            
        # Remove files
        success = document_remover.remove_documents(matching_files)
        
        if success:
            logger.info("File removal completed successfully")
            return 0
        else:
            logger.error("File removal failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 