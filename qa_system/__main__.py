"""Main module for the QA system.

This module serves as the command-line interface and entry point for the QA system.
It handles argument parsing, configuration loading, logging setup, and orchestrates
the document processing workflow.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from qa_system.config import get_config, Config
from qa_system.logging_setup import setup_logging
from qa_system.file_scanner import FileScanner

# Create parser at module level
parser = argparse.ArgumentParser(description="QA System")
parser.add_argument(
    "--add",
    type=str,
    action='append',
    help="Path to process (can be a directory or individual file). Can be specified multiple times.",
)
parser.add_argument(
    "--config",
    type=str,
    default="./config/config.yaml",
    help="Path to configuration file",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging",
)

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    return parser.parse_args()

def main() -> None:
    """Main entry point for the QA system.
    
    This function:
    1. Parses command line arguments
    2. Loads system configuration
    3. Sets up logging based on configuration
    4. Processes input files using the file scanner
    5. Handles errors and provides appropriate exit codes
    
    Exit codes:
        0: Successful execution
        1: Error occurred during execution
    """
    try:
        args = parse_args()
        logger: Optional[logging.Logger] = None

        # Load configuration first
        config: Config = get_config(args.config)
        
        # Setup logging using configuration values
        # Pass arguments in order: LOG_FILE, LOG_LEVEL, DEBUG as per architecture spec
        setup_logging(
            config.get_nested('LOGGING.LOG_FILE'),
            config.get_nested('LOGGING.LEVEL', default="INFO"),
            args.debug
        )
        
        logger = logging.getLogger(__name__)
        logger.debug(f"Configuration loaded successfully from {args.config}")

        if args.add:
            paths: List[str] = args.add
            logger.info(f"Processing {len(paths)} input paths")
            scanner = FileScanner(config)
            
            # Process each provided path
            total_files = 0
            for path in paths:
                logger.info(f"Scanning path: {path}")
                try:
                    file_info = scanner.scan_files(path)
                    total_files += len(file_info)
                    logger.debug(f"Found {len(file_info)} files in {path}")
                except Exception as e:
                    logger.error(f"Failed to process path {path}: {str(e)}")
                    continue
                
                # TODO: Add document processing, embedding generation, and vector storage
                # This will be implemented in subsequent phases as per architecture Section 4.1
            
            if total_files > 0:
                logger.info(f"Total files found: {total_files}")
                logger.info("Document processing completed successfully")
            else:
                logger.warning("No files were processed")
        else:
            # Print help if no paths provided
            parser.print_help()
            sys.exit(0)
            
        # Successful execution
        sys.exit(0)

    except Exception as e:
        # Ensure we have a logger even if setup failed
        if logger is None:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger(__name__)
        
        logger.error(f"Fatal error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 