"""Main entry point for the qa_system package."""

import sys
import logging
from pathlib import Path
from embed_files.config import get_config
from embed_files.document_processor import DocumentProcessor
from embed_files.logging_setup import setup_logging

def main():
    """Main function to run the qa_system."""
    import argparse
    
    # Set up basic console logging until we load the full configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='QA System')
    parser.add_argument('--add', help='Add documents from the specified directory')
    parser.add_argument('--config', default='./config/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = get_config()
        config.load_config(args.config)
        
        # Setup logging with full configuration
        setup_logging(config)
        logger.info("Logging system initialized successfully")
        
        # Initialize document processor
        processor = DocumentProcessor(config)
        
        # Process command
        if args.add:
            logger.info(f"Processing documents from directory: {args.add}")
            for result in processor.process_directory(Path(args.add)):
                logger.info(f"Processed document: {result['path']}")
        else:
            parser.print_help()
            
        return 0
        
    except Exception as e:
        logger.error(f"Failed to initialize QA system: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 