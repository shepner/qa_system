"""Main entry point for the qa_system package."""

import sys
import logging
from pathlib import Path
from embed_files.config import get_config, ConfigurationError
from embed_files.document_processor import DocumentProcessor
from embed_files.logging_setup import setup_logging

def main():
    """Main function to run the qa_system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QA System')
    parser.add_argument('--add', help='Add documents from the specified directory')
    parser.add_argument('--config', default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize logging with debug flag
        setup_logging(
            log_file="qa_system.log",
            enable_debug=args.debug
        )
        logger = logging.getLogger(__name__)
        
        # Load configuration with specified config file
        logger.info(f"Loading configuration from {args.config}")
        config = get_config(args.config)
        logger.debug("Configuration loaded successfully")
        
        # Initialize document processor
        processor = DocumentProcessor(config)
        
        # Process command
        if args.add:
            input_path = Path(args.add)
            if not input_path.exists():
                logger.error(f"Directory does not exist: {input_path}")
                return 1
                
            logger.info(f"Processing documents from directory: {input_path}")
            try:
                processed_count = 0
                for result in processor.process_directory(input_path):
                    logger.info(f"Processed document: {result['path']}")
                    processed_count += 1
                logger.info(f"Successfully processed {processed_count} documents")
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}", exc_info=True)
                return 1
        else:
            parser.print_help()
            
        return 0
        
    except ConfigurationError as e:
        # Since logging might not be set up yet in case of early config errors
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        # Use print for early errors before logging is setup
        print(f"Failed to initialize QA system: {str(e)}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 