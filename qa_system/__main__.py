"""Main entry point for the qa_system package."""

import sys
from qa_system.config import get_config
from qa_system.document_processor import DocumentProcessor

def main():
    """Main function to run the qa_system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QA System')
    parser.add_argument('--add', help='Add documents from the specified directory')
    parser.add_argument('--config', default='./config/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    config.load(args.config)
    
    # Initialize document processor
    processor = DocumentProcessor(config)
    
    # Process command
    if args.add:
        processor.process_directory(args.add)
    else:
        parser.print_help()

if __name__ == '__main__':
    sys.exit(main()) 