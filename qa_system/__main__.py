"""
Main entry point for the QA system.

This module provides a command-line interface with three modes:
- Help Mode: Displays usage information when no parameters are provided
- Interactive Mode: Provides an interactive shell for working with the system
- Directive Mode: Executes a single command with arguments
"""
import asyncio
import logging
import sys
import yaml
from pathlib import Path

from .cli import CLI, parse_args

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def async_main():
    """Main async entry point."""
    args = parse_args()
    
    # Help Mode - no arguments provided
    if not args.config_file:
        CLI.display_help()
        return
        
    # Load configuration
    try:
        config_path = Path(args.config_file)
        if not config_path.exists():
            print(f"Configuration file not found: {args.config_file}")
            return
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return
        
    # Initialize and run CLI
    cli = CLI(config)
    await cli.run(args)

def main():
    """Main entry point."""
    setup_logging()
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logging.exception("Unhandled error")
        sys.exit(1)

if __name__ == "__main__":
    main() 