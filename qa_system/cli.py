"""
Command-line interface for the QA system

This module provides a command-line interface with two modes:
- Interactive Mode: Provides an interactive Q&A shell when no flags are provided
- Flag Mode: Executes a single operation based on provided flags
"""
import asyncio
import argparse
from typing import List, Optional, Dict, Any, Union, TypeVar, cast
import sys
from pathlib import Path
import logging
import textwrap
import json

from .core import QASystem
from .remove_excluded import remove_excluded_documents
from .document_processor import DocumentProcessor
from .qa_engine import QAEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    description = textwrap.dedent("""
        QA System - Local Document Question Answering
        
        Usage Modes:
          1. Interactive Mode (no flags):
             qa_system config.yaml
          2. Flag Mode (one operation flag):
             qa_system config.yaml --flag [args]
             
        Example:
          qa_system config.yaml --ask "What is the capital of France?"
          qa_system config.yaml --add /path/to/documents
          qa_system config.yaml --list
          qa_system config.yaml --remove doc_123
          qa_system config.yaml --cleanup
    """)
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration file is an optional positional argument
    parser.add_argument(
        'config_file',
        nargs='?',  # Make config_file optional
        type=str,
        help='Path to configuration file'
    )
    
    # Optional directives with -- flags
    parser.add_argument(
        '--add',
        help='Process and index documents from the given path',
        metavar='PATH'
    )
    
    parser.add_argument(
        '--ask',
        help='Get an answer to a specific question',
        metavar='QUESTION'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Show all indexed documents'
    )
    
    parser.add_argument(
        '--remove',
        help='Remove a document from the index by ID',
        metavar='DOC_ID'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not args.config_file:
        parser.print_help()
        sys.exit(0)
    
    return args

def display_help():
    """Display help information."""
    help_text = """
    QA System Interactive Mode
    
    Just type your question and press Enter to get an answer.
    To perform other operations, restart the program with appropriate flags:
    
    --add PATH      Process and index documents from the given path
    --ask QUESTION  Get an answer to a specific question
    --list         Show all indexed documents
    --remove ID    Remove a document from the index
    --cleanup      Clean up the index
    
    To exit, press Ctrl+C or type 'exit'
    """
    print(textwrap.dedent(help_text))

class CLI:
    """Command-line interface for the QA system."""
    
    def __init__(self, config: Union[str, Path, Dict[str, Any]], args: Optional[argparse.Namespace] = None):
        """Initialize the CLI.
        
        Args:
            config: Configuration file path or dictionary
            args: Command line arguments
        """
        self.config = config if isinstance(config, dict) else Path(config)
        self.args = args
        self.qa_system = None

    async def initialize(self):
        """Initialize the QA system and its components."""
        if self.qa_system is None:
            self.qa_system = QASystem(self.config)
            await self.qa_system.initialize()
    
    async def cleanup(self):
        """Clean up resources used by the CLI."""
        logger.info("Cleaning up CLI resources...")
        try:
            if self.qa_system:
                await self.qa_system.cleanup()
                self.qa_system = None
            logger.info("CLI cleanup complete")
        except Exception as e:
            logger.error(f"Error during CLI cleanup: {str(e)}", exc_info=True)
            raise RuntimeError(f"CLI cleanup failed: {str(e)}")

    async def _run_interactive(self):
        """Run the CLI in interactive mode."""
        await self.initialize()
        
        print("Welcome to the QA System Interactive Shell!")
        print("Type 'help' for a list of commands or 'exit' to quit.")
        
        try:
            while True:
                try:
                    command = input("\nEnter command: ").strip().lower()
                    
                    if command == "exit":
                        break
                    elif command == "help":
                        display_help()
                    elif command == "list":
                        await self.handle_list()
                    elif command.startswith("add "):
                        path = command[4:].strip()
                        await self.handle_add(path)
                    elif command.startswith("ask "):
                        question = command[4:].strip()
                        await self.handle_ask(question)
                    elif command.startswith("remove "):
                        doc_id = command[7:].strip()
                        await self.handle_remove(doc_id)
                    else:
                        # Treat any other input as a question
                        await self.handle_ask(command)
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit or Ctrl+C again to force quit")
                except Exception as e:
                    print(f"Error processing command: {str(e)}")
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
        finally:
            await self.cleanup()

    async def run(self, args: Optional[argparse.Namespace] = None):
        """Run the CLI in the appropriate mode based on provided arguments.
        
        Args:
            args: Command line arguments. If not provided, uses the args from initialization.
        """
        try:
            # Update instance args if provided
            if args is not None:
                self.args = args
            
            await self.initialize()
            
            # Process the stored args
            if self.args.add:
                await self.handle_add(self.args.add)
            elif self.args.ask:
                await self.handle_ask(self.args.ask)
            elif self.args.list:
                await self.handle_list()
            elif self.args.remove:
                await self.handle_remove(self.args.remove)
            else:
                await self._run_interactive()
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            print(f"An error occurred: {str(e)}")
        finally:
            await self.cleanup()

    async def handle_add(self, path: str) -> None:
        """Handle the add command to process and index documents."""
        try:
            result = await self.qa_system.add_documents(path)
            print(f"Successfully processed and indexed documents from: {path}")
            print(f"Added {result['processed']} documents")
            if result.get('skipped', 0) > 0:
                print(f"Skipped {result['skipped']} documents")
            if result.get('failed', 0) > 0:
                print(f"Failed to process {result['failed']} documents")
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
    
    async def handle_ask(self, question: str) -> None:
        """Handle a question from the user.
        
        Args:
            question: The question to answer
        """
        try:
            response = await self.qa_system.ask(question)
            print("\nAnswer:", response["answer"])
            
            if response.get("sources"):
                print("\nSources:")
                for source in response["sources"]:
                    print(f"- {source['filename']} (relevance: {source['relevance_score']:.2f})")
                    if source.get('excerpt'):
                        print(f"  Excerpt: {source['excerpt']}")
            
            if response.get("confidence"):
                print(f"\nConfidence: {response['confidence']:.2%}")
        except Exception as e:
            print(f"Error getting answer: {str(e)}")
    
    async def handle_list(self) -> None:
        """Handle the list command to show indexed documents."""
        try:
            documents = await self.qa_system.list_documents()
            if not documents:
                print("No documents indexed.")
                return
                
            print("\nIndexed Documents:")
            for doc in documents:
                # Get filename from path if filename is not available
                filename = doc.get('filename') or Path(doc.get('path', '')).name or 'Unknown'
                print(f"📄 {filename}")
                
                chunk_count = doc.get('chunk_count')
                if chunk_count:
                    print(f"   • Contains {chunk_count} pieces of information")
                    
                doc_id = doc.get('id', 'Unknown')
                print(f"   • ID: {doc_id}")
                
                file_type = doc.get('file_type', 'Unknown')
                print(f"   • Type: {file_type}")
                
                path = doc.get('path')
                if path:
                    print(f"   • Path: {path}")
                    
                print("-" * 40)
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
    
    async def handle_remove(self, doc_id: str) -> None:
        """Handle the remove command to delete documents."""
        try:
            await self.qa_system.remove_document(doc_id)
            print(f"Successfully removed document: {doc_id}")
        except Exception as e:
            print(f"Error removing document: {str(e)}")

def main():
    """Main entry point for the CLI."""
    cli = CLI(Path(parse_args().config_file))
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, cleaning up...")
        asyncio.run(cli.cleanup())
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        asyncio.run(cli.cleanup())
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 