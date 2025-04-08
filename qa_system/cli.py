"""
Command-line interface for the QA system
"""
import asyncio
import argparse
from typing import List
import sys
from pathlib import Path

from .core import QASystem
from .remove_excluded import remove_excluded_documents

async def interactive_mode(qa: QASystem):
    """Run the QA system in interactive mode."""
    print("\nHi! I'm your QA assistant. I'm here to help answer your questions and manage your documents.")
    print("You can chat with me naturally or use these commands:")
    print("  add <path>         - I'll add new documents to my knowledge base")
    print("  list              - I'll show you what documents I know about")
    print("  remove <doc_id>    - I'll remove a document you specify")
    print("  cleanup           - I'll clean up any excluded documents")
    print("  help              - I'll show you this help message again")
    print("  exit              - Say goodbye and end our conversation")
    
    while True:
        try:
            # Get user input
            user_input = input("\nWhat can I help you with? ").strip()
            
            if not user_input:
                continue
                
            # Split into command and arguments
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "exit":
                print("Thanks for chatting! Have a great day!")
                break
                
            elif command == "help":
                print("\nHere's what I can do for you:")
                print("  add <path>         - I'll add new documents to my knowledge base")
                print("  list              - I'll show you what documents I know about")
                print("  remove <doc_id>    - I'll remove a document you specify")
                print("  cleanup           - I'll clean up any excluded documents")
                print("  help              - I'll show you this help message again")
                print("  exit              - Say goodbye and end our conversation")
                
            elif command == "add":
                if not args:
                    print("Could you please provide a path to the documents you'd like me to add?")
                    continue
                    
                print(f"I'll process the documents from {args} for you...")
                stats = await qa.add_documents(args)
                print(f"\nAll done! Here's what I did:")
                print(f"✓ Successfully processed {stats['processed']} files")
                if stats['failed'] > 0:
                    print(f"✗ Had trouble with {stats['failed']} files")
                print(f"📚 Added {stats['chunks']} chunks of information to my knowledge base")
                
            elif command == "list":
                docs = qa.list_documents()
                if not docs:
                    print("I don't have any documents in my knowledge base yet. Would you like to add some using the 'add' command?")
                else:
                    print("\nHere are the documents I know about:")
                    for doc in docs:
                        print(f"📄 {doc['filename']}")
                        print(f"   • Contains {doc['chunk_count']} pieces of information")
                        print(f"   • ID: {doc['id']}")
                        print(f"   • Type: {doc['file_type']}")
                        
            elif command == "remove":
                if not args:
                    print("Could you please provide the ID of the document you'd like me to remove?")
                    continue
                    
                success = await qa.remove_document(args)
                if success:
                    print(f"I've successfully removed the document with ID {args}.")
                else:
                    print(f"I couldn't remove the document with ID {args}. Are you sure that's the correct ID?")
                    
            elif command == "cleanup":
                print("I'll remove any documents that match the exclude patterns...")
                await remove_excluded_documents()
                print("All cleaned up!")
                    
            else:
                # Treat any unrecognized input as a question
                print("Let me think about that...")
                response = await qa.ask(user_input)
                
                print("\nHere's what I found:")
                print(response["answer"])
                
                if response["sources"]:
                    print("\nI found this information in:")
                    for source in response["sources"]:
                        confidence = source['relevance_score'] * 100
                        print(f"📚 {source['filename']} (Confidence: {confidence:.0f}%)")
                
                confidence = response["confidence"] * 100
                if confidence >= 90:
                    print(f"\nI'm very confident about this answer ({confidence:.0f}%)")
                elif confidence >= 70:
                    print(f"\nI'm fairly confident about this answer ({confidence:.0f}%)")
                else:
                    print(f"\nI'm not entirely sure about this answer ({confidence:.0f}%). You might want to verify it.")
                
        except KeyboardInterrupt:
            print("\nThanks for chatting! Have a great day!")
            break
        except Exception as e:
            print(f"Oops! Something went wrong: {str(e)}")
            print("Could you try that again or rephrase your request?")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local File Question-Answering System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add documents
    add_parser = subparsers.add_parser("add", help="Add documents to the system")
    add_parser.add_argument(
        "path",
        help="Path to document or directory"
    )
    
    # Ask question
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument(
        "question",
        help="Question to ask"
    )
    
    # List documents
    list_parser = subparsers.add_parser("list-docs", help="List indexed documents")
    
    # Remove document
    remove_parser = subparsers.add_parser("remove", help="Remove a document")
    remove_parser.add_argument(
        "doc_id",
        help="Document ID to remove"
    )
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove documents matching exclude patterns")
    
    return parser.parse_args()

async def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Initialize QA system
    qa = QASystem()
    
    try:
        if not args.command:
            # No command provided, enter interactive mode
            await interactive_mode(qa)
            return
            
        if args.command == "add":
            stats = await qa.add_documents(args.path)
            print(f"Processing complete:")
            print(f"- Processed: {stats['processed']} files")
            print(f"- Failed: {stats['failed']} files")
            print(f"- Total chunks: {stats['chunks']}")
            
        elif args.command == "ask":
            response = await qa.ask(args.question)
            print("\nAnswer:")
            print(response["answer"])
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source['filename']} (Relevance: {source['relevance_score']:.2f})")
            print(f"\nConfidence: {response['confidence']:.2f}")
            
        elif args.command == "list-docs":
            docs = qa.list_documents()
            if not docs:
                print("No documents indexed.")
            else:
                print("\nIndexed Documents:")
                for doc in docs:
                    print(f"- {doc['filename']} ({doc['chunk_count']} chunks)")
                    print(f"  ID: {doc['id']}")
                    print(f"  Type: {doc['file_type']}")
                    
        elif args.command == "remove":
            success = await qa.remove_document(args.doc_id)
            if success:
                print(f"Document {args.doc_id} removed successfully.")
            else:
                print(f"Failed to remove document {args.doc_id}.")
                
        elif args.command == "cleanup":
            print("Removing documents that match exclude patterns...")
            await remove_excluded_documents()
            print("Cleanup complete.")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def run():
    """Entry point for the CLI tool."""
    asyncio.run(main()) 