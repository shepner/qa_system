#!/usr/bin/env python3
"""
Sample PDF chunking results for inspection and tuning.

Usage:
    python sample_pdf_chunks.py <filename.pdf> [--show-metadata] [--num-samples N]

Prints representative samples of the chunks produced by the PDF chunking logic.
"""
import sys
import argparse
from pathlib import Path

# Import config and PDF processor from qa_system
sys.path.insert(0, str(Path(__file__).parent.parent))  # Ensure project root is in sys.path
from qa_system.config import get_config
from qa_system.document_processors.pdf_processor import PDFDocumentProcessor


def print_chunk(chunk, idx):
    print(f"\n=== Chunk {idx} ===")
    print(chunk['text'][:1000])
    if len(chunk['text']) > 1000:
        print("... [truncated]")
    print("\n---")


def main():
    parser = argparse.ArgumentParser(description="Show logical Markdown-like chunks from a PDF.")
    parser.add_argument("filename", help="Path to PDF file")
    args = parser.parse_args()

    config = get_config()
    processor = PDFDocumentProcessor(config)
    result = processor.process(args.filename)
    chunks = result['chunks']
    # Print the entire Markdown-formatted document as a single string
    full_markdown = "\n\n".join(chunk['text'] for chunk in chunks)
    print(full_markdown)


if __name__ == "__main__":
    main() 