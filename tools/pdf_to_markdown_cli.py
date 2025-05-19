#!/usr/bin/env python3
import sys
import os
import argparse

# Ensure qa_system/query is in the path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qa_system.query.pdf_to_markdown import pdf_to_markdown

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown with images (tool version).")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument("--output_dir", default="./tmp", help="Directory to save images (default: ./tmp)")
    parser.add_argument("--output_md", default=None, help="Optional: Path to save Markdown output.")
    args = parser.parse_args()

    md = pdf_to_markdown(args.pdf_path, args.output_dir)
    if args.output_md:
        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Markdown written to {args.output_md}")
    else:
        print(md)

if __name__ == "__main__":
    main() 