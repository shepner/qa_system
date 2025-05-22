#!/usr/bin/env python3
"""
Run the UserInteractionProcessor to generate contextual summaries from user interaction logs.

- Reads all .md files in USER_INTERACTION_DIRECTORY (from config)
- Writes context files to DOCUMENT_PROCESSING.USER_CONTEXT_DIRECTORY (from config)
- No command-line arguments required; all configuration is read from config/config.yaml
- Prints a summary of processed files and output locations
"""
import sys
import os
# Load .env from ./secrets/.env before anything else
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'secrets', '.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from qa_system.document_processors.user_interaction_processor import UserInteractionProcessor

def main():
    logging.basicConfig(level=logging.DEBUG)
    # Explicitly set Gemini LLM logger to DEBUG
    logging.getLogger("qa_system.query.gemini_llm").setLevel(logging.DEBUG)
    try:
        processor = UserInteractionProcessor()
        print(f"Input directory:   {processor.input_dir}")
        print(f"Output directory:  {processor.output_dir}")
        all_files = processor._get_all_files() if hasattr(processor, '_get_all_files') else None
        if all_files is None:
            import glob, os
            all_files = sorted(glob.glob(os.path.join(processor.input_dir, '*.md')))
        print(f"Found {len(all_files)} input file(s).\n")
        processor.process()
        print(f"\nProcessed {len(all_files)} file(s). Context files written to: {processor.output_dir}")
        print("User interaction context generation complete.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 