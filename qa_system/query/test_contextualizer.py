"""
Test script for contextualizer.py

Runs the contextualizer to generate context from the source file and prints the output file contents.
"""

from qa_system.config import get_config
from qa_system.query.contextualizer import generate_contextual_data


def main():
    # Run the contextualizer
    generate_contextual_data()
    # Get the output file path from config
    config = get_config()
    user_context_file = config.get_nested('QUERY.USER_CONTEXT_FILE')
    print(f"\n--- Contents of {user_context_file} ---\n")
    with open(user_context_file, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == "__main__":
    main() 