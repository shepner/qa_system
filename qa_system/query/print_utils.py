"""
@file: print_utils.py
Utility functions for displaying query results in a user-friendly format.

This module provides functions to print QueryResponse objects and related output for the QA system.
"""

def print_response(response):
    """
    Print the answer and sources from a QueryResponse object in a user-friendly format.
    Args:
        response (QueryResponse): The response object to print.
    """
    print("\nAnswer:")
    print("-" * 80)
    print(response.text)
    print("\nSources:")
    print("-" * 80)
    for source in response.sources:
        print(f"- {source.document} (similarity: {source.similarity:.2f})") 