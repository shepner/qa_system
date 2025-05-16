"""
@file: __init__.py
Query module public API for the QA system.

This module exposes the main query-related classes and functions for external use, including:
    - Source, QueryResponse: Data models for query results and sources
    - QueryProcessor: Main class for processing semantic queries
    - apply_scoring: Utility for scoring sources

All other internal utilities are kept private to the module.
"""
from .models import Source, QueryResponse
from .processor import QueryProcessor
from .scoring import apply_scoring
from .print_utils import print_response
