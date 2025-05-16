"""
@file: models.py
Data models for query responses and source attribution in the QA system.

This module defines the core data structures used to represent the results of a query,
including the main response object and the structure for individual sources (documents/chunks)
that contributed to the answer. These models are used throughout the query pipeline to
standardize the representation of results, confidence, provenance, and error handling.

Classes:
    Source: Represents a single source (document chunk) contributing to a query result.
    QueryResponse: Represents the full response to a query, including answer text, sources, and metadata.
"""
from typing import Optional

class Source:
    """
    Represents a single source (document chunk) that contributed to a query result.

    Attributes:
        document (str): The name or identifier of the source document.
        chunk (str): The specific chunk or excerpt from the document.
        similarity (float): Similarity score between the query and this chunk.
        context (str): Additional context or explanation for this chunk (optional).
        metadata (dict): Arbitrary metadata about the source (optional).
        original_similarity (float): The original similarity score before any adjustments (optional).
        boost (float): Any boost factor applied to this source (optional).
    """
    def __init__(self, document: str, chunk: str, similarity: float, context: str = '', metadata: Optional[dict] = None, original_similarity: float = None, boost: float = None):
        """
        Initialize a Source object.

        Args:
            document (str): The name or identifier of the source document.
            chunk (str): The specific chunk or excerpt from the document.
            similarity (float): Similarity score between the query and this chunk.
            context (str, optional): Additional context or explanation for this chunk. Defaults to ''.
            metadata (dict, optional): Arbitrary metadata about the source. Defaults to None.
            original_similarity (float, optional): The original similarity score before any adjustments. Defaults to None.
            boost (float, optional): Any boost factor applied to this source. Defaults to None.
        """
        self.document = document
        self.chunk = chunk
        self.similarity = similarity
        self.context = context
        self.metadata = metadata or {}
        self.original_similarity = original_similarity
        self.boost = boost

class QueryResponse:
    """
    Represents the full response to a query, including answer text, sources, and metadata.

    Attributes:
        text (str): The answer or response text generated for the query.
        sources (list): List of Source objects that contributed to the answer.
        confidence (float): Confidence score for the answer (default: 1.0).
        processing_time (float): Time taken to process the query, in seconds (default: 0.0).
        error (str): Error message if the query failed (optional).
        success (bool): Whether the query was successful (default: True).
    """
    def __init__(self, text: str, sources: list, confidence: float = 1.0, processing_time: float = 0.0, error: Optional[str] = None, success: bool = True):
        """
        Initialize a QueryResponse object.

        Args:
            text (str): The answer or response text generated for the query.
            sources (list): List of Source objects that contributed to the answer.
            confidence (float, optional): Confidence score for the answer. Defaults to 1.0.
            processing_time (float, optional): Time taken to process the query, in seconds. Defaults to 0.0.
            error (str, optional): Error message if the query failed. Defaults to None.
            success (bool, optional): Whether the query was successful. Defaults to True.
        """
        self.text = text
        self.sources = sources
        self.confidence = confidence
        self.processing_time = processing_time
        self.error = error
        self.success = success 