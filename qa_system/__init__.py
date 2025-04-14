"""
QA System - A lean and efficient question answering system.
"""

from qa_system.vector_store import VectorStore
from qa_system.document_processor import DocumentProcessor
from qa_system.embeddings import EmbeddingModel

__version__ = "0.1.0"
__all__ = ["VectorStore", "DocumentProcessor", "EmbeddingModel"] 