"""
QA System - A lean and efficient question answering system.
"""

from embed_files.vector_store import VectorStore
from embed_files.document_processor import DocumentProcessor
from embed_files.embeddings import EmbeddingModel

__version__ = "0.1.0"
__all__ = ["VectorStore", "DocumentProcessor", "EmbeddingModel"] 