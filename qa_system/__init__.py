"""
Local File Question-Answering System using Google Gemini
"""

__version__ = "1.0.0"

from .core import QASystem
from .document_processor import DocumentProcessor
from .query_engine import QueryEngine
from .vector_store import VectorStore

__all__ = ["QASystem", "DocumentProcessor", "QueryEngine", "VectorStore"] 