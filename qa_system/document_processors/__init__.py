"""
@file: __init__.py
Document processor factory and list handler for QA System.

This module provides a factory function to select the appropriate document processor
based on file type, and a ListHandler class for listing document metadata.

Exports:
    - get_processor_for_file_type: Returns a processor instance for a given file type.
    - ListHandler: Class for listing document metadata.
"""

from qa_system.file_scanner import FileScanner
import logging
from .text_processor import TextDocumentProcessor
from .markdown_processor import MarkdownDocumentProcessor
from .pdf_processor import PDFDocumentProcessor
from .csv_processor import CSVDocumentProcessor
from .vision_processor import VisionDocumentProcessor

def get_processor_for_file_type(path, config):
    """
    Return the appropriate document processor instance for the given file type.

    Args:
        path: Path to the document file (str or Path).
        config: Configuration object for the processor.

    Returns:
        An instance of a document processor class suitable for the file type.
        If the file type is unsupported, returns a dummy processor.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Called get_processor_for_file_type(path={path}, config={config})")
    ext = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    if ext == 'txt':
        return TextDocumentProcessor(config)
    if ext == 'md':
        return MarkdownDocumentProcessor(config)
    if ext == 'pdf':
        return PDFDocumentProcessor(config)
    if ext == 'csv':
        return CSVDocumentProcessor(config)
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        return VisionDocumentProcessor(config)
    class DummyProcessor:
        """Fallback processor for unsupported file types."""
        def process(self):
            logger.info("Called DummyProcessor.process()")
            return {'chunks': [], 'metadata': {}}
    return DummyProcessor()

class ListHandler:
    """
    Handler for listing document metadata.
    """
    def __init__(self, config):
        """
        Initialize the ListHandler.

        Args:
            config: Configuration object for the handler.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Called ListHandler.__init__(config={config})")
    def list_metadata(self, filter_pattern=None):
        """
        List document metadata, optionally filtered by a pattern.

        Args:
            filter_pattern: Optional pattern to filter metadata (default: None).

        Returns:
            List of metadata entries (currently always empty).
        """
        self.logger.info(f"Called ListHandler.list_metadata(filter_pattern={filter_pattern})")
        return []
