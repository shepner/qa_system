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
from .image_processor import ImageDocumentProcessor

def get_processor_for_file_type(path, config, query_processor=None):
    """
    Return the appropriate document processor instance for the given file type.

    Args:
        path: Path to the document file (str or Path).
        config: Configuration object for the processor.
        query_processor: Query processor object to be passed to processors that need it (optional).

    Returns:
        An instance of a document processor class suitable for the file type.
        If the file type is unsupported, returns a dummy processor.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Called get_processor_for_file_type(path={path}, config={config}, query_processor={query_processor})")
    ext = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    if ext == 'txt':
        return TextDocumentProcessor(config)
    if ext == 'md':
        return MarkdownDocumentProcessor(config)
    if ext == 'pdf':
        return PDFDocumentProcessor(config, query_processor=query_processor)
    if ext == 'csv':
        return CSVDocumentProcessor(config)
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        if query_processor is not None:
            logger.debug("Passing query_processor to ImageDocumentProcessor.")
            return ImageDocumentProcessor(query_processor=query_processor)
        else:
            logger.warning("No query_processor provided for image file. Skipping image processing.")
            class SkippedImageProcessor:
                def process(self, file_path):
                    logger.warning(f"Skipping image file {file_path} due to missing query_processor.")
                    return {'chunks': [], 'metadata': {'file_path': file_path, 'skipped': True, 'skip_reason': 'query_processor not available'}}
            return SkippedImageProcessor()
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
