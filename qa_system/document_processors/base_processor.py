"""
@file: base_processor.py
Base class for document processors providing common services such as logging, metadata extraction, chunking, and error handling.

This module defines the BaseDocumentProcessor class, which should be subclassed by specific document processors.
"""

import logging
from pathlib import Path
from datetime import datetime
from qa_system.exceptions import ProcessingError

class BaseDocumentProcessor:
    """
    Base class for document processors.

    Provides:
        - Logging
        - Metadata extraction
        - Sentence-aware chunking
        - Error handling hooks
        - Configurable chunking parameters

    Usage:
        Subclasses should implement process(file_path, metadata=None).
        Call run(file_path, metadata) to invoke with error handling.
    """
    def __init__(self, config):
        """
        Initialize the document processor with configuration.

        Args:
            config: Configuration object with chunking and processing parameters.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Called __init__(config={config})")
        self.chunk_size = self._get_config('DOCUMENT_PROCESSING.MAX_CHUNK_SIZE', 3072)
        self.min_chunk_size = self._get_config('DOCUMENT_PROCESSING.MIN_CHUNK_SIZE', 1024)
        self.chunk_overlap = self._get_config('DOCUMENT_PROCESSING.CHUNK_OVERLAP', 768)
        self.preserve_sentences = self._get_config('DOCUMENT_PROCESSING.PRESERVE_SENTENCES', True)

    def _get_config(self, key, default=None):
        """
        Retrieve a configuration value by key, supporting nested configs.

        Args:
            key (str): Configuration key.
            default: Default value if key is not found.
        Returns:
            The configuration value or default.
        """
        self.logger.info(f"Called _get_config(key={key}, default={default})")
        if hasattr(self.config, 'get_nested'):
            return self.config.get_nested(key, default)
        return getattr(self.config, key, default)

    def extract_metadata(self, file_path):
        """
        Extract basic file metadata from a file path.

        Args:
            file_path (str or Path): Path to the file.
        Returns:
            dict: Metadata including path, filename, type, and timestamps.
        """
        self.logger.info(f"Called extract_metadata(file_path={file_path})")
        path = Path(file_path)
        stat = path.stat()
        return {
            'path': str(path.resolve()),
            'relative_path': str(path),
            'directory': str(path.parent),
            'filename_full': path.name,
            'filename_stem': path.stem,
            'file_type': path.suffix.lstrip('.').lower(),
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def chunk_text(self, text):
        """
        Chunk text into sentence-aware segments.

        Args:
            text (str): The text to chunk.
        Returns:
            list[str]: List of text chunks.
        
        Implements a conditional chunking strategy: if the document is shorter than min_chunk_size, it is retained as a single chunk.
        """
        import re
        if len(text.strip()) <= self.min_chunk_size:
            return [text.strip()]
        # Split into sentences (basic, can be replaced by nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            if not sent:
                continue
            if current_len + len(sent) > self.chunk_size and current:
                chunks.append(' '.join(current))
                # Overlap
                if self.chunk_overlap > 0 and len(current) > 1:
                    overlap = []
                    overlap_len = 0
                    for s in reversed(current):
                        overlap.insert(0, s)
                        overlap_len += len(s)
                        if overlap_len >= self.chunk_overlap:
                            break
                    current = overlap
                    current_len = sum(len(s) for s in current)
                else:
                    current = []
                    current_len = 0
            current.append(sent)
            current_len += len(sent)
        if current:
            chunks.append(' '.join(current))
        # Remove tiny chunks unless it's the only chunk (i.e., short doc)
        return [c for c in chunks if len(c) >= self.min_chunk_size or len(chunks) == 1]

    def process(self, file_path, metadata=None):
        """
        Main entry point for processing a file. Should be implemented by subclasses.

        Args:
            file_path (str or Path): Path to the file to process.
            metadata (dict, optional): Additional metadata.
        Returns:
            Any: The result of processing.
        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        self.logger.info(f"Called process(file_path={file_path}, metadata={metadata})")
        raise NotImplementedError("Subclasses must implement process()")

    def run(self, file_path, metadata=None):
        """
        Run the processor with error handling and logging.

        Calls self.process(file_path, metadata) and wraps errors as ProcessingError.

        Args:
            file_path (str or Path): Path to the file to process.
            metadata (dict, optional): Additional metadata.
        Returns:
            Any: The result of processing.
        Raises:
            ProcessingError: If processing fails.
        """
        self.logger.info(f"Called run(file_path={file_path}, metadata={metadata})")
        try:
            return self.process(file_path, metadata)
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise ProcessingError(str(e)) 