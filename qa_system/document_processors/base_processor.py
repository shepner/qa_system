"""Base class for document processors providing common services."""

import logging
from pathlib import Path
from datetime import datetime
from qa_system.logging_setup import setup_logging
from qa_system.exceptions import ProcessingError

class BaseDocumentProcessor:
    """
    Base class for document processors. Provides:
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
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called __init__(config={config})")
        # setup_logging()  # Do not configure logging here; let main do it
        self.chunk_size = self._get_config('DOCUMENT_PROCESSING.MAX_CHUNK_SIZE', 3072)
        self.min_chunk_size = self._get_config('DOCUMENT_PROCESSING.MIN_CHUNK_SIZE', 1024)
        self.chunk_overlap = self._get_config('DOCUMENT_PROCESSING.CHUNK_OVERLAP', 768)
        self.preserve_sentences = self._get_config('DOCUMENT_PROCESSING.PRESERVE_SENTENCES', True)

    def _get_config(self, key, default=None):
        self.logger.debug(f"Called _get_config(key={key}, default={default})")
        if hasattr(self.config, 'get_nested'):
            return self.config.get_nested(key, default)
        return getattr(self.config, key, default)

    def extract_metadata(self, file_path):
        self.logger.debug(f"Called extract_metadata(file_path={file_path})")
        """Extract basic file metadata."""
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
        self.logger.debug(f"Called chunk_text(text=<len {len(text)}>)")
        """
        Chunk text into sentence-aware segments.
        Returns a list of text chunks.
        """
        import re
        # Split into sentences (very basic, can be replaced by nltk or spacy)
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
        # Remove tiny chunks
        return [c for c in chunks if len(c) >= self.min_chunk_size or len(chunks) == 1]

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Called process(file_path={file_path}, metadata={metadata})")
        """
        Main entry point for processing a file. Should be implemented by subclasses.
        Do not call directly; use run() to invoke with error handling.
        """
        raise NotImplementedError("Subclasses must implement process()")

    def run(self, file_path, metadata=None):
        self.logger.debug(f"Called run(file_path={file_path}, metadata={metadata})")
        """
        Run the processor with error handling and logging.
        Calls self.process(file_path, metadata) and wraps errors as ProcessingError.
        """
        try:
            return self.process(file_path, metadata)
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise ProcessingError(str(e)) 