"""Text document processor for the QA system.

This module provides functionality for processing text files, including encoding
detection, memory-efficient processing, and proper error handling.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Iterator
import chardet
from tenacity import retry, stop_after_attempt, wait_exponential

from qa_system.document_processors.base_processor import BaseDocumentProcessor
from qa_system.exceptions import ProcessingError, ValidationError

class TextDocumentProcessor(BaseDocumentProcessor):
    """Processor for handling text files."""
    
    # Maximum file size (100MB by default)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Sample size for encoding detection (64KB)
    SAMPLE_SIZE = 64 * 1024
    
    # Chunk size for reading large files (1MB)
    CHUNK_SIZE = 1024 * 1024
    
    def __init__(self):
        """Initialize the text document processor."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def _check_file_size(self, file_path: str) -> None:
        """Check if file size is within limits.
        
        Args:
            file_path: Path to the file to check
            
        Raises:
            ValidationError: If file size exceeds limit
        """
        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_SIZE:
            raise ValidationError(
                f"File size {file_size} bytes exceeds limit of {self.MAX_FILE_SIZE} bytes"
            )
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding (defaults to utf-8)
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(self.SAMPLE_SIZE)
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def _read_text_chunks(self, file_path: str, encoding: str) -> Iterator[str]:
        """Read text file in chunks to handle large files efficiently.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Yields:
            Text chunks
        """
        with open(file_path, 'r', encoding=encoding) as f:
            while True:
                chunk = f.read(self.CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text file and extract its content and metadata.
        
        Args:
            file_path: Path to the text file
            metadata: Dictionary containing metadata about the file
            
        Returns:
            Dictionary containing metadata and extracted text chunks
            
        Raises:
            ProcessingError: If processing fails
            ValidationError: If file validation fails
        """
        self.logger.debug(f"Processing text file: {file_path}")
        
        try:
            # Validate file size
            self._check_file_size(file_path)
            
            # Enhance metadata with standard fields
            metadata = self._enhance_metadata(metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Detect file encoding
            encoding = self._detect_encoding(file_path)
            
            # Process text in chunks
            text_chunks = []
            total_size = 0
            
            try:
                current_chunk = []
                current_size = 0
                
                for text_chunk in self._read_text_chunks(file_path, encoding):
                    if self._should_stop():
                        self.logger.warning("Processing interrupted")
                        break
                    
                    # Split chunk into sentences
                    sentences = self._split_into_sentences(text_chunk)
                    
                    for sentence in sentences:
                        sentence_size = len(sentence)
                        
                        # If adding this sentence would exceed chunk size, start new chunk
                        if current_size + sentence_size > self.CHUNK_SIZE:
                            if current_chunk:
                                text_chunks.append("\n".join(current_chunk))
                            current_chunk = [sentence]
                            current_size = sentence_size
                        else:
                            current_chunk.append(sentence)
                            current_size += sentence_size
                        
                        total_size += sentence_size
                
                # Add final chunk if any
                if current_chunk:
                    text_chunks.append("\n".join(current_chunk))
                
                # Calculate total tokens
                total_tokens = sum(self._count_tokens(chunk) for chunk in text_chunks)
                
                # Update processing metadata
                metadata.update({
                    'processor_type': 'text',
                    'total_tokens': total_tokens,
                    'chunk_count': len(text_chunks),
                    'average_chunk_size': total_tokens / len(text_chunks) if text_chunks else 0,
                    'content_type': 'text/plain',
                    'encoding': encoding,
                    'file_size': total_size,
                    'sentence_count': sum(len(self._split_into_sentences(chunk)) for chunk in text_chunks)
                })
                
                return {
                    'metadata': metadata,
                    'chunks': text_chunks
                }
                
            except UnicodeDecodeError as e:
                raise ProcessingError(f"Failed to decode text with encoding {encoding}: {str(e)}")
            except Exception as e:
                raise ProcessingError(f"Failed to process text: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    
    def _should_stop(self) -> bool:
        """Check if processing should be stopped."""
        return False  # Override in subclass if needed
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving formatting.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentences = []
        current = []
        
        for line in text.split('\n'):
            if not line.strip():
                if current:
                    sentences.append('\n'.join(current))
                    current = []
                sentences.append('')
            else:
                current.append(line)
        
        if current:
            sentences.append('\n'.join(current))
        
        return sentences 