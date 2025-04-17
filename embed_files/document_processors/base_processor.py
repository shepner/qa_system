from typing import Dict, Any, List, Union
import logging
import re
import os
from abc import ABC, abstractmethod
import tiktoken
from datetime import datetime
from pathlib import Path

class BaseDocumentProcessor(ABC):
    """Base class for document processors.
    
    This abstract class defines the interface and common functionality for all document processors.
    Each specific document type (PDF, TXT, etc.) should implement this interface.
    
    The processor handles:
    - Document text extraction
    - Metadata extraction and validation
    - Text chunking with configurable size and overlap
    - Token counting using tiktoken
    - Error handling and logging
    """
    
    def __init__(self) -> None:
        """Initialize the base document processor."""
        self.logger = logging.getLogger(__name__)
        # Initialize tiktoken encoder for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    @abstractmethod
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and extract its content and metadata.
        
        Args:
            file_path: Path to the document file
            metadata: Initial metadata dictionary to be enhanced
            
        Returns:
            Dict containing:
            - Document metadata (type, timestamps, etc.)
            - Extracted text chunks with position information
            - Document-specific metadata
            - Token counts per chunk and total
            
        Raises:
            ValueError: If file format is invalid or metadata is incomplete
            IOError: If file cannot be read
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate the required metadata fields are present.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = {
            'path',
            'relative_path',
            'directory',
            'filename_full',
            'filename_stem',
            'file_type',
            'created_at',
            'last_modified'
        }
        
        missing_fields = required_fields - set(metadata.keys())
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {missing_fields}")
            
    def _enhance_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata with standard fields.
        
        Args:
            file_path: Path to the document file
            metadata: Initial metadata dictionary
            
        Returns:
            Enhanced metadata dictionary
        """
        path = Path(file_path)
        workspace_root = os.getenv('WORKSPACE_ROOT', os.getcwd())
        
        # Update metadata with standard fields
        metadata.update({
            'path': str(path.absolute()),
            'relative_path': str(path.relative_to(workspace_root)),
            'directory': str(path.parent),
            'filename_full': path.name,
            'filename_stem': path.stem,
            'file_type': path.suffix.lstrip('.').lower(),
            'created_at': datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            'processor_type': self.__class__.__name__
        })
        
        return metadata
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
        
    def _chunk_text_with_sentences(
        self, 
        text: str, 
        max_chunk_size: int = 2000, 
        overlap: int = 200
    ) -> List[Dict[str, Union[int, str]]]:
        """Split text into overlapping chunks while preserving sentence boundaries.
        
        Args:
            text: UTF-8 encoded text content to chunk
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing:
            - chunk_number: Sequential number of the chunk
            - start_pos: Starting character position in original text
            - end_pos: Ending character position in original text
            - content: The chunk text content
            - token_count: Number of tokens in the chunk
            
        Raises:
            ValueError: If max_chunk_size <= 0 or overlap < 0 or overlap >= max_chunk_size
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for chunking")
            return []
            
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_chunk_size:
            raise ValueError("overlap must be less than max_chunk_size")
            
        # Ensure minimum effective chunk size
        min_chunk_size = 100  # Arbitrary minimum to prevent tiny chunks
        if max_chunk_size - overlap < min_chunk_size:
            raise ValueError(f"Effective chunk size (max_chunk_size - overlap) must be at least {min_chunk_size} characters")
            
        # Normalize whitespace and split into sentences
        text = ' '.join(text.split())
        # More comprehensive sentence boundary pattern
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.!?])\s*$')
        
        chunks = []
        chunk_number = 1
        current_pos = 0
        total_tokens = 0
        
        while current_pos < len(text):
            # Calculate the maximum possible end position for this chunk
            end_pos = min(current_pos + max_chunk_size, len(text))
            
            # If we're not at the end of the text, try to break at a sentence boundary
            if end_pos < len(text):
                # Search for the last sentence boundary within the chunk
                matches = list(sentence_pattern.finditer(text[current_pos:end_pos]))
                if matches:
                    # Use the last complete sentence boundary
                    last_match = matches[-1]
                    end_pos = current_pos + last_match.end()
                else:
                    # If no sentence boundary found, try to break at last space
                    last_space = text[current_pos:end_pos].rfind(' ')
                    if last_space > min_chunk_size:  # Only break at space if resulting chunk is large enough
                        end_pos = current_pos + last_space
            
            chunk_text = text[current_pos:end_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                token_count = self._count_tokens(chunk_text)
                total_tokens += token_count
                chunks.append({
                    'chunk_number': chunk_number,
                    'start_pos': current_pos,
                    'end_pos': end_pos,
                    'content': chunk_text,
                    'token_count': token_count
                })
                chunk_number += 1
            
            # Move to next chunk position, ensuring minimum chunk size
            next_pos = end_pos - overlap
            if next_pos <= current_pos:  # Prevent getting stuck
                next_pos = current_pos + min_chunk_size
            current_pos = next_pos
        
        # Update metadata with token counts
        if chunks:
            chunks[-1]['total_tokens'] = total_tokens
        
        return chunks 