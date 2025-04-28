from typing import Dict, Any, Optional, List
import logging
import fitz  # PyMuPDF
import re
import tiktoken
from datetime import datetime
import os
from pathlib import Path
import asyncio
from .base_processor import BaseDocumentProcessor
from qa_system.exceptions import ProcessingError, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

class PDFDocumentProcessor(BaseDocumentProcessor):
    """PDF document processor for the QA system.

    This module provides functionality for processing PDF documents, extracting text
    and metadata, and handling errors appropriately.
    """
    
    # Maximum file size (100MB by default)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, config):
        """Initialize the PDF document processor.
        
        Args:
            config: Configuration object or path
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Get document processing configuration
        doc_processing = config.get_nested('DOCUMENT_PROCESSING', {})
        
        # Set chunk parameters
        self.max_chunk_size = doc_processing.get('MAX_CHUNK_SIZE', 3072)
        self.min_chunk_size = doc_processing.get('MIN_CHUNK_SIZE', 1024)
        self.chunk_overlap = doc_processing.get('CHUNK_OVERLAP', 768)
        self.preserve_sentences = doc_processing.get('PRESERVE_SENTENCES', True)
        
        # Get PDF-specific settings
        pdf_config = doc_processing.get('PDF_HEADER_RECOGNITION', {})
        self.header_recognition = pdf_config.get('ENABLED', True)
        self.min_font_size = pdf_config.get('MIN_FONT_SIZE', 12)
        self.header_patterns = pdf_config.get('PATTERNS', [])
        self.max_header_length = pdf_config.get('MAX_HEADER_LENGTH', 100)
        
        # Validate configuration
        if self.max_chunk_size < 1:
            raise ValueError(f"max_chunk_size must be positive, got {self.max_chunk_size}")
        if self.min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than max_chunk_size ({self.max_chunk_size})")
        if not isinstance(self.header_patterns, list):
            raise ValueError(f"header_patterns must be a list, got {type(self.header_patterns)}")
        
        # Initialize tiktoken encoder for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Store workspace root for relative path calculation
        self.workspace_root = doc_processing.get('DOCUMENT_PATH', os.getcwd())
        
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF file and extract text and metadata.
        
        Args:
            file_path: Path to the PDF file
            metadata: Dictionary containing metadata about the file
            
        Returns:
            Dictionary containing metadata and extracted text chunks
            
        Raises:
            ProcessingError: If processing fails
            ValidationError: If file validation fails
        """
        self.logger.debug(f"Processing PDF file: {file_path}")
        
        try:
            # Validate file size
            self._check_file_size(file_path)
            
            # Enhance metadata with standard fields
            metadata = self._enhance_metadata(file_path, metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Process PDF file
            pdf_document = None
            try:
                pdf_document = fitz.open(file_path)
                
                # Extract document metadata
                pdf_metadata = pdf_document.metadata
                metadata.update({
                    'pdf_title': pdf_metadata.get('title', ''),
                    'pdf_author': pdf_metadata.get('author', ''),
                    'pdf_subject': pdf_metadata.get('subject', ''),
                    'pdf_keywords': pdf_metadata.get('keywords', ''),
                    'pdf_creator': pdf_metadata.get('creator', ''),
                    'pdf_producer': pdf_metadata.get('producer', ''),
                    'pdf_creation_date': pdf_metadata.get('creationDate', ''),
                    'pdf_modification_date': pdf_metadata.get('modDate', '')
                })
                
                # Extract text from each page
                text_chunks = []
                toc = []
                
                for page_num in range(len(pdf_document)):
                    if self._should_stop():
                        self.logger.warning("Processing interrupted")
                        break
                        
                    page = pdf_document[page_num]
                    
                    # Extract text with page context
                    text = page.get_text()
                    if text.strip():
                        chunk = f"Page {page_num + 1}:\n{text}"
                        text_chunks.append(chunk)
                    
                    # Extract links for table of contents
                    links = page.get_links()
                    for link in links:
                        if 'uri' in link:
                            toc.append({
                                'page': page_num + 1,
                                'uri': link['uri']
                            })
                
                # Calculate total tokens
                total_tokens = sum(self._count_tokens(chunk) for chunk in text_chunks)
                
                # Update processing metadata
                metadata.update({
                    'processor_type': 'pdf',
                    'total_tokens': total_tokens,
                    'chunk_count': len(text_chunks),
                    'average_chunk_size': total_tokens / len(text_chunks) if text_chunks else 0,
                    'content_type': 'application/pdf',
                    'page_count': len(pdf_document),
                    'table_of_contents': toc
                })
                
                return {
                    'metadata': metadata,
                    'chunks': text_chunks
                }
                
            except fitz.FileDataError as e:
                raise ValidationError(f"Invalid PDF file: {str(e)}")
            except Exception as e:
                raise ProcessingError(f"Failed to process PDF: {str(e)}")
            finally:
                if pdf_document:
                    try:
                        pdf_document.close()
                    except Exception as e:
                        self.logger.warning(f"Failed to close PDF document: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            raise
    
    def _should_stop(self) -> bool:
        """Check if processing should be stopped."""
        return False  # Override in subclass if needed

    def _enhance_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the metadata dictionary with additional information."""
        # Add file path and timestamp
        metadata.update({
            'file_path': file_path,
            'timestamp': datetime.now().isoformat()
        })
        return metadata

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate the metadata dictionary."""
        # Add file path and timestamp
        metadata.update({
            'file_path': metadata['file_path'],
            'timestamp': metadata['timestamp']
        })

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text."""
        return len(self.tokenizer.encode(text))

    def _chunk_text_with_sentences(self, text: str) -> List[str]:
        """Chunk the text into sentences."""
        sentences = re.split(r'(?<=[。！？])', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            if current_tokens + len(sentence) > self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = len(sentence)
            else:
                current_chunk += sentence
                current_tokens += len(sentence)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks 