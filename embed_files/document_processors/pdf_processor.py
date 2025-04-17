from typing import Dict, Any, Optional, List
import logging
import fitz  # PyMuPDF
import re
import tiktoken
from datetime import datetime
import os
from pathlib import Path
from .base_processor import BaseDocumentProcessor

class PDFDocumentProcessor(BaseDocumentProcessor):
    """PDF document processor implementation.
    
    Handles extraction and processing of PDF documents using PyMuPDF (fitz).
    Implements the BaseDocumentProcessor interface with PDF-specific functionality.
    Features:
    - Text extraction with page preservation
    - Header pattern recognition
    - Intelligent chunking based on section boundaries
    - PDF-specific metadata extraction
    - Token counting using tiktoken
    """
    
    def __init__(self, config):
        """Initialize the PDF document processor.
        
        Args:
            config: Either a dictionary or Config object containing configuration settings.
                   Must contain DOCUMENT_PROCESSING section with required parameters.
        
        Raises:
            ValueError: If config is invalid or required parameters are not met
        """
        super().__init__()
        
        if config is None:
            raise ValueError("Config cannot be None")
            
        # First determine the type of config we received
        if hasattr(config, 'get_nested'):
            # Config object - extract document processing settings
            doc_processing = config.get_nested('DOCUMENT_PROCESSING', {})
            if not isinstance(doc_processing, dict):
                raise ValueError("DOCUMENT_PROCESSING section from Config object must be a dictionary")
        elif isinstance(config, dict):
            # Dictionary config - either direct settings or nested under DOCUMENT_PROCESSING
            if 'DOCUMENT_PROCESSING' in config:
                doc_processing = config['DOCUMENT_PROCESSING']
                if not isinstance(doc_processing, dict):
                    raise ValueError("DOCUMENT_PROCESSING section must be a dictionary")
            else:
                # Assume direct settings
                doc_processing = config
        else:
            raise ValueError(f"Config must be either a dictionary or a Config object with get_nested method, got {type(config)}")

        # Extract and validate settings with defaults
        self.max_chunk_size = doc_processing.get('MAX_CHUNK_SIZE', 1500)
        self.chunk_overlap = doc_processing.get('CHUNK_OVERLAP', 300)
        self.header_patterns = []
        
        # Extract header patterns if available
        pdf_header_config = doc_processing.get('PDF_HEADER_RECOGNITION', {})
        if isinstance(pdf_header_config, dict) and pdf_header_config.get('ENABLED', True):
            self.header_patterns = pdf_header_config.get('PATTERNS', [])

        # Validate configuration parameters
        if not isinstance(self.max_chunk_size, int) or self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be a positive integer, got {self.max_chunk_size}")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than max_chunk_size ({self.max_chunk_size})")
        if not isinstance(self.header_patterns, list):
            raise ValueError(f"header_patterns must be a list, got {type(self.header_patterns)}")
        
        # Initialize tiktoken encoder for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Store workspace root for relative path calculation
        self.workspace_root = doc_processing.get('DOCUMENT_PATH', os.getcwd())
        
        self.logger = logging.getLogger(__name__)
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF document and extract its content and metadata.
        
        Args:
            file_path: Path to the PDF file
            metadata: Initial metadata dictionary
            
        Returns:
            Dict containing:
            - Document metadata (type, timestamps, etc.)
            - Extracted text chunks
            - PDF-specific metadata (title, author, etc.)
            - Token counts
            - Headers and structure information
            
        Raises:
            ValueError: If file is not a valid PDF or metadata is invalid
            IOError: If file cannot be read
        """
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
            
        try:
            # Enhance metadata with required fields
            path_obj = Path(file_path)
            relative_path = str(path_obj.relative_to(self.workspace_root))
            
            metadata.update({
                "path": str(path_obj.absolute()),
                "relative_path": relative_path,
                "directory": str(path_obj.parent),
                "filename_full": path_obj.name,
                "filename_stem": path_obj.stem,
                "file_type": "pdf",
                "created_at": datetime.fromtimestamp(path_obj.stat().st_ctime).isoformat(),
                "last_modified": datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
            })
            
            # Open and process the PDF
            with fitz.open(file_path) as pdf_doc:
                if len(pdf_doc) == 0:
                    raise ValueError("PDF file contains no pages")
                    
                # Extract text from all pages with proper separation
                text_parts = []
                for page in pdf_doc:
                    page_text = page.get_text().strip()
                    if page_text:  # Only add non-empty pages
                        text_parts.append(page_text)
                
                # Join pages with newlines to preserve separation
                text = '\n\n'.join(text_parts)
                if not text.strip():
                    self.logger.warning(f"No text content found in PDF: {file_path}")
                
                # Get PDF metadata
                pdf_info = pdf_doc.metadata
                if pdf_info:
                    # Clean and validate metadata values
                    cleaned_info = {
                        key: str(value).strip() if value else ''
                        for key, value in pdf_info.items()
                    }
                    metadata.update({
                        'title': cleaned_info.get('title', ''),
                        'author': cleaned_info.get('author', ''),
                        'subject': cleaned_info.get('subject', ''),
                        'keywords': cleaned_info.get('keywords', ''),
                        'creator': cleaned_info.get('creator', ''),
                        'producer': cleaned_info.get('producer', ''),
                        'page_count': len(pdf_doc)
                    })
                
                # Add processing metadata
                metadata.update({
                    'processor': 'PDFDocumentProcessor',
                    'processed_at': datetime.utcnow().isoformat(),
                    'content_type': 'application/pdf',
                    'file_path': file_path,
                    'has_text_content': bool(text.strip())
                })
                
                # Chunk the extracted text if any exists
                if text.strip():
                    chunks = self._chunk_text_with_sentences(
                        text,
                        max_chunk_size=self.max_chunk_size,
                        overlap=self.chunk_overlap
                    )
                    metadata['chunks'] = chunks
                else:
                    metadata['chunks'] = []
                
                return metadata
                
        except fitz.FileDataError as e:
            raise ValueError(f"Invalid PDF file: {e}")
        except IOError as e:
            raise IOError(f"Failed to read PDF file: {e}")
        except Exception as e:
            # Log the specific error type for debugging
            self.logger.error(f"Error processing PDF {file_path}: {type(e).__name__}: {e}")
            raise 