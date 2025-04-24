from typing import Dict, Any, List, Optional, Iterator
import logging
from pathlib import Path
from datetime import datetime
import os
import csv
from io import StringIO
from .base_processor import BaseDocumentProcessor
from qa_system.exceptions import ProcessingError, ValidationError
import chardet
from tenacity import retry, stop_after_attempt, wait_exponential

class CSVDocumentProcessor(BaseDocumentProcessor):
    """Document processor for CSV files.
    
    Handles processing of CSV files according to the architecture specification.
    Features:
    - CSV parsing with header detection
    - Row-based chunking with header preservation
    - Metadata extraction
    - Token counting
    """
    
    # Maximum file size (100MB by default)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum sample size for dialect detection (64KB)
    SAMPLE_SIZE = 64 * 1024
    
    def __init__(self, config=None):
        """Initialize the CSV document processor.
        
        Args:
            config: Configuration dictionary or object (optional)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Set default configuration
        self.max_rows_per_chunk = 100
        self.include_headers = True
        
        # Update configuration if provided
        if config:
            if hasattr(config, 'get_nested'):
                doc_processing = config.get_nested('DOCUMENT_PROCESSING', {})
            else:
                doc_processing = config.get('DOCUMENT_PROCESSING', {})
                
            self.max_rows_per_chunk = doc_processing.get('CSV_MAX_ROWS_PER_CHUNK', self.max_rows_per_chunk)
            self.include_headers = doc_processing.get('CSV_INCLUDE_HEADERS', self.include_headers)
    
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
    
    def _detect_dialect(self, file_path: str, encoding: str) -> csv.Dialect:
        """Detect CSV dialect using csv.Sniffer.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            Detected CSV dialect
            
        Raises:
            ValidationError: If dialect cannot be detected
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(self.SAMPLE_SIZE)
                dialect = csv.Sniffer().sniff(sample)
                return dialect
        except Exception as e:
            raise ValidationError(f"Failed to detect CSV dialect: {str(e)}")
    
    def _process_csv_rows(self, file_path: str, encoding: str, dialect: csv.Dialect) -> Iterator[Dict[str, str]]:
        """Process CSV rows in a memory-efficient way.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            dialect: CSV dialect
            
        Yields:
            Dictionary for each row with non-empty values
        """
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                # Filter out empty values and format as key-value pairs
                formatted_row = {
                    k: v for k, v in row.items()
                    if v and v.strip() and k and k.strip()
                }
                if formatted_row:
                    yield formatted_row
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CSV file and extract text and metadata.
        
        Args:
            file_path: Path to the CSV file
            metadata: Dictionary containing metadata about the file
            
        Returns:
            Dictionary containing metadata and extracted text chunks
            
        Raises:
            ProcessingError: If processing fails
            ValidationError: If file validation fails
        """
        self.logger.debug(f"Processing CSV file: {file_path}")
        
        try:
            # Validate file size
            self._check_file_size(file_path)
            
            # Enhance metadata with standard fields
            metadata = self._enhance_metadata(file_path, metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Detect file encoding and dialect
            encoding = self._detect_encoding(file_path)
            dialect = self._detect_dialect(file_path, encoding)
            
            # Process CSV rows
            text_chunks = []
            row_count = 0
            headers = None
            
            try:
                for row in self._process_csv_rows(file_path, encoding, dialect):
                    if self._should_stop():
                        self.logger.warning("Processing interrupted")
                        break
                    
                    if not headers:
                        headers = list(row.keys())
                    
                    # Format row as text
                    row_text = "\n".join(f"{k}: {v}" for k, v in row.items())
                    if row_text:
                        text_chunks.append(row_text)
                    row_count += 1
                
                # Calculate total tokens
                total_tokens = sum(self._count_tokens(chunk) for chunk in text_chunks)
                
                # Update processing metadata
                metadata.update({
                    'processor_type': 'csv',
                    'total_tokens': total_tokens,
                    'chunk_count': len(text_chunks),
                    'average_chunk_size': total_tokens / len(text_chunks) if text_chunks else 0,
                    'content_type': 'text/csv',
                    'encoding': encoding,
                    'dialect': {
                        'delimiter': dialect.delimiter,
                        'quotechar': dialect.quotechar,
                        'escapechar': dialect.escapechar or '',
                        'doublequote': dialect.doublequote,
                        'skipinitialspace': dialect.skipinitialspace,
                    },
                    'row_count': row_count,
                    'headers': headers or []
                })
                
                return {
                    'metadata': metadata,
                    'chunks': text_chunks
                }
                
            except csv.Error as e:
                raise ProcessingError(f"CSV processing error: {str(e)}")
            except Exception as e:
                raise ProcessingError(f"Failed to process CSV: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            raise
    
    def _should_stop(self) -> bool:
        """Check if processing should be stopped."""
        return False  # Override in subclass if needed
    
    def _chunk_csv(self, rows: List[List[str]], headers: List[str]) -> List[Dict[str, Any]]:
        """Split CSV rows into chunks.
        
        Args:
            rows: List of CSV rows
            headers: List of column headers
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = []
        chunk_number = 1
        
        for row in rows:
            current_chunk.append(row)
            
            if len(current_chunk) >= self.max_rows_per_chunk:
                # Convert chunk to text
                chunk_text = self._rows_to_text(current_chunk, headers)
                token_count = self._count_tokens(chunk_text)
                
                chunks.append({
                    'chunk_number': chunk_number,
                    'start_row': (chunk_number - 1) * self.max_rows_per_chunk + 1,
                    'end_row': chunk_number * self.max_rows_per_chunk,
                    'content': chunk_text,
                    'row_count': len(current_chunk),
                    'token_count': token_count
                })
                
                chunk_number += 1
                current_chunk = []
        
        # Handle remaining rows
        if current_chunk:
            chunk_text = self._rows_to_text(current_chunk, headers)
            token_count = self._count_tokens(chunk_text)
            
            chunks.append({
                'chunk_number': chunk_number,
                'start_row': (chunk_number - 1) * self.max_rows_per_chunk + 1,
                'end_row': (chunk_number - 1) * self.max_rows_per_chunk + len(current_chunk),
                'content': chunk_text,
                'row_count': len(current_chunk),
                'token_count': token_count
            })
        
        return chunks
    
    def _rows_to_text(self, rows: List[List[str]], headers: List[str]) -> str:
        """Convert CSV rows to text format.
        
        Args:
            rows: List of CSV rows
            headers: List of column headers
            
        Returns:
            Formatted text representation
        """
        output = StringIO()
        writer = csv.writer(output)
        
        if self.include_headers:
            writer.writerow(headers)
        writer.writerows(rows)
        
        return output.getvalue() 