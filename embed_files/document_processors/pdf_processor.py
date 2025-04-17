from typing import Dict, Any, List, Tuple
import PyPDF2
import tiktoken
import re
from .base_processor import BaseDocumentProcessor
from embed_files.config import get_config
import datetime

class PDFDocumentProcessor(BaseDocumentProcessor):
    """Document processor for PDF files."""
    
    def __init__(self):
        """Initialize the PDF processor with configuration."""
        super().__init__()
        config = get_config()
        processing_config = config.get_nested('DOCUMENT_PROCESSING', {})
        self.max_chunk_size = processing_config.get('MAX_CHUNK_SIZE', 1500)
        self.chunk_overlap = processing_config.get('CHUNK_OVERLAP', 300)
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Common header patterns
        self.header_patterns = [
            r'^(?:Chapter|Section)\s+\d+',  # Chapter 1, Section 2
            r'^\d+\.\d+(?:\.\d+)*\s+[A-Z]', # 1.1, 1.2.1
            r'^[A-Z][A-Z\s]{4,}(?:\n|$)',   # ALL CAPS HEADER
            r'^\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z]'  # Roman numerals
        ]
        self.header_regex = re.compile('|'.join(self.header_patterns), re.MULTILINE)
    
    def _extract_text(self, pdf_reader: PyPDF2.PdfReader) -> str:
        """Extract and normalize text from PDF.
        
        Args:
            pdf_reader: PyPDF2 reader object
            
        Returns:
            str: Normalized text content
        """
        text_content = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text:
                # Enhanced text normalization
                text = ' '.join(text.split())  # Normalize whitespace
                text = text.replace('\x00', '')  # Remove null characters
                text = text.strip()  # Remove leading/trailing whitespace
                if text:  # Only add non-empty pages
                    text_content.append({
                        'page_number': page_num,
                        'content': text
                    })
        
        # Join with page markers for better context
        return '\n\n'.join(f"--- Page {page['page_number']} ---\n\n{page['content']}" for page in text_content)
    
    def _find_section_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """Find section boundaries in text based on header patterns.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of tuples containing (start_pos, end_pos, header_text)
        """
        sections = []
        last_pos = 0
        
        # Find all potential headers
        for match in self.header_regex.finditer(text):
            start = match.start()
            header = match.group(0)
            
            # If we have a previous section, add it
            if last_pos > 0:
                sections.append((last_pos, start, text[last_pos:last_pos + 50] + "..."))
            
            last_pos = start
        
        # Add the final section
        if last_pos < len(text):
            sections.append((last_pos, len(text), text[last_pos:last_pos + 50] + "..."))
        
        return sections
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata, preferring section boundaries.
        
        Args:
            text: Text content to chunk
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        chunks = []
        chunk_number = 1
        
        # Find section boundaries
        sections = self._find_section_boundaries(text)
        
        for section_start, section_end, section_header in sections:
            section_text = text[section_start:section_end]
            current_pos = 0
            
            # If section is smaller than max chunk size, keep it as one chunk
            if len(section_text) <= self.max_chunk_size:
                chunk_tokens = self.tokenizer.encode(section_text)
                chunks.append({
                    'chunk_number': chunk_number,
                    'start_pos': section_start,
                    'end_pos': section_end,
                    'content': section_text,
                    'token_count': len(chunk_tokens),
                    'section_header': section_header
                })
                chunk_number += 1
                continue
            
            # Split large sections into chunks with overlap
            while current_pos < len(section_text):
                end_pos = min(current_pos + self.max_chunk_size, len(section_text))
                
                # If not at the end, try to break at a sentence boundary
                if end_pos < len(section_text):
                    for i in range(end_pos - 1, current_pos, -1):
                        if section_text[i] in '.!?' and (i + 1 == len(section_text) or section_text[i + 1].isspace()):
                            end_pos = i + 1
                            break
                
                chunk_text = section_text[current_pos:end_pos].strip()
                chunk_tokens = self.tokenizer.encode(chunk_text)
                
                chunks.append({
                    'chunk_number': chunk_number,
                    'start_pos': section_start + current_pos,
                    'end_pos': section_start + end_pos,
                    'content': chunk_text,
                    'token_count': len(chunk_tokens),
                    'section_header': section_header
                })
                
                chunk_number += 1
                current_pos = end_pos - self.chunk_overlap
        
        return chunks
    
    def _extract_pdf_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract PDF-specific metadata.
        
        Args:
            pdf_reader: PyPDF2 reader object
            
        Returns:
            Dict[str, Any]: PDF metadata
        """
        metadata = {
            'document_info': {},
            'structure_info': {},
            'processing_info': {}
        }
        
        # Extract document info if available
        if pdf_reader.metadata:
            info = pdf_reader.metadata
            metadata['document_info'].update({
                'title': info.get('/Title', ''),
                'author': info.get('/Author', ''),
                'subject': info.get('/Subject', ''),
                'keywords': info.get('/Keywords', ''),
                'creator': info.get('/Creator', ''),
                'producer': info.get('/Producer', ''),
                'creation_date': info.get('/CreationDate', ''),
                'modification_date': info.get('/ModDate', '')
            })
        
        # Add PDF structure information
        metadata['structure_info'].update({
            'page_count': len(pdf_reader.pages),
            'is_encrypted': pdf_reader.is_encrypted,
            'file_size_bytes': pdf_reader.stream.getbuffer().nbytes if hasattr(pdf_reader, 'stream') else None
        })
        
        return metadata
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF document.
        
        Args:
            file_path: Path to the PDF file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing PDF file: {file_path}")
        
        try:
            with open(file_path, 'rb') as pdf_file:
                # Create PDF reader
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text content
                text_content = self._extract_text(pdf_reader)
                
                # Chunk the text
                chunks = self._chunk_text(text_content)
                
                # Get total tokens for the entire document
                total_tokens = sum(chunk['token_count'] for chunk in chunks)
                
                # Extract PDF metadata
                pdf_metadata = self._extract_pdf_metadata(pdf_reader)
                
                # Update metadata with processing results
                metadata.update({
                    'file_metadata': metadata,  # Original file metadata
                    'pdf_metadata': pdf_metadata,  # PDF-specific metadata
                    'processing_metadata': {
                        'chunks': chunks,
                        'chunk_count': len(chunks),
                        'total_tokens': total_tokens,
                        'processing_status': 'success',
                        'processor_version': '1.1.0',
                        'processing_timestamp': datetime.datetime.now().isoformat()
                    }
                })
                
                self.logger.info(f"Successfully processed PDF file: {file_path}")
                
        except Exception as e:
            error_msg = f"Error processing PDF file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            metadata.update({
                'processing_metadata': {
                    'processing_status': 'error',
                    'error_message': error_msg,
                    'processor_version': '1.1.0',
                    'processing_timestamp': datetime.datetime.now().isoformat()
                }
            })
        
        return metadata 