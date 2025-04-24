from typing import Dict, Any, List
import logging
from pathlib import Path
from datetime import datetime
import os
import re
import markdown
from bs4 import BeautifulSoup
from .base_processor import BaseDocumentProcessor

class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """Document processor for Markdown files.
    
    Handles processing of Markdown files according to the architecture specification.
    Features:
    - Text extraction with markdown structure preservation
    - Content chunking with header-aware boundaries
    - Metadata extraction including frontmatter
    - Token counting
    - Internal link and hashtag detection
    """
    
    def __init__(self, config=None):
        """Initialize the markdown document processor.
        
        Args:
            config: Configuration dictionary or object (optional)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Set default configuration
        self.max_chunk_size = 2000
        self.chunk_overlap = 200
        
        # Update configuration if provided
        if config:
            if hasattr(config, 'get_nested'):
                doc_processing = config.get_nested('DOCUMENT_PROCESSING', {})
            else:
                doc_processing = config.get('DOCUMENT_PROCESSING', {})
                
            self.max_chunk_size = doc_processing.get('MAX_CHUNK_SIZE', self.max_chunk_size)
            self.chunk_overlap = doc_processing.get('CHUNK_OVERLAP', self.chunk_overlap)
    
    def _extract_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown text.
        
        Args:
            text: Raw markdown text
            
        Returns:
            Tuple of (frontmatter dict, remaining text)
        """
        frontmatter = {}
        content = text
        
        # Check for YAML frontmatter
        if text.startswith('---\n'):
            parts = text[4:].split('\n---\n', 1)
            if len(parts) == 2:
                try:
                    # Simple frontmatter parsing (can be enhanced with PyYAML)
                    fm_lines = parts[0].strip().split('\n')
                    for line in fm_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            frontmatter[key.strip()] = value.strip()
                    content = parts[1]
                except Exception as e:
                    self.logger.warning(f"Failed to parse frontmatter: {e}")
        
        return frontmatter, content
    
    def _extract_links_and_tags(self, text: str) -> tuple[List[str], List[str]]:
        """Extract internal links and hashtags from markdown text.
        
        Args:
            text: Markdown text content
            
        Returns:
            Tuple of (list of internal links, list of hashtags)
        """
        # Find internal links [[link]]
        internal_links = re.findall(r'\[\[(.*?)\]\]', text)
        
        # Find hashtags #tag
        hashtags = re.findall(r'#(\w+)', text)
        
        return internal_links, hashtags
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Markdown file and extract its content and metadata.
        
        Args:
            file_path: Path to the Markdown file
            metadata: Initial metadata dictionary
            
        Returns:
            Dictionary containing:
            - Document metadata
            - Extracted text chunks
            - Token counts
            
        Raises:
            ValueError: If file format is invalid
            IOError: If file cannot be read
        """
        try:
            # Enhance metadata with standard fields
            metadata = self._enhance_metadata(file_path, metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert Markdown to HTML
            html = markdown.markdown(content)
            
            # Extract text from HTML
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n\n')
            
            # Chunk the text while preserving sentence boundaries
            chunks = self._chunk_text_with_sentences(text)
            
            # Calculate total tokens
            total_tokens = sum(chunk['token_count'] for chunk in chunks)
            
            # Extract headers for better context
            headers = []
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headers.append({
                    'level': int(header.name[1]),
                    'text': header.get_text()
                })
            
            # Add processing metadata
            metadata.update({
                'processor_type': self.__class__.__name__,
                'total_tokens': total_tokens,
                'chunk_count': len(chunks),
                'average_chunk_size': len(text) / len(chunks) if chunks else 0,
                'content_type': 'text/markdown',
                'headers': headers
            })
            
            return {
                'metadata': metadata,
                'chunks': chunks
            }
            
        except Exception as e:
            self.logger.error(
                f"Error processing Markdown file {file_path}: {str(e)}",
                extra={
                    'component': 'markdown_processor',
                    'operation': 'process',
                    'file_path': file_path,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise 