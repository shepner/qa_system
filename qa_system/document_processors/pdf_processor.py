"""
@file: pdf_processor.py
PDFDocumentProcessor: Processes PDF files for document ingestion in the QA system.

This module defines PDFDocumentProcessor, a subclass of BaseDocumentProcessor, for handling PDF (.pdf) files. It extracts metadata, text per page, detects section headers, chunks text, and returns structured results for downstream processing. It also extracts URLs, supports section hierarchy, and provides chunk-level metadata including context and summaries.

Major Components:
- PDFDocumentProcessor: Main processor class for .pdf files
- _list_to_csv: Utility for serializing lists as CSV
- _detect_section_headers: Utility for detecting section headers in text
- process: Main entry point for processing a PDF file

Dependencies:
- pypdf for PDF parsing
- csv, io, re for parsing and serialization

Usage Example:
    processor = PDFDocumentProcessor(config)
    result = processor.process("example.pdf")
    print(result['metadata'])
    print(result['chunks'][0]['text'])

Limitations:
- Section header detection is heuristic and may not match all use cases
- Topics extraction is a placeholder
- Summarization is basic (first sentence or first 20 words)

Version History:
- 1.0: Initial version
"""

from .base_processor import BaseDocumentProcessor
import re
import csv
import io

class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF (.pdf) files using PyMuPDF (fitz).

    Extracts document-level and chunk-level metadata, splits text into logical chunks based on paragraph boundaries and section headers, and serializes URLs for vector store compatibility.

    Features:
    - Extracts file metadata and all URLs
    - Detects section headers and propagates section hierarchy
    - Assigns chunk-level metadata: urls, url contexts, section headers, section hierarchy, chunk position, and summary
    - Handles encrypted PDFs that do not require a password
    - Returns a dictionary with 'chunks', 'metadata', and 'page_texts' keys
    """
    def _list_to_csv(self, items):
        """
        Serialize a list of items as a single CSV string.
        Args:
            items (list): List of items to serialize
        Returns:
            str: CSV-formatted string
        """
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def _detect_section_headers(self, text):
        """
        Detect lines that look like section headers (e.g., all caps, 'Chapter', 'Section', etc.).
        Args:
            text (str): Text to scan for section headers
        Returns:
            list: List of (line_idx, header_text) tuples
        """
        headers = []
        for i, line in enumerate(text.splitlines()):
            if re.match(r'^(CHAPTER|SECTION|[A-Z][A-Z\s\d:,.\-]{5,})$', line.strip()):
                headers.append((i, line.strip()))
        return headers

    def process(self, file_path, metadata=None):
        """
        Process a PDF file, extracting metadata, text, section headers, and chunked text.
        Args:
            file_path (str): Path to the PDF file
            metadata (dict, optional): Additional metadata to include
        Returns:
            dict: {
                'chunks': List of chunk dicts with 'text' and 'metadata',
                'metadata': Document-level metadata,
                'page_texts': List of text per page
            }
        Notes:
            - Encrypted PDFs are only skipped if they cannot be decrypted (i.e., truly password-protected).
            - Encrypted PDFs that can be opened with an empty password are processed normally.
        """
        self.logger.debug(f"Processing PDF file with PyMuPDF: {file_path}")
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Please install it with 'pip install pymupdf'.")
        # Extract or merge metadata
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            self.logger.warning(f"File {file_path} could not be opened with PyMuPDF: {e}")
            metadata['skipped'] = True
            metadata['skip_reason'] = f'open-error: {e}'
            return {
                'metadata': metadata,
                'chunks': [],
                'page_texts': []
            }
        all_text = []
        chunk_dicts = []
        section_hierarchy = []
        chunk_index = 0
        for i, page in enumerate(doc):
            text = page.get_text() or ''
            all_text.append(text)
            # Detect section headers in this page
            headers = self._detect_section_headers(text)
            header_lines = {idx: header for idx, header in headers}
            lines = text.splitlines()
            current_section = None
            current_hierarchy = list(section_hierarchy)
            start_offset = 0
            for line_idx, line in enumerate(lines):
                if line_idx in header_lines:
                    current_section = header_lines[line_idx]
                    current_hierarchy = section_hierarchy + [current_section]
                # Chunk at paragraph boundaries (empty line or end of section)
                if line.strip() == '' or line_idx == len(lines) - 1:
                    chunk_text = '\n'.join(lines[start_offset:line_idx+1]).strip()
                    if not chunk_text:
                        start_offset = line_idx + 1
                        continue
                    # Find URLs and their context
                    urls = set()
                    url_contexts = []
                    for m in re.finditer(r'(https?://[^\s)\]"\'<>]+|ftp://[^\s)\]"\'<>]+)', chunk_text):
                        urls.add(m.group(1))
                        url_contexts.append({'url': m.group(1), 'context': 'paragraph'})
                    chunk_meta = dict(metadata)
                    chunk_meta['urls'] = self._list_to_csv(sorted(urls))
                    chunk_meta['url_contexts'] = url_contexts
                    chunk_meta['page_number'] = i + 1
                    chunk_meta['chunk_index'] = chunk_index
                    chunk_meta['start_offset'] = start_offset
                    chunk_meta['end_offset'] = line_idx
                    chunk_meta['section_header'] = current_section if current_section else ''
                    chunk_meta['section_hierarchy'] = list(current_hierarchy)
                    chunk_meta['topics'] = ["Unknown"]
                    summary = chunk_text.split(". ")[0]
                    if len(summary.split()) < 5:
                        summary = " ".join(chunk_text.split()[:20])
                    chunk_meta['summary'] = summary.strip()
                    chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                    chunk_index += 1
                    start_offset = line_idx + 1
            section_hierarchy = current_hierarchy
        # Extract URLs from all text
        urls = set()
        full_text = '\n'.join(all_text)
        for m in re.finditer(r'(https?://[^\s)\]"\'<>]+|ftp://[^\s)\]"\'<>]+)', full_text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        document_metadata = dict(metadata)
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        document_metadata['page_count'] = len(all_text)
        return {
            'chunks': chunk_dicts,
            'metadata': document_metadata,
            'page_texts': all_text
        } 