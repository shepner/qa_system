"""
@file: markdown_processor.py
Processor for Markdown (.md) files in the QA system.

This module defines the MarkdownDocumentProcessor class, which extracts metadata, splits text into structured chunks, and serializes metadata for vector store compatibility. It supports YAML frontmatter, hashtags, and URL extraction, and assigns chunk-level metadata including section hierarchy and summaries.

Major Components:
- MarkdownDocumentProcessor: Main processor class for .md files
- _list_to_csv: Utility for serializing lists as CSV
- _parse_headers: Utility for extracting markdown headers
- process: Main entry point for processing a markdown file

Dependencies:
- PyYAML for YAML frontmatter parsing
- csv, io, re for parsing and serialization

Usage Example:
    processor = MarkdownDocumentProcessor()
    result = processor.process("example.md")
    print(result['metadata'])
    print(result['chunks'][0]['text'])

Limitations:
- Only basic YAML frontmatter is supported
- Chunking is based on headers and may not match all use cases
- Topics extraction is a placeholder

Version History:
- 1.0: Initial version
"""

from .base_processor import BaseDocumentProcessor
import re
import yaml
import csv
import io

class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for Markdown (.md) files.
    
    Extracts document-level and chunk-level metadata, splits text into logical chunks based on headers, and serializes tags and URLs for vector store compatibility.
    
    Features:
    - Extracts YAML frontmatter tags and hashtags
    - Extracts URLs from markdown links and raw URLs
    - Assigns chunk-level metadata: tags, urls, section headers, section hierarchy, chunk position, and summary
    - Returns a dictionary with 'chunks' and 'metadata' keys
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
        return output.getvalue().strip()  # Remove trailing newline

    def _parse_headers(self, lines):
        """
        Parse markdown headers from a list of lines.
        Args:
            lines (list): List of lines from the markdown body
        Returns:
            list: List of (line_idx, header_level, header_text) tuples
        """
        headers = []
        for idx, line in enumerate(lines):
            m = re.match(r'^(#+)\s+(.*)$', line)
            if m:
                headers.append((idx, len(m.group(1)), m.group(2).strip()))
        return headers

    def _process_markdown(self, text, metadata=None, file_path=None):
        """
        Core logic for processing markdown from a string.
        Args:
            text (str): Markdown content
            metadata (dict, optional): Additional metadata
            file_path (str, optional): Path to the file (for logging only)
        Returns:
            dict: { 'chunks': [...], 'metadata': ... }
        """
        self.logger.debug(f"Processing markdown {'file: ' + file_path if file_path else 'string input'}")
        # Extract or merge metadata
        if metadata is None:
            metadata = self.extract_metadata(file_path) if file_path else {}
        else:
            extracted = self.extract_metadata(file_path) if file_path else {}
            metadata = {**extracted, **metadata}
        # --- Extract document-level tags (YAML frontmatter) ---
        doc_tags = set()
        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
        if yaml_match:
            try:
                yaml_data = yaml.safe_load(yaml_match.group(1))
                if isinstance(yaml_data, dict):
                    for key in ['tags', 'tag', 'categories', 'category']:
                        if key in yaml_data:
                            val = yaml_data[key]
                            if isinstance(val, str):
                                doc_tags.update([v.strip() for v in val.split(',')])
                            elif isinstance(val, list):
                                doc_tags.update([str(v).strip() for v in val])
            except Exception as e:
                self.logger.warning(f"YAML frontmatter parse error: {e}")
        # Remove YAML frontmatter from body
        body = text[yaml_match.end():] if yaml_match else text
        lines = body.splitlines()
        # If the file only contains a YAML header and no body, log a clear warning
        if not lines or all(not line.strip() for line in lines):
            self.logger.warning(f"No embeddings generated for {'file ' + file_path if file_path else 'string input'}: only YAML header and no body text.")
            return {
                'chunks': [],
                'metadata': metadata
            }
        # Parse headers for sectioning
        headers = self._parse_headers(lines)
        chunk_dicts = []
        chunk_index = 0
        section_hierarchy = []  # Stack of (line_idx, header_level, header_text)
        start_offset = 0
        # Iterate through lines, splitting at headers
        for idx, line in enumerate(lines + ['']):  # Add sentinel for last chunk
            header_match = re.match(r'^(#+)\s+(.*)$', line) if idx < len(lines) else None
            if header_match or idx == len(lines):
                # Process previous chunk (if any), up to but not including this header
                if start_offset < idx:
                    chunk_lines = lines[start_offset:idx]
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if chunk_text:
                        # --- Extract hashtags ---
                        hashtags = set(
                            re.sub(r'\[\^\d+\]$', '', tag)
                            for tag in re.findall(r'(?<!#)(?<!\w)#([A-Za-z0-9_-]+)\b', chunk_text)
                        )
                        # --- Extract URLs (markdown links and raw URLs) ---
                        urls = set()
                        url_contexts = []
                        for m_url in re.finditer(r'\[[^\]]+\]\(([^)\s]+)\)', chunk_text):
                            urls.add(m_url.group(1))
                            url_contexts.append({'url': m_url.group(1), 'context': 'markdown_link'})
                        for m_url in re.finditer(r'(https?://[^\s)\]"\'<>]+|ftp://[^\s)\]"\'<>]+)', chunk_text):
                            urls.add(m_url.group(1))
                            url_contexts.append({'url': m_url.group(1), 'context': 'raw_url'})
                        # --- Compose chunk metadata ---
                        chunk_meta = dict(metadata)  # inherit document-level metadata
                        chunk_meta['tags'] = self._list_to_csv(sorted(doc_tags | hashtags))
                        chunk_meta['urls'] = self._list_to_csv(sorted(urls))
                        chunk_meta['url_contexts'] = url_contexts
                        chunk_meta['chunk_index'] = chunk_index
                        chunk_meta['start_offset'] = start_offset
                        chunk_meta['end_offset'] = idx - 1
                        if section_hierarchy:
                            chunk_meta['section_header'] = section_hierarchy[-1][2]
                            chunk_meta['section_hierarchy'] = [h[2] for h in section_hierarchy]
                        else:
                            chunk_meta['section_header'] = ''
                            chunk_meta['section_hierarchy'] = []
                        chunk_meta['topics'] = ["Unknown"]  # Placeholder for topic extraction
                        # --- Generate summary ---
                        summary = chunk_text.split(". ")[0]
                        if len(summary.split()) < 5:
                            summary = " ".join(chunk_text.split()[:20])
                        chunk_meta['summary'] = summary.strip()
                        chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                        chunk_index += 1
                # --- Update section hierarchy for new header ---
                if header_match:
                    header_level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    # Pop higher/equal level headers
                    while section_hierarchy and section_hierarchy[-1][1] >= header_level:
                        section_hierarchy.pop()
                    section_hierarchy.append((idx, header_level, header_text))
                # Set start_offset to idx (the header line), so next chunk starts at the header
                start_offset = idx
        # --- Compose document-level metadata ---
        document_metadata = dict(metadata)
        document_metadata['tags'] = self._list_to_csv(sorted(doc_tags))
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        # Ensure all required fields are present, even if empty
        for key in ['tags', 'urls']:
            if key not in document_metadata:
                document_metadata[key] = ''
        return {
            'chunks': chunk_dicts,
            'metadata': document_metadata
        }

    def process(self, file_path, metadata=None):
        """
        Process a markdown file, extracting metadata and splitting into structured chunks.
        Args:
            file_path (str): Path to the markdown file
            metadata (dict, optional): Additional metadata to include
        Returns:
            dict: {
                'chunks': List of chunk dicts with 'text' and 'metadata',
                'metadata': Document-level metadata
            }
        Notes:
            - If the file only contains a YAML header and no body, no chunks/embeddings will be generated, and a warning will be logged explaining this.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._process_markdown(text, metadata=metadata, file_path=file_path)

    def process_from_string(self, text, metadata=None):
        """
        Process a markdown string, extracting metadata and splitting into structured chunks.
        Args:
            text (str): Markdown content
            metadata (dict, optional): Additional metadata to include
        Returns:
            dict: {
                'chunks': List of chunk dicts with 'text' and 'metadata',
                'metadata': Document-level metadata
            }
        """
        return self._process_markdown(text, metadata=metadata, file_path=None) 