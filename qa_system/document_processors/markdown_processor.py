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
    def __init__(self, config, chunk_size=3072):
        super().__init__(config)
        self.chunk_size = chunk_size

    def count_tokens(self, text):
        """
        Estimate the number of tokens in a string for Gemini models.
        Args:
            text (str): Input text
        Returns:
            int: Estimated number of tokens (1 token ≈ 4 characters)
        """
        return max(1, len(text) // 3) # lets be conservative and assume 3 characters per token

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

    def _split_section_into_subchunks(self, section_lines, max_chunk_size):
        """
        Split a section (list of lines) into sub-chunks as evenly as possible, using sentence boundaries.
        Try to avoid splitting lists or tables by adjusting split points.
        Args:
            section_lines (list[str]): Lines in the section
            max_chunk_size (int): Maximum chunk size in characters
        Returns:
            list[list[str]]: List of sub-chunks (each a list of lines)
        """
        import re
        # Join lines to a single string for sentence splitting
        section_text = '\n'.join(section_lines)
        # Split into sentences (keep newlines for list/table detection)
        # Use a regex that splits on sentence boundaries but keeps newlines
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9#*-])')
        sentences = sentence_pattern.split(section_text)
        # Reconstruct sentences with their original newlines
        # (This is a best-effort; for more accuracy, use NLP libraries)
        # Now group sentences into chunks
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            sent = sent.rstrip('\n')
            if not sent.strip():
                continue
            if current_len + len(sent) > max_chunk_size and current:
                # Try to avoid splitting in the middle of a list or table
                # If the last line in current or the first line in sent is a list/table, adjust
                def is_list_or_table(line):
                    return bool(re.match(r'\s*([-*+]|\d+\.|\|)', line.strip()))
                # Check last line of current and first line of sent
                last_line = current[-1].split('\n')[-1] if current else ''
                first_line = sent.split('\n')[0]
                if is_list_or_table(last_line) or is_list_or_table(first_line):
                    # Try to move the split point up or down
                    # If possible, move sent to next chunk
                    if len(current) > 1:
                        # Move last sentence to next chunk
                        sent = current.pop() + ' ' + sent
                        current_len = sum(len(s) for s in current)
                    # else: just split here
                chunks.append(current)
                current = []
                current_len = 0
            current.append(sent)
            current_len += len(sent)
        if current:
            chunks.append(current)
        # Join sentences back to text
        return [' '.join(chunk).split('\n') for chunk in chunks if chunk]

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
                self.logger.debug(f"Parsed YAML frontmatter for {file_path if file_path else '[string input]'}: {yaml_data}")
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
        self.logger.debug(f"Parsed {len(headers)} headers in {file_path if file_path else '[string input]'}: {headers}")
        chunk_dicts = []
        chunk_index = 0
        section_hierarchy = []  # Stack of (line_idx, header_level, header_text)
        start_offset = 0
        max_chunk_size = getattr(self, 'chunk_size', 3072)
        header_indices = [h[0] for h in headers]
        header_indices.append(len(lines))  # Sentinel for last section
        # If no headers, treat as a single section and break on paragraphs
        if not headers:
            self.logger.debug(f"No headers found, splitting {file_path if file_path else '[string input]'} on paragraphs.")
            # Split on paragraphs (blank lines)
            para_chunks = []
            para = []
            for line in lines + ['']:
                if not line.strip():
                    if para:
                        para_chunks.append(para)
                        para = []
                else:
                    para.append(line)
            # Now group paragraphs into chunks
            current = []
            current_len = 0
            for para in para_chunks:
                para_text = '\n'.join(para)
                if current_len + len(para_text) > max_chunk_size and current:
                    chunk_lines = current
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if chunk_text:
                        chunk_meta = dict(metadata)
                        chunk_meta['tags'] = self._list_to_csv(sorted(doc_tags))
                        chunk_meta['urls'] = ''
                        chunk_meta['url_contexts'] = []
                        chunk_meta['chunk_index'] = chunk_index
                        chunk_meta['start_offset'] = 0
                        chunk_meta['end_offset'] = 0
                        chunk_meta['section_header'] = ''
                        chunk_meta['section_hierarchy'] = []
                        chunk_meta['topics'] = ["Unknown"]
                        summary = chunk_text.split(". ")[0]
                        if len(summary.split()) < 5:
                            summary = " ".join(chunk_text.split()[:20])
                        chunk_meta['summary'] = summary.strip()
                        chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                        self.logger.debug(f"Created chunk {chunk_index} (size {len(chunk_text)}) for {file_path if file_path else '[string input]'}")
                        chunk_index += 1
                    current = []
                    current_len = 0
                current.extend(para)
                current_len += len(para_text)
            if current:
                chunk_lines = current
                chunk_text = '\n'.join(chunk_lines).strip()
                if chunk_text:
                    chunk_meta = dict(metadata)
                    chunk_meta['tags'] = self._list_to_csv(sorted(doc_tags))
                    chunk_meta['urls'] = ''
                    chunk_meta['url_contexts'] = []
                    chunk_meta['chunk_index'] = chunk_index
                    chunk_meta['start_offset'] = 0
                    chunk_meta['end_offset'] = 0
                    chunk_meta['section_header'] = ''
                    chunk_meta['section_hierarchy'] = []
                    chunk_meta['topics'] = ["Unknown"]
                    summary = chunk_text.split(". ")[0]
                    if len(summary.split()) < 5:
                        summary = " ".join(chunk_text.split()[:20])
                    chunk_meta['summary'] = summary.strip()
                    chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                    self.logger.debug(f"Created chunk {chunk_index} (size {len(chunk_text)}) for {file_path if file_path else '[string input]'}")
                    chunk_index += 1
        else:
            # There are headers; split into sections at each header
            self.logger.debug(f"Splitting {file_path if file_path else '[string input]'} into sections at headers.")
            for i, header in enumerate(headers):
                section_start = header[0]
                section_end = header_indices[i+1] if i+1 < len(header_indices) else len(lines)
                section_lines = lines[section_start:section_end]
                # Remove leading/trailing blank lines
                while section_lines and not section_lines[0].strip():
                    section_lines = section_lines[1:]
                while section_lines and not section_lines[-1].strip():
                    section_lines = section_lines[:-1]
                if not section_lines:
                    continue
                section_text = '\n'.join(section_lines)
                if len(section_text) <= max_chunk_size:
                    # Single chunk
                    chunk_lines_list = [section_lines]
                else:
                    # Split into sub-chunks
                    chunk_lines_list = self._split_section_into_subchunks(section_lines, max_chunk_size)
                for chunk_lines in chunk_lines_list:
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if not chunk_text:
                        continue
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
                    chunk_meta = dict(metadata)
                    chunk_meta['tags'] = self._list_to_csv(sorted(doc_tags | hashtags))
                    chunk_meta['urls'] = self._list_to_csv(sorted(urls))
                    chunk_meta['url_contexts'] = url_contexts
                    chunk_meta['chunk_index'] = chunk_index
                    chunk_meta['start_offset'] = section_start
                    chunk_meta['end_offset'] = section_end - 1
                    chunk_meta['section_header'] = header[2]
                    chunk_meta['section_hierarchy'] = [h[2] for h in headers[:i+1]]
                    chunk_meta['topics'] = ["Unknown"]
                    summary = chunk_text.split(". ")[0]
                    if len(summary.split()) < 5:
                        summary = " ".join(chunk_text.split()[:20])
                    chunk_meta['summary'] = summary.strip()
                    chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                    self.logger.debug(f"Created chunk {chunk_index} (size {len(chunk_text)}) for {file_path if file_path else '[string input]'}")
                    chunk_index += 1
        # --- Compose document-level metadata ---
        document_metadata = dict(metadata)
        document_metadata['tags'] = self._list_to_csv(sorted(doc_tags))
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        # Ensure all required fields are present, even if empty
        for key in ['tags', 'urls']:
            if key not in document_metadata:
                document_metadata[key] = ''
        self.logger.info(f"Markdown processing complete for {file_path if file_path else '[string input]'}: {len(chunk_dicts)} chunks, {document_metadata['total_tokens']} total characters.")
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
        self.logger.info(f"[START] Processing markdown file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.logger.debug(f"Read {len(text)} characters from {file_path}")
            result = self._process_markdown(text, metadata=metadata, file_path=file_path)
            num_chunks = len(result.get('chunks', []))
            chunk_sizes = [len(chunk['text']) for chunk in result.get('chunks', [])]
            self.logger.info(f"[END] Processed markdown file: {file_path} | Chunks: {num_chunks} | Chunk sizes: {chunk_sizes}")
            if num_chunks == 0:
                self.logger.warning(f"No chunks generated for {file_path}. Check if the file is empty or only contains YAML header.")
            return result
        except Exception as e:
            import traceback
            self.logger.error(f"Exception while processing markdown file {file_path}: {e}")
            self.logger.error(traceback.format_exc())
            return {'chunks': [], 'metadata': {'error': str(e), 'file_path': file_path}}

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