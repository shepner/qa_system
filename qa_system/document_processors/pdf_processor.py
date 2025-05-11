from .base_processor import BaseDocumentProcessor
import re
import csv
import io

class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF (.pdf) files. Extracts metadata, extracts text per page, chunks text, and returns results.
    Attempts to preserve page boundaries in chunking.
    Skips password-protected PDFs with a warning.
    Extracts URLs from all text and stores as CSV string in metadata.
    Now supports section header detection, hierarchical propagation, url context, and chunk position metadata.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def _detect_section_headers(self, text):
        # Placeholder: Detect lines that look like section headers (e.g., all caps, 'Chapter', 'Section', etc.)
        headers = []
        for i, line in enumerate(text.splitlines()):
            if re.match(r'^(CHAPTER|SECTION|[A-Z][A-Z\s\d:,.\-]{5,})$', line.strip()):
                headers.append((i, line.strip()))
        return headers

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing PDF file: {file_path}")
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Please install it with 'pip install pypdf'.")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'rb') as f:
            try:
                reader = pypdf.PdfReader(f)
            except Exception as e:
                # Handle cryptography errors for encrypted PDFs
                if 'cryptography' in str(e) and 'AES' in str(e):
                    self.logger.warning(f"File {file_path} is encrypted (AES) and cannot be processed: {e}. Skipping.")
                    metadata['skipped'] = True
                    metadata['skip_reason'] = 'encrypted-pdf-unsupported-crypto'
                    return {
                        'metadata': metadata,
                        'chunks': [],
                        'page_texts': []
                    }
                raise
            if reader.is_encrypted:
                self.logger.warning(f"File {file_path} is password-protected and will be skipped.")
                metadata['skipped'] = True
                metadata['skip_reason'] = 'password-protected'
                return {
                    'metadata': metadata,
                    'chunks': [],
                    'page_texts': []
                }
            all_text = []
            chunk_dicts = []
            section_hierarchy = []
            chunk_index = 0
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
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
                        for m in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', chunk_text):
                            urls.add(m.group(1))
                            # Context: header, paragraph, table, etc. (placeholder: use 'paragraph')
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
                        # NLP-based topic modeling/classification (placeholder)
                        chunk_meta['topics'] = ["Unknown"]
                        # NLP-based summarization (placeholder: first sentence or first 20 words)
                        summary = chunk_text.split(". ")[0]
                        if len(summary.split()) < 5:
                            summary = " ".join(chunk_text.split()[:20])
                        chunk_meta['summary'] = summary.strip()
                        chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                        chunk_index += 1
                        start_offset = line_idx + 1
                # Update section_hierarchy for next page
                section_hierarchy = current_hierarchy
        # Extract URLs from all text
        urls = set()
        full_text = '\n'.join(all_text)
        for m in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', full_text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        document_metadata = dict(metadata)
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        document_metadata['page_count'] = len(all_text)
        return {
            'chunks': chunk_dicts,
            'document_metadata': document_metadata,
            'page_texts': all_text
        } 