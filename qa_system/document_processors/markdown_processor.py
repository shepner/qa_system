from .base_processor import BaseDocumentProcessor
import re
import yaml
import csv
import io

class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for Markdown (.md) files. Extracts metadata, chunks text, and returns results.
    Attempts to preserve markdown structure in chunking.
    Also extracts tags (YAML frontmatter and hashtags) and URLs (markdown links and raw URLs).
    Serializes tags and urls as CSV strings in metadata for vector store compatibility.
    Now assigns chunk-level metadata for tags, URLs, section headers, section hierarchy, and chunk position.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()  # Remove trailing newline

    def _parse_headers(self, lines):
        # Returns a list of (line_idx, header_level, header_text)
        headers = []
        for idx, line in enumerate(lines):
            m = re.match(r'^(#+)\s+(.*)$', line)
            if m:
                headers.append((idx, len(m.group(1)), m.group(2).strip()))
        return headers

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing markdown file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
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
        headers = self._parse_headers(lines)
        chunk_dicts = []
        chunk_index = 0
        section_hierarchy = []
        start_offset = 0
        last_header_idx = -1
        for idx, line in enumerate(lines + ['']):  # Add sentinel for last chunk
            m = re.match(r'^(#+)\s+(.*)$', line) if idx < len(lines) else None
            if m or idx == len(lines):
                # New header or end of file: process previous chunk
                if start_offset < idx:
                    chunk_lines = lines[start_offset:idx]
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if chunk_text:
                        # Find hashtags in this chunk
                        hashtags = set(re.findall(r'(?<![\w-])#([\w-]+)', chunk_text))
                        # Find URLs in this chunk (markdown links and raw URLs)
                        urls = set()
                        url_contexts = []
                        for m_url in re.finditer(r'\[[^\]]+\]\(([^)\s]+)\)', chunk_text):
                            urls.add(m_url.group(1))
                            url_contexts.append({'url': m_url.group(1), 'context': 'markdown_link'})
                        for m_url in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', chunk_text):
                            urls.add(m_url.group(1))
                            url_contexts.append({'url': m_url.group(1), 'context': 'raw_url'})
                        # Compose chunk metadata
                        chunk_meta = dict(metadata)  # inherit document-level metadata
                        chunk_meta['tags'] = self._list_to_csv(sorted(doc_tags | hashtags))
                        chunk_meta['urls'] = self._list_to_csv(sorted(urls))
                        chunk_meta['url_contexts'] = url_contexts
                        chunk_meta['chunk_index'] = chunk_index
                        chunk_meta['start_offset'] = start_offset
                        chunk_meta['end_offset'] = idx - 1
                        # Section header and hierarchy
                        if section_hierarchy:
                            chunk_meta['section_header'] = section_hierarchy[-1][2]
                            chunk_meta['section_hierarchy'] = [h[2] for h in section_hierarchy]
                        else:
                            chunk_meta['section_header'] = ''
                            chunk_meta['section_hierarchy'] = []
                        # NLP-based topic modeling/classification (placeholder)
                        chunk_meta['topics'] = ["Unknown"]
                        # NLP-based summarization (placeholder: first sentence or first 20 words)
                        summary = chunk_text.split(". ")[0]
                        if len(summary.split()) < 5:
                            summary = " ".join(chunk_text.split()[:20])
                        chunk_meta['summary'] = summary.strip()
                        chunk_dicts.append({'text': chunk_text, 'metadata': chunk_meta})
                        chunk_index += 1
                # Update section hierarchy for new header
                if m:
                    header_level = len(m.group(1))
                    header_text = m.group(2).strip()
                    # Pop deeper/equal levels
                    while section_hierarchy and section_hierarchy[-1][1] >= header_level:
                        section_hierarchy.pop()
                    section_hierarchy.append((idx, header_level, header_text))
                start_offset = idx + 1
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