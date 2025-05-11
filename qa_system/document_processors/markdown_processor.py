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
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()  # Remove trailing newline

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing markdown file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # --- Extract tags ---
        tags = set()
        # YAML frontmatter (if present)
        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
        if yaml_match:
            try:
                yaml_data = yaml.safe_load(yaml_match.group(1))
                if isinstance(yaml_data, dict):
                    # Common keys: 'tags', 'tag', 'categories', etc.
                    for key in ['tags', 'tag', 'categories', 'category']:
                        if key in yaml_data:
                            val = yaml_data[key]
                            if isinstance(val, str):
                                tags.update([v.strip() for v in val.split(',')])
                            elif isinstance(val, list):
                                tags.update([str(v).strip() for v in val])
            except Exception as e:
                self.logger.warning(f"YAML frontmatter parse error: {e}")
        # Hashtags in body (e.g., #tag)
        body = text[yaml_match.end():] if yaml_match else text
        hashtag_matches = re.findall(r'(?<![\w-])#([\w-]+)', body)
        tags.update(hashtag_matches)
        metadata['tags'] = self._list_to_csv(sorted(tags))
        # --- Extract URLs ---
        urls = set()
        # Markdown links: [text](url)
        for m in re.finditer(r'\[[^\]]+\]\(([^)\s]+)\)', text):
            urls.add(m.group(1))
        # Raw URLs (http/https/ftp)
        for m in re.finditer(r'(https?://[^\s)\]\'"<>]+|ftp://[^\s)\]\'"<>]+)', text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        # --- Chunking logic (as before) ---
        header_chunks = re.split(r'(^#+ .*$)', text, flags=re.MULTILINE)
        merged_chunks = []
        i = 0
        while i < len(header_chunks):
            if header_chunks[i].startswith('#'):
                header = header_chunks[i]
                content = header_chunks[i+1] if i+1 < len(header_chunks) else ''
                merged_chunks.append(header + '\n' + content)
                i += 2
            else:
                if header_chunks[i].strip():
                    merged_chunks.append(header_chunks[i])
                i += 1
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self.chunk_text(chunk))
            else:
                final_chunks.append(chunk)
        metadata['chunk_count'] = len(final_chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in final_chunks)
        return {
            'metadata': metadata,
            'chunks': final_chunks
        } 