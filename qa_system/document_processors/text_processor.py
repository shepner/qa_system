from .base_processor import BaseDocumentProcessor
import re
import csv
import io

class TextDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for plain text (.txt) files. Extracts metadata, chunks text, and returns results.
    Extracts URLs from text and stores as CSV string in metadata.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing text file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            # Merge/override with extracted metadata
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Extract URLs from text
        urls = set()
        for m in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        chunks = self.chunk_text(text)
        chunk_dicts = []
        offset = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_urls = set()
            url_contexts = []
            for m in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', chunk):
                chunk_urls.add(m.group(1))
                url_contexts.append({'url': m.group(1), 'context': 'paragraph'})
            chunk_meta = dict(metadata)
            chunk_meta['urls'] = self._list_to_csv(sorted(chunk_urls))
            chunk_meta['url_contexts'] = url_contexts
            chunk_meta['chunk_index'] = chunk_index
            chunk_meta['start_offset'] = offset
            chunk_meta['end_offset'] = offset + len(chunk) - 1
            offset += len(chunk)
            # NLP-based topic modeling/classification (placeholder)
            chunk_meta['topics'] = ["Unknown"]
            # NLP-based summarization (placeholder: first sentence or first 20 words)
            summary = chunk.split(". ")[0]
            if len(summary.split()) < 5:
                summary = " ".join(chunk.split()[:20])
            chunk_meta['summary'] = summary.strip()
            chunk_dicts.append({'text': chunk, 'metadata': chunk_meta})
        document_metadata = dict(metadata)
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        return {
            'chunks': chunk_dicts,
            'metadata': document_metadata
        } 