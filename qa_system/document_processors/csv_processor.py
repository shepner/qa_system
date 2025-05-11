from .base_processor import BaseDocumentProcessor
import csv
import io
import re

class CSVDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for CSV (.csv) files. Extracts metadata, chunks text, and returns results.
    Extracts URLs from all cell values and stores as CSV string in metadata.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing CSV file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Extract URLs from all cell values
        urls = set()
        for row in rows:
            for cell in row:
                for m in re.finditer(r'(https?://[^\s)\]\'"<>]+|ftp://[^\s)\]\'"<>]+)', cell):
                    urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        # Flatten rows for chunking
        text = '\n'.join([','.join(row) for row in rows])
        chunks = self.chunk_text(text)
        metadata['chunk_count'] = len(chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in chunks)
        return {
            'metadata': metadata,
            'chunks': chunks
        } 