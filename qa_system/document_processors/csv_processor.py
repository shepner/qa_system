"""
@file: csv_processor.py
CSVDocumentProcessor: Processes CSV files for document ingestion.

This module defines CSVDocumentProcessor, a subclass of BaseDocumentProcessor, for handling CSV (.csv) files. It extracts metadata, finds URLs, chunks text, and returns structured results for downstream processing.
"""

from .base_processor import BaseDocumentProcessor
import csv
import io
import re

class CSVDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for CSV (.csv) files.

    Responsibilities:
        - Extracts file and CSV metadata (header fields, row count, etc.)
        - Extracts all URLs from cell values and stores as CSV string in metadata
        - Chunks the CSV content for downstream processing
        - Ensures all required metadata fields are present in output
    """
    def _list_to_csv(self, items):
        """
        Convert a list of items to a single CSV-formatted string.

        Args:
            items (list): List of items to join as CSV.
        Returns:
            str: CSV-formatted string of items.
        """
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def process(self, file_path, metadata=None):
        """
        Process a CSV file, extracting metadata, URLs, and chunked text.

        Args:
            file_path (str or Path): Path to the CSV file.
            metadata (dict, optional): Additional metadata to merge.
        Returns:
            dict: {
                'chunks': List of chunk dicts with text and metadata,
                'metadata': Document-level metadata
            }
        """
        self.logger.debug(f"Processing CSV file: {file_path}")
        # Extract or merge metadata
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        # Read CSV rows
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Extract URLs from all cell values
        urls = set()
        for row in rows:
            for cell in row:
                for m in re.finditer(r'(https?://[^\s)\]"\'<>"]+|ftp://[^\s)\]"\'<>"]+)', cell):
                    urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        # Always include header_fields and row_count
        metadata['header_fields'] = rows[0] if rows else []
        metadata['row_count'] = len(rows) - 1 if len(rows) > 1 else 0
        # Flatten rows for chunking
        text = '\n'.join([','.join(row) for row in rows])
        chunks = self.chunk_text(text)
        chunk_dicts = []
        offset = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_urls = set()
            url_contexts = []
            for m in re.finditer(r'(https?://[^\s)\]"\'<>"]+|ftp://[^\s)\]"\'<>"]+)', chunk):
                chunk_urls.add(m.group(1))
                url_contexts.append({'url': m.group(1), 'context': 'cell'})
            chunk_meta = dict(metadata)
            chunk_meta['urls'] = self._list_to_csv(sorted(chunk_urls))
            chunk_meta['url_contexts'] = url_contexts
            chunk_meta['chunk_index'] = chunk_index
            chunk_meta['start_offset'] = offset
            chunk_meta['end_offset'] = offset + len(chunk) - 1
            offset += len(chunk)
            # Always include required fields, even if empty
            chunk_meta['tags'] = ''
            chunk_meta['topics'] = ["Unknown"]
            summary = chunk.split(". ")[0]
            if len(summary.split()) < 5:
                summary = " ".join(chunk.split()[:20])
            chunk_meta['summary'] = summary.strip() if summary else ''
            # Ensure all required fields are present
            for key in ['tags', 'urls', 'url_contexts', 'topics', 'summary']:
                if key not in chunk_meta:
                    chunk_meta[key] = '' if key in ['tags', 'summary', 'urls'] else ([] if key == 'url_contexts' else ["Unknown"])
            chunk_dicts.append({'text': chunk, 'metadata': chunk_meta})
        document_metadata = dict(metadata)
        document_metadata['chunk_count'] = len(chunk_dicts)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunk_dicts)
        # Ensure all required fields are present, even if empty
        for key in ['header_fields', 'row_count', 'urls']:
            if key not in document_metadata:
                document_metadata[key] = [] if key == 'header_fields' else (0 if key == 'row_count' else '')
        return {
            'chunks': chunk_dicts,
            'metadata': document_metadata
        } 