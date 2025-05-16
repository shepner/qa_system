"""
@file: vision_processor.py
Image document processor for extracting metadata and simulating Vision API results.

This module defines VisionDocumentProcessor, a subclass of BaseDocumentProcessor, for handling image files (jpg, jpeg, png, gif, bmp, webp). It extracts image metadata, simulates Vision API fields, and produces chunked outputs for downstream processing.
"""

from .base_processor import BaseDocumentProcessor
from PIL import Image
from datetime import datetime, timezone
import csv
import io
import re

class VisionDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for image files (jpg, jpeg, png, gif, bmp, webp).
    Extracts image metadata and simulates Vision API results for testing.
    
    Methods:
        process(file_path, metadata=None):
            Process an image file, extract metadata, and simulate Vision API fields.
    """
    def _list_to_csv(self, items):
        """
        Convert a list of items to a CSV string.

        Args:
            items (list): List of items to convert.
        Returns:
            str: CSV-formatted string of items.
        """
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def process(self, file_path, metadata=None):
        """
        Process an image file, extract metadata, and simulate Vision API fields.

        Args:
            file_path (str or Path): Path to the image file.
            metadata (dict, optional): Additional metadata to merge.
        Returns:
            dict: Dictionary with 'chunks' (list of chunk dicts) and 'metadata' (document-level metadata).
        """
        self.logger.debug(f"Processing image file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                metadata['image_dimensions'] = {'width': width, 'height': height}
                metadata['image_format'] = img.format
                metadata['color_profile'] = img.mode
        except Exception as e:
            # If image cannot be opened, record error and ensure all required fields are present
            metadata['error_states'] = [str(e)]
            for key, default in [
                ('image_dimensions', {}),
                ('image_format', ''),
                ('color_profile', ''),
                ('vision_labels', []),
                ('ocr_text', ''),
                ('face_detection', []),
                ('safe_search', {}),
                ('feature_confidence', {}),
                ('processing_timestamp', ''),
                ('error_states', []),
                ('urls', ''),
            ]:
                if key not in metadata:
                    metadata[key] = default
            return {'chunks': []}

        # Simulate Vision API fields for demonstration/testing
        metadata['vision_labels'] = ['label1', 'label2']
        metadata['ocr_text'] = 'Simulated OCR text.'
        metadata['face_detection'] = []
        metadata['safe_search'] = {'adult': 'VERY_UNLIKELY', 'spoof': 'UNLIKELY'}
        metadata['feature_confidence'] = {'label1': 0.95, 'label2': 0.88}
        metadata['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
        metadata['error_states'] = []

        # For images, the "chunk" is just the OCR text and label summary
        chunks = [metadata['ocr_text'], ', '.join(metadata['vision_labels'])]
        chunk_dicts = []
        chunk_types = ['ocr_text', 'vision_labels']
        offset = 0
        for idx, chunk in enumerate(chunks):
            chunk_meta = dict(metadata)
            chunk_meta['chunk_type'] = chunk_types[idx]
            # Extract URLs for this chunk
            chunk_urls = set()
            url_contexts = []
            for m in re.finditer(r'(https?://[^\s)\]\'\"<>]+|ftp://[^\s)\]\'\"<>]+)', chunk):
                chunk_urls.add(m.group(1))
                url_contexts.append({'url': m.group(1), 'context': chunk_types[idx]})
            chunk_meta['urls'] = self._list_to_csv(sorted(chunk_urls))
            chunk_meta['url_contexts'] = url_contexts
            chunk_meta['chunk_index'] = idx
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
        # Ensure all required fields are present, even if empty
        for key, default in [
            ('image_dimensions', {}),
            ('image_format', ''),
            ('color_profile', ''),
            ('vision_labels', []),
            ('ocr_text', ''),
            ('face_detection', []),
            ('safe_search', {}),
            ('feature_confidence', {}),
            ('processing_timestamp', ''),
            ('error_states', []),
            ('urls', ''),
        ]:
            if key not in document_metadata:
                document_metadata[key] = default
        return {
            'chunks': chunk_dicts,
            'metadata': document_metadata
        } 