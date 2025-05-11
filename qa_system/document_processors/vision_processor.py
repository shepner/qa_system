from .base_processor import BaseDocumentProcessor
from PIL import Image
import os
from datetime import datetime, timezone
import csv
import io
import re

class VisionDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for image files (jpg, jpeg, png, gif, bmp, webp). Extracts metadata and image-specific fields.
    Simulates Vision API results for testing.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

    def process(self, file_path, metadata=None):
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
            metadata['error_states'] = [str(e)]
            return {'metadata': metadata, 'chunks': []}
        # Simulate Vision API fields
        metadata['vision_labels'] = ['label1', 'label2']
        metadata['ocr_text'] = 'Simulated OCR text.'
        metadata['face_detection'] = []
        metadata['safe_search'] = {'adult': 'VERY_UNLIKELY', 'spoof': 'UNLIKELY'}
        metadata['feature_confidence'] = {'label1': 0.95, 'label2': 0.88}
        metadata['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
        metadata['error_states'] = []
        # For images, the "chunk" is just the OCR text and label summary
        chunks = [metadata['ocr_text'], ', '.join(metadata['vision_labels'])]
        metadata['chunk_count'] = len(chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in chunks)
        # Extract URLs from detected text
        text = self.ocr_extract_text(file_path)
        urls = set()
        for m in re.finditer(r'(https?://[^\s)\]\'"<>]+|ftp://[^\s)\]\'"<>]+)', text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls)) if urls else ''
        return {
            'metadata': metadata,
            'chunks': chunks
        } 