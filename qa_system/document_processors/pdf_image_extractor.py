"""
@file: pdf_image_extractor.py
Extracts all images from a PDF and processes them using ImageDocumentProcessor.

This module defines PDFImageExtractor, which extracts all images from a PDF file using PyMuPDF (fitz), saves them to a temporary directory, and processes each image using the provided ImageDocumentProcessor. It aggregates the results for downstream ingestion.

Usage Example:
    from qa_system.document_processors.image_processor import ImageDocumentProcessor
    from qa_system.document_processors.pdf_image_extractor import PDFImageExtractor
    
    image_processor = ImageDocumentProcessor(query_processor)
    extractor = PDFImageExtractor()
    result = extractor.process('example.pdf', image_processor)
    print(result['chunks'])
    print(result['metadata'])
"""

import os
import fitz  # PyMuPDF
import logging
from typing import Optional, Dict, Any
import hashlib
import json

class PDFImageExtractor:
    """
    Extracts all images from a PDF and processes them using ImageDocumentProcessor.
    Implements per-image checkpointing and output retention.
    """
    def __init__(self, config, logger: Optional[logging.Logger] = None, query_processor=None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.query_processor = query_processor

    def _get_checksum(self, file_path):
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _get_extract_dir(self, pdf_path):
        checksum = self._get_checksum(pdf_path)
        return os.path.join('tmp', 'extracted', checksum)

    def _get_checkpoint_path(self, extract_dir):
        return os.path.join(extract_dir, 'checkpoint.json')

    def _load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, checkpoint_path, checkpoint):
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def process(self, pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Extract all images from the PDF and process them with the image ingestion pipeline.
        Uses per-image checkpointing and output retention.
        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save extracted images. If None, uses tmp/extracted/<checksum>/
        Returns:
            dict: {'chunks': [...], 'metadata': {...}}
        """
        self.logger.info(f"Extracting images from PDF: {pdf_path}")
        extract_dir = self._get_extract_dir(pdf_path)
        os.makedirs(extract_dir, exist_ok=True)
        checkpoint_path = self._get_checkpoint_path(extract_dir)
        checkpoint = self._load_checkpoint(checkpoint_path)
        doc = fitz.open(pdf_path)
        image_count = 0
        image_paths = []
        image_filenames = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image['image']
                    ext = base_image.get('ext', 'png')
                    image_name = f"pdfimg_{os.path.basename(pdf_path).replace('.', '_')}_p{page_num+1}_{img_index+1}.{ext}"
                    image_path = os.path.join(extract_dir, image_name)
                    if not os.path.exists(image_path):
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                    image_paths.append(image_path)
                    image_filenames.append(image_name)
                    image_count += 1
                    self.logger.debug(f"Extracted image: {image_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract image xref {xref} on page {page_num+1}: {e}")
        if not image_paths:
            self.logger.info("No images found in PDF.")
            return {'chunks': [], 'metadata': {'pdf_path': pdf_path, 'image_count': 0, 'error': 'No images found'}}
        # Process each image with checkpointing
        all_chunks = []
        all_metadata = {
            'pdf_path': pdf_path,
            'image_count': image_count,
            'image_files': image_paths,
        }
        from qa_system.document_processors import get_processor_for_file_type
        for idx, (image_path, image_name) in enumerate(zip(image_paths, image_filenames)):
            # Check checkpoint
            entry = checkpoint.get(image_name, {})
            if entry.get('status') == 'done' and 'output' in entry:
                self.logger.info(f"Skipping already processed image {image_name} (checkpointed)")
                # Restore output chunks
                for chunk in entry['output'].get('chunks', []):
                    # Defensive: add pdf_image_index, pdf_path, extracted_image_path
                    chunk['metadata']['pdf_image_index'] = idx
                    chunk['metadata']['pdf_path'] = pdf_path
                    chunk['metadata']['extracted_image_path'] = image_path
                all_chunks.extend(entry['output'].get('chunks', []))
                continue
            self.logger.info(f"Processing extracted image {idx+1}/{len(image_paths)}: {image_path}")
            image_processor = get_processor_for_file_type(
                image_path, self.config, query_processor=self.query_processor
            )
            try:
                result = image_processor.process(image_path)
                for chunk in result.get('chunks', []):
                    chunk['metadata']['pdf_image_index'] = idx
                    chunk['metadata']['pdf_path'] = pdf_path
                    chunk['metadata']['extracted_image_path'] = image_path
                all_chunks.extend(result.get('chunks', []))
                checkpoint[image_name] = {
                    'status': 'done',
                    'output': result,
                    'error': None
                }
                self.logger.info(f"Image {image_name} processed and checkpointed.")
            except Exception as e:
                checkpoint[image_name] = {
                    'status': 'error',
                    'output': None,
                    'error': str(e)
                }
                self.logger.error(f"Error processing image {image_name}: {e}")
            self._save_checkpoint(checkpoint_path, checkpoint)
        all_metadata['chunk_count'] = len(all_chunks)
        return {'chunks': all_chunks, 'metadata': all_metadata} 