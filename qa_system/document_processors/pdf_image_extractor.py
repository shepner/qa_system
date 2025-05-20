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

class PDFImageExtractor:
    """
    Extracts all images from a PDF and processes them using ImageDocumentProcessor.
    """
    def __init__(self, config, logger: Optional[logging.Logger] = None, query_processor=None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.query_processor = query_processor

    def process(self, pdf_path: str, output_dir: str = './tmp') -> Dict[str, Any]:
        """
        Extract all images from the PDF and process them with the image ingestion pipeline.

        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save extracted images.
        Returns:
            dict: {'chunks': [...], 'metadata': {...}}
        """
        self.logger.info(f"Extracting images from PDF: {pdf_path}")
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_count = 0
        image_paths = []
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
                    image_path = os.path.join(output_dir, image_name)
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_paths.append(image_path)
                    image_count += 1
                    self.logger.debug(f"Extracted image: {image_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract image xref {xref} on page {page_num+1}: {e}")
        if not image_paths:
            self.logger.info("No images found in PDF.")
            return {'chunks': [], 'metadata': {'pdf_path': pdf_path, 'image_count': 0, 'error': 'No images found'}}
        # Process each image with the main pipeline
        all_chunks = []
        all_metadata = {
            'pdf_path': pdf_path,
            'image_count': image_count,
            'image_files': image_paths,
        }
        for idx, image_path in enumerate(image_paths):
            self.logger.info(f"Processing extracted image {idx+1}/{len(image_paths)}: {image_path}")
            from qa_system.document_processors import get_processor_for_file_type
            image_processor = get_processor_for_file_type(
                image_path, self.config, query_processor=self.query_processor
            )
            result = image_processor.process(image_path)
            for chunk in result.get('chunks', []):
                chunk['metadata']['pdf_image_index'] = idx
                chunk['metadata']['pdf_path'] = pdf_path
                chunk['metadata']['extracted_image_path'] = image_path
            all_chunks.extend(result.get('chunks', []))
        all_metadata['chunk_count'] = len(all_chunks)
        return {'chunks': all_chunks, 'metadata': all_metadata} 