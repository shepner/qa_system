"""
@file: image_processor.py
Image captioning processor using Gemini API.

This module defines ImageDocumentProcessor, which uploads an image to the Gemini API and retrieves a caption for the image. It also supports generating tags and extracting URLs from the caption.
"""

import os
import logging
from dotenv import load_dotenv
from google import genai
from PIL import Image
import re
from qa_system.query.keywords import derive_keywords
from qa_system.query.image_caption import generate_image_caption

class ImageDocumentProcessor:
    """
    Processor for image files using Gemini API for captioning, tag generation, and URL extraction.
    
    Methods:
        process(file_path):
            Uploads the image, gets a caption, generates tags, extracts URLs, and returns structured output.
    """
    def __init__(self, query_processor=None):
        load_dotenv()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        self.client = genai.Client(api_key=self.api_key)
        self.query_processor = query_processor  # Should have .llm attribute for derive_keywords

    def process(self, file_path):
        """
        Process an image file, upload to Gemini, and get a caption, tags, and URLs.

        Args:
            file_path (str): Path to the image file.
        Returns:
            dict: {'chunks': [...], 'metadata': {...}}
        """
        self.logger.debug(f"Starting process for file: {file_path}")
        metadata = {'file_path': file_path}
        metadata['path'] = os.path.abspath(file_path)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                metadata['image_width'] = width
                metadata['image_height'] = height
                metadata['image_format'] = img.format
                metadata['color_profile'] = img.mode
            self.logger.debug(f"Image opened successfully: width={width}, height={height}, format={img.format}, color_profile={img.mode}")
        except Exception as e:
            self.logger.error(f"Failed to open image: {e}")
            metadata['error'] = str(e)
            return {'chunks': [], 'metadata': metadata}

        # Defensive check for query_processor
        if self.query_processor is None:
            self.logger.error("query_processor is not set. Cannot proceed with captioning.")
            metadata['error'] = "Internal error: query_processor not set."
            return {'chunks': [], 'metadata': metadata}
        if not hasattr(self.query_processor, 'llm'):
            self.logger.error("query_processor is missing required .llm attribute.")
            metadata['error'] = "Internal error: query_processor missing .llm attribute."
            return {'chunks': [], 'metadata': metadata}

        # Generate caption (for embedding) using GeminiLLM.upload_and_generate
        self.logger.debug("Calling GeminiLLM.upload_and_generate for caption...")
        caption_prompt = (
            "Describe exactly and only what is visible in this image using a literal sentence. "
            "Do not guess, infer, or add any details that are not clearly present. "
            "Do not be creative or embellish."
        )
        try:
            caption = self.query_processor.llm.upload_and_generate(
                file_path=file_path,
                prompts=caption_prompt
            )
            gemini_file_uri = None  # Not used in this pipeline, but kept for compatibility
        except Exception as e:
            metadata['error'] = f"Caption error: {e}"
            self.logger.error(f"Caption generation failed for file: {file_path}: {e}")
            return {'chunks': [], 'metadata': metadata}
        if not caption:
            metadata['error'] = "Caption error: Failed to generate caption."
            self.logger.error("Caption generation failed for file: %s", file_path)
            return {'chunks': [], 'metadata': metadata}
        metadata['gemini_file_uri'] = gemini_file_uri
        metadata['caption'] = caption
        self.logger.debug(f"Caption generated: {caption}")

        # Generate tags from caption using derive_keywords in 'keywords' mode
        self.logger.debug("Generating tags from caption...")
        tags = set()
        try:
            tags = derive_keywords(self.query_processor, caption, mode='image_tags', logger=self.logger)
        except Exception as e:
            self.logger.error(f"Failed to derive keywords: {e}")
        metadata['tags'] = list(tags)
        self.logger.debug(f"Tags generated: {tags}")

        # Extract URLs from caption
        self.logger.debug("Extracting URLs from caption...")
        urls = self._extract_urls(caption)
        metadata['urls'] = urls
        self.logger.debug(f"URLs extracted: {urls}")

        chunk = {
            'text': caption,
            'metadata': {
                'chunk_type': 'caption',
                'file_path': file_path,
                'image_width': metadata.get('image_width'),
                'image_height': metadata.get('image_height'),
                'image_format': metadata.get('image_format'),
                'color_profile': metadata.get('color_profile'),
                'tags': list(tags),
                'urls': urls,
            }
        }
        # Clean None values from metadata and chunk['metadata']
        metadata = {k: v for k, v in metadata.items() if v is not None}
        chunk_metadata = {k: v for k, v in chunk['metadata'].items() if v is not None}
        chunk['metadata'] = chunk_metadata
        self.logger.debug(f"Returning processed chunk and metadata for file: {file_path}")
        return {
            'chunks': [chunk],
            'metadata': metadata
        }

    def _extract_urls(self, text):
        """
        Extract URLs from the given text using regex.
        """
        if not text:
            return []
        url_pattern = re.compile(r"https?://[\w\.-]+(?:/[\w\.-]*)*")
        return url_pattern.findall(text) 