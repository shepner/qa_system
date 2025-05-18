"""
image_caption.py
Image captioning utility for the QA system using Gemini LLM.

This module provides a function to generate a caption for an image file using the Gemini LLM API.
"""

import logging
from typing import Optional, Tuple


def generate_image_caption(processor, image_path: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a caption for an image file using the Gemini LLM API.

    Args:
        processor: QueryProcessor or object with .llm and .llm.client attributes (GeminiLLM and genai.Client).
        image_path (str): Path to the image file to caption.
        logger (Optional[logging.Logger]): Optional logger for debug and error messages.

    Returns:
        Tuple[str or None, str or None]: (caption, gemini_file_uri). Returns (None, None) on error.
    """
    try:
        if logger:
            logger.info(f"Uploading image file to Gemini: {image_path}")
        my_file = processor.llm.client.files.upload(file=image_path)
        gemini_file_uri = getattr(my_file, 'uri', None)
    except Exception as e:
        if logger:
            logger.error(f"Failed to upload image to Gemini: {e}")
        return None, None

    try:
        if logger:
            logger.info("Requesting image caption from Gemini API...")
        caption_response = processor.llm.client.models.generate_content(
            model=getattr(processor.llm, 'model_name', 'gemini-2.0-flash'),
            contents=[my_file, "Caption this image."]
        )
        caption = getattr(caption_response, 'text', None)
        if not caption:
            raise ValueError("No caption returned from Gemini API.")
        return caption, gemini_file_uri
    except Exception as e:
        if logger:
            logger.error(f"Failed to get caption from Gemini: {e}")
        return None, gemini_file_uri 