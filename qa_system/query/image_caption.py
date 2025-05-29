"""
image_caption.py
Image captioning utility for the QA system using Gemini LLM.

This module provides a function to generate a caption for an image file using the Gemini LLM API.
"""

import logging
from typing import Optional, Tuple
import time
from qa_system.exceptions import RateLimitError
import traceback
import signal
from PIL import Image


def _redact_key(key):
    if not key or len(key) < 8:
        return "<redacted>"
    return key[:4] + "..." + key[-4:]

def _handle_interrupt(signum, frame):
    logging.warning(f"Process interrupted by signal {signum}. Cleaning up and exiting.")
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)

def generate_image_caption(processor, image_path: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a caption for an image file using the Gemini LLM API via the generic GeminiLLM interface.

    Args:
        processor: QueryProcessor or object with .llm attribute (GeminiLLM instance).
        image_path (str): Path to the image file to caption.
        logger (Optional[logging.Logger]): Optional logger for debug and error messages.

    Returns:
        Tuple[str or None, None]: (caption, None). Returns (None, None) on error.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            prompt = (
                "Describe exactly and only what is visible in this image using a literal sentence. "
                "Do not guess, infer, or add any details that are not clearly present. "
                "Do not be creative or embellish."
)
            caption = processor.llm.generate_response(
                user_prompt=prompt,
                system_prompt=None,
                contents=[img, prompt]
            )
            return caption, None
    except Exception as e:
        if logger:
            logger.error(f"Error generating image caption: {e}")
        return None, None 