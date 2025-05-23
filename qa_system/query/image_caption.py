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
    Generate a caption for an image file using the Gemini LLM API.

    Args:
        processor: QueryProcessor or object with .llm and .llm.client attributes (GeminiLLM and genai.Client).
        image_path (str): Path to the image file to caption.
        logger (Optional[logging.Logger]): Optional logger for debug and error messages.

    Returns:
        Tuple[str or None, str or None]: (caption, gemini_file_uri). Returns (None, None) on error.
    """
    max_retries = 5
    delay = 10  # start with 10 seconds
    retry_start = time.monotonic()
    for attempt in range(max_retries):
        try:
            if logger:
                logger.info(f"Uploading image file to Gemini: {image_path}")
                logger.info(f"Gemini API client: {processor.llm.client}, API key: {_redact_key(getattr(processor.llm.client, 'api_key', ''))}")
            my_file = processor.llm.client.files.upload(file=image_path)
            gemini_file_uri = getattr(my_file, 'uri', None)
            break
        except Exception as e:
            elapsed = time.monotonic() - retry_start
            msg = str(e)
            if logger:
                logger.error(f"Exception during image upload, attempt {attempt+1}: {type(e).__name__}: {msg}\n{traceback.format_exc()}")
            if 'RESOURCE_EXHAUSTED' in msg or '429' in msg:
                if logger:
                    logger.warning(f"Rate limit hit (429/RESOURCE_EXHAUSTED) during upload. Sleeping for {delay} seconds before retrying (attempt {attempt+1}/{max_retries}), elapsed {elapsed:.1f}s.")
                time.sleep(delay)
                delay *= 2
                continue
            if logger:
                logger.error(f"Failed to upload image to Gemini: {e}")
            return None, None
    else:
        if logger:
            logger.error("Max retries exceeded for Gemini API upload due to rate limiting.")
        raise RateLimitError("Max retries exceeded for Gemini API upload due to rate limiting.")

    delay = 10  # reset delay for caption step
    retry_start = time.monotonic()
    for attempt in range(max_retries):
        try:
            if logger:
                logger.info("Requesting image caption from Gemini API...")
            caption_response = processor.llm.client.models.generate_content(
                model=getattr(processor.llm, 'model_name', 'gemini-2.0-flash'),
                contents=[
                    my_file,
                    "Provide a single, direct, descriptive caption for this image. Do not provide options or commentary. Be factual and specific."
                ]
            )
            caption = getattr(caption_response, 'text', None)
            if not caption:
                raise ValueError("No caption returned from Gemini API.")
            return caption, gemini_file_uri
        except Exception as e:
            elapsed = time.monotonic() - retry_start
            msg = str(e)
            if logger:
                logger.error(f"Exception during caption, attempt {attempt+1}: {type(e).__name__}: {msg}\n{traceback.format_exc()}")
            if 'RESOURCE_EXHAUSTED' in msg or '429' in msg:
                if logger:
                    logger.warning(f"Rate limit hit (429/RESOURCE_EXHAUSTED) during caption. Sleeping for {delay} seconds before retrying (attempt {attempt+1}/{max_retries}), elapsed {elapsed:.1f}s.")
                time.sleep(delay)
                delay *= 2
                continue
            if logger:
                logger.error(f"Failed to get caption from Gemini: {e}")
            return None, gemini_file_uri
    else:
        if logger:
            logger.error("Max retries exceeded for Gemini API caption due to rate limiting.")
        raise RateLimitError("Max retries exceeded for Gemini API caption due to rate limiting.") 