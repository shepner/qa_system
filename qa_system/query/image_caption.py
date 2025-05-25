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

def _reset_client(processor, logger=None):
    if logger:
        logger.info("Resetting Gemini API client due to 429/RESOURCE_EXHAUSTED.")
    try:
        del processor.llm.client
    except Exception:
        pass
    if logger:
        logger.info(f"Recreating Gemini API client with API key: <redacted>")
    from google import genai
    processor.llm.client = genai.Client(api_key=processor.llm.client.api_key)


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
    delay = 10
    max_delay = 600
    # Upload step
    retry_start = time.monotonic()
    attempt = 0
    while True:
        try:
            if logger:
                logger.info(f"Uploading image file to Gemini: {image_path}")
                logger.info(f"Gemini API client: {processor.llm.client}, API key: {_redact_key(getattr(processor.llm.client, 'api_key', ''))}")
            my_file = processor.llm.client.files.create(file=image_path)
            gemini_file_uri = getattr(my_file, 'uri', None)
            break
        except Exception as e:
            attempt += 1
            elapsed = time.monotonic() - retry_start
            msg = str(e)
            if logger:
                logger.error(f"Exception during Gemini image upload, attempt {attempt}: {type(e).__name__}: {msg}\n{traceback.format_exc()}")
            if (hasattr(e, 'args') and e.args and 'RESOURCE_EXHAUSTED' in str(e.args[0])) or 'RESOURCE_EXHAUSTED' in msg or '429' in msg:
                if logger:
                    logger.warning(f"Rate limit hit (429/RESOURCE_EXHAUSTED). Sleeping for {delay} seconds before retrying upload. Attempt {attempt}, elapsed {elapsed:.1f}s.")
                _reset_client(processor, logger)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            else:
                if logger:
                    logger.error(f"Non-rate-limit error, aborting upload. Exception: {type(e).__name__}: {msg}\n{traceback.format_exc()}")
                raise

    # Caption step
    delay = 10
    retry_start = time.monotonic()
    attempt = 0
    while True:
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
            attempt += 1
            elapsed = time.monotonic() - retry_start
            msg = str(e)
            if logger:
                logger.error(f"Exception during caption, attempt {attempt}: {type(e).__name__}: {msg}\n{traceback.format_exc()}")
            if 'RESOURCE_EXHAUSTED' in msg or '429' in msg:
                if logger:
                    logger.warning(f"Rate limit hit (429/RESOURCE_EXHAUSTED) during caption. Sleeping for {delay} seconds before retrying (attempt {attempt}), elapsed {elapsed:.1f}s.")
                _reset_client(processor, logger)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            if logger:
                logger.error(f"Failed to get caption from Gemini: {e}")
            return None, gemini_file_uri 