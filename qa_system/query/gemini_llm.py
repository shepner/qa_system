import logging
import os
from typing import Optional
from dotenv import load_dotenv
from qa_system.exceptions import RateLimitError

try:
    from google import genai
except ImportError:
    genai = None  # Will error at runtime if used

"""
@file: gemini_llm.py
Wrapper for interacting with the Gemini LLM API using the google-generativeai package.

This module provides the GeminiLLM class, which encapsulates configuration-driven access to Gemini LLM models.
It supports system/user prompts, configurable generation parameters, and robust error handling.

Environment:
    - Requires GEMINI_API_KEY to be set in the environment (via .env or system env).
    - Requires the google-generativeai package (imported as google.genai).

Typical usage example:
    config = ...  # Your config object with get_nested method
    llm = GeminiLLM(config)
    response = llm.generate_response("What is the capital of France?", system_prompt="You are a helpful assistant.")
"""

class GeminiLLM:
    """
    Wrapper for Gemini LLM API calls, supporting system prompt and config-driven parameters.

    Attributes:
        logger (logging.Logger): Logger for the class.
        config: Configuration object with get_nested method.
        gemini_api_key (str): API key for Gemini LLM, loaded from environment.
        client: Gemini API client instance.
        model_name (str): Model name to use for generation.
    """
    def __init__(self, config):
        """
        Initialize the GeminiLLM wrapper.

        Args:
            config: Configuration object with get_nested method for retrieving settings.

        Raises:
            RuntimeError: If GEMINI_API_KEY is not set in the environment.
            ImportError: If google-generativeai is not installed.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        if genai is None:
            raise ImportError("google-genai is not installed. Please install the google-generativeai package.")
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.model_name = self.config.get_nested('QUERY.MODEL_NAME', 'gemini-2.0-flash')

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from Gemini LLM.

        Args:
            user_prompt (str): The user prompt/question.
            system_prompt (Optional[str]): The system prompt/instructions (optional).
            temperature (Optional[float]): Sampling temperature (optional, overrides config).
            max_output_tokens (Optional[int]): Max tokens in response (optional, overrides config).
            **kwargs: Additional generation config parameters for the Gemini API.

        Returns:
            str: The LLM response text.

        Raises:
            RuntimeError: If the API key is missing.
            ImportError: If the google-generativeai package is not installed.
            RateLimitError: If the API call is rate limited (HTTP 429 or RESOURCE_EXHAUSTED).
            Exception: For any other errors during the LLM call.

        Example:
            >>> llm = GeminiLLM(config)
            >>> llm.generate_response("Hello!", system_prompt="You are a helpful assistant.")
        """
        # Compose the prompt as a list of messages if system_prompt is provided
        if system_prompt:
            contents = [
                {"role": "system", "parts": [system_prompt]},
                {"role": "user", "parts": [user_prompt]}
            ]
        else:
            contents = user_prompt

        gen_config = {
            "temperature": temperature if temperature is not None else float(self.config.get_nested('QUERY.TEMPERATURE', default=0.2)),
            "max_output_tokens": max_output_tokens if max_output_tokens is not None else int(self.config.get_nested('QUERY.MAX_TOKENS', default=1024)),
        }
        gen_config.update(kwargs)

        try:
            # --- INFO log before model call ---
            self.logger.info(f"Calling Gemini model '{self.model_name}' with prompt (user length: {len(str(user_prompt))} chars, system: {bool(system_prompt)})")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                generation_config=gen_config
            )
            # --- INFO log after model call ---
            self.logger.info(f"Gemini model '{self.model_name}' call complete. Response type: {type(response)}, response: {str(response)[:200]}...")
            return getattr(response, 'text', None) or str(response)
        except TypeError as e:
            # Fallback for older google-genai versions
            if "unexpected keyword argument 'generation_config'" in str(e):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents
                )
                return getattr(response, 'text', None) or str(response)
            else:
                self.logger.error(f"Gemini LLM call failed: {e}")
                raise
        except Exception as e:
            # Centralized rate limit handling
            # Try to detect HTTP 429 or gRPC RESOURCE_EXHAUSTED
            message = str(e)
            retry_after = None
            endpoint = f"gemini.models.generate_content:{self.model_name}"
            # Check for HTTP 429
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    # Try to get Retry-After header if present
                    retry_after = None
                    if hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('Retry-After')
                        try:
                            retry_after = float(retry_after)
                        except (TypeError, ValueError):
                            retry_after = None
                    self.logger.warning(f"Gemini LLM rate limited (HTTP 429) on {endpoint}. Retry-After: {retry_after}")
                    raise RateLimitError(message, endpoint=endpoint, retry_after=retry_after)
            # Check for gRPC RESOURCE_EXHAUSTED or explicit 429 in message
            if 'RESOURCE_EXHAUSTED' in message or '429' in message or 'rate limit' in message.lower():
                self.logger.warning(f"Gemini LLM rate limited (RESOURCE_EXHAUSTED/429) on {endpoint}.")
                raise RateLimitError(message, endpoint=endpoint)
            self.logger.error(f"Gemini LLM call failed: {e}")
            raise 

    def upload_and_generate(self, file_path: str, prompts, **kwargs):
        """
        Generic method to upload a file and generate content using Gemini.

        Args:
            file_path (str): Path to the file to upload (image, PDF, etc.).
            prompts (str or list): Prompt(s) to send to the model after the file.
            **kwargs: Additional parameters for the generate_content call.

        Returns:
            The Gemini model response (text or object).

        Raises:
            RateLimitError: If the API call is rate limited.
            Exception: For other errors.
        """
        try:
            file_handle = self.client.files.upload(file=file_path)
            # Compose contents: file handle first, then prompt(s)
            if isinstance(prompts, list):
                contents = [file_handle] + prompts
            else:
                contents = [file_handle, prompts]
            # --- INFO log before model call ---
            self.logger.info(f"Calling Gemini model '{self.model_name}'")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                **kwargs
            )
            return getattr(response, 'text', None) or str(response)
        except Exception as e:
            message = str(e)
            retry_after = None
            endpoint = f"gemini.models.generate_content:{self.model_name}"
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    if hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('Retry-After')
                        try:
                            retry_after = float(retry_after)
                        except (TypeError, ValueError):
                            retry_after = None
                    self.logger.warning(f"Gemini LLM rate limited (HTTP 429) on {endpoint}. Retry-After: {retry_after}")
                    raise RateLimitError(message, endpoint=endpoint, retry_after=retry_after)
            if 'RESOURCE_EXHAUSTED' in message or '429' in message or 'rate limit' in message.lower():
                self.logger.warning(f"Gemini LLM rate limited (RESOURCE_EXHAUSTED/429) on {endpoint}.")
                raise RateLimitError(message, endpoint=endpoint)
            self.logger.error(f"Gemini LLM call failed: {e}")
            raise 