"""
Embedding model implementation for generating text embeddings using Google's Gemini API.
"""

from typing import List, Dict, Any
import os
import re
import time
from datetime import datetime
import logging
import httpx
from google.cloud import aiplatform

class EmbeddingModel:
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding model with configuration."""
        self.config = config["EMBEDDING_MODEL"]
        self._setup_logging()
        self._setup_client()
        self._request_count = 0
        self._last_reset = datetime.now()
        self._window_requests = 0
        self._window_start = datetime.now()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = getattr(logging, self.config["LOG_LEVEL"])
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)

    def _setup_client(self) -> None:
        """Set up Google Cloud client."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
        
        aiplatform.init(project=project_id)
        self.client = aiplatform.TextEmbeddingModel.from_pretrained(self.config["MODEL_NAME"])

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration."""
        if self.config["PREPROCESSING"]["STRIP_HTML"]:
            # Simple HTML tag removal - can be enhanced with proper HTML parsing if needed
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.config["PREPROCESSING"]["NORMALIZE_WHITESPACE"]:
            # Normalize whitespace and newlines
            text = ' '.join(text.split())
        
        if self.config["PREPROCESSING"]["LOWERCASE"]:
            text = text.lower()
        
        return text

    def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        now = datetime.now()
        
        # Reset daily counter if needed
        if (now - self._last_reset).days > 0:
            self._request_count = 0
            self._last_reset = now
        
        # Check daily limit
        if self._request_count >= self.config["RATE_LIMITS"]["MAX_REQUESTS_PER_DAY"]:
            raise Exception("Daily request limit exceeded")
        
        # Reset window counter if needed
        window_seconds = self.config["RATE_LIMITS"]["RATE_LIMIT_WINDOW"]
        if (now - self._window_start).total_seconds() > window_seconds:
            self._window_requests = 0
            self._window_start = now
        
        # Check window limit
        if self._window_requests >= self.config["RATE_LIMITS"]["MAX_REQUESTS_PER_WINDOW"]:
            if self.config["RATE_LIMITS"]["ENABLE_THROTTLING"]:
                sleep_time = window_seconds - (now - self._window_start).total_seconds()
                time.sleep(max(0, sleep_time))
                self._window_requests = 0
                self._window_start = datetime.now()
            else:
                raise Exception("Rate limit exceeded for current window")

    async def _retry_with_exponential_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        max_retries = self.config["ERROR_HANDLING"]["MAX_RETRIES"]
        base_delay = self.config["ERROR_HANDLING"]["RETRY_BASE_DELAY"]
        max_delay = self.config["ERROR_HANDLING"]["RETRY_MAX_DELAY"]
        retry_status_codes = self.config["ERROR_HANDLING"]["RETRY_STATUS_CODES"]
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in retry_status_codes or attempt == max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay)
                self.logger.warning(f"Request failed with status {e.response.status_code}. "
                                  f"Retrying in {delay} seconds...")
                time.sleep(delay)
            except Exception as e:
                if attempt == max_retries:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay)
                self.logger.warning(f"Request failed with error: {str(e)}. "
                                  f"Retrying in {delay} seconds...")
                time.sleep(delay)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Truncate texts to max length if needed
        max_length = self.config["MAX_LENGTH"]
        truncated_texts = [text[:max_length] for text in processed_texts]
        
        # Process in batches
        batch_size = self.config["BATCH_SIZE"]
        embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            
            # Check rate limits
            self._check_rate_limits()
            
            try:
                # Generate embeddings with retry
                batch_embeddings = await self._retry_with_exponential_backoff(
                    self.client.get_embeddings,
                    batch
                )
                
                # Update rate limit counters
                self._request_count += 1
                self._window_requests += 1
                
                # Log metrics if enabled
                if self.config["MONITORING"]["ENABLE_METRICS"]:
                    self.logger.info(
                        f"Generated embeddings for batch of {len(batch)} texts. "
                        f"Dimensions: {len(batch_embeddings[0])}"
                    )
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                raise
        
        return embeddings 