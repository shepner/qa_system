import logging
from typing import List, Dict, Any
import os
from qa_system.exceptions import EmbeddingError

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # Will error at runtime if used

class EmbeddingGenerator:
    """
    EmbeddingGenerator generates vector embeddings for document chunks and images.
    Integrates with configuration for model settings and batching.
    """
    def __init__(self, config):
        """
        Initialize the EmbeddingGenerator with configuration.
        Args:
            config: Configuration object (must support get_nested)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Called EmbeddingGenerator.__init__(config={config})")
        self.config = config
        self.model_name = self.config.get_nested('EMBEDDING_MODEL.MODEL_NAME', 'embedding-001')
        self.batch_size = self.config.get_nested('EMBEDDING_MODEL.BATCH_SIZE', 32)
        self.max_length = self.config.get_nested('EMBEDDING_MODEL.MAX_LENGTH', 3072)
        self.dimensions = self.config.get_nested('EMBEDDING_MODEL.DIMENSIONS', 768)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        if genai is None:
            raise ImportError("google-genai is not installed. Please install the google-generativeai package.")
        self.client = genai.Client(api_key=self.gemini_api_key)

    def generate_embeddings(self, texts: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of text chunks using Gemini API.
        Args:
            texts: List of text chunks to embed
            metadata: Metadata dictionary for the document
        Returns:
            Dictionary with keys: 'vectors', 'texts', 'metadata'
        """
        self.logger.debug(f"Called EmbeddingGenerator.generate_embeddings(texts=<len {len(texts)}>, metadata={metadata})")
        if not texts:
            self.logger.warning("No texts provided for embedding generation.")
            return {'vectors': [], 'texts': [], 'metadata': []}
        vectors = []
        batch_size = self.batch_size or 32
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Gemini API expects a list of strings
                result = self.client.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                # result.embeddings is a list of Embedding objects, each with a .values attribute
                # If only one input, result.embeddings may be a single Embedding object
                if hasattr(result, 'embeddings'):
                    embeddings = result.embeddings
                    # If only one embedding, wrap in list
                    if not isinstance(embeddings, list):
                        embeddings = [embeddings]
                    for emb in embeddings:
                        # emb.values is the vector
                        vectors.append(list(emb.values))
                else:
                    raise EmbeddingError("No embeddings returned from Gemini API.")
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
        return {
            'vectors': vectors,
            'texts': texts,
            'metadata': [metadata] * len(texts)
        }
