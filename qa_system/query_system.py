"""Query system for semantic search and response generation.

This module handles:
- Query processing and embedding generation
- Semantic search using vector database
- Response generation using Gemini model
- Source attribution and confidence scoring
"""

import logging
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from qa_system.config import get_config
from qa_system.vector_system import VectorStore
from qa_system.embedding_system import EmbeddingGenerator
from .exceptions import (
    QASystemError,
    QueryError,
    ValidationError,
    handle_exception
)

class QuerySystem:
    """Handles querying the vector database with natural language queries."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the query system.
        
        Args:
            config_path: Optional path to configuration file
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Load configuration
            self.config = get_config(config_path)
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            
            # Initialize components
            self.vector_db = VectorStore(config_path)
            self.embedding_system = EmbeddingGenerator(config_path)
            
            # Get model configuration
            model_config = self.config.get_nested('EMBEDDING_MODEL', {})
            self.model_name = model_config.get('MODEL_NAME', 'models/embedding-001')
            
            # Initialize Gemini
            credentials = self.config.get_nested('SECURITY.GOOGLE_APPLICATION_CREDENTIALS')
            project_id = self.config.get_nested('SECURITY.GOOGLE_CLOUD_PROJECT')
            
            if not credentials or not project_id:
                raise RuntimeError("Missing required Google Cloud credentials")
                
            genai.configure(
                project_id=project_id,
                credentials=credentials
            )
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            self.logger.info(
                "Query system initialized",
                extra={
                    'component': 'query_system',
                    'model': self.model_name
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Query system initialization failed: {str(e)}",
                extra={
                    'component': 'query_system',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise RuntimeError(f"Query system initialization failed: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Process a query and generate a response.
        
        Args:
            query_text: The query text
            top_k: Optional number of similar documents to retrieve
            
        Returns:
            Dictionary containing:
            - response: Generated response text
            - sources: List of source documents used
            - confidence: Confidence score
            - stats: Processing statistics
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If processing fails
        """
        try:
            if not query_text:
                raise ValidationError("Query text cannot be empty")
            
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_system.generate_embedding(query_text)
            
            # Query vector store
            results = self.vector_db.query_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Generate response
            response = self._generate_response(query_text, results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(response, results)
            
            # Extract relevant sources
            sources = self._extract_sources(results)
            
            # Calculate stats
            duration = time.time() - start_time
            stats = {
                'duration_seconds': duration,
                'source_count': len(sources),
                'confidence': confidence
            }
            
            self.logger.info(
                "Query processing complete",
                extra={
                    'component': 'query_system',
                    'operation': 'query',
                    'stats': stats
                }
            )
            
            return {
                'response': response.text,
                'sources': sources,
                'confidence': confidence,
                'stats': stats
            }
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error processing query",
                reraise=False
            )
            raise QueryError(
                f"Failed to process query: {error_details['message']}"
            ) from e
    
    def chat(self, messages: List[Dict[str, str]], top_k: Optional[int] = None) -> Dict[str, Any]:
        """Process a chat conversation and generate a response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            top_k: Optional number of similar documents to retrieve
            
        Returns:
            Dictionary containing:
            - response: Generated response text
            - sources: List of source documents used
            - confidence: Confidence score
            - stats: Processing statistics
            
        Raises:
            ValueError: If messages are invalid
            RuntimeError: If processing fails
        """
        try:
            if not messages:
                raise ValidationError("Messages list cannot be empty")
            
            start_time = time.time()
            
            # Extract latest message
            latest_message = messages[-1].get('content', '')
            if not latest_message:
                raise ValidationError("Latest message content cannot be empty")
            
            # Generate embedding for latest message
            query_embedding = self.embedding_system.generate_embedding(latest_message)
            
            # Query vector store
            results = self.vector_db.query_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Generate response considering conversation history
            response = self._generate_chat_response(messages, results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(response, results)
            
            # Extract relevant sources
            sources = self._extract_sources(results)
            
            # Calculate stats
            duration = time.time() - start_time
            stats = {
                'duration_seconds': duration,
                'source_count': len(sources),
                'confidence': confidence,
                'message_count': len(messages)
            }
            
            self.logger.info(
                "Chat processing complete",
                extra={
                    'component': 'query_system',
                    'operation': 'chat',
                    'stats': stats
                }
            )
            
            return {
                'response': response.text,
                'sources': sources,
                'confidence': confidence,
                'stats': stats
            }
            
        except Exception as e:
            error_details = handle_exception(
                e,
                "Error processing chat",
                reraise=False
            )
            raise QueryError(
                f"Failed to process chat: {error_details['message']}"
            ) from e

    def _generate_embedding(self, text: str):
        # Implementation of _generate_embedding method
        pass

    def _generate_response(self, query_text: str, results: Dict[str, Any]):
        # Implementation of _generate_response method
        pass

    def _calculate_confidence(self, response: Dict[str, Any], results: Dict[str, Any]):
        # Implementation of _calculate_confidence method
        pass

    def _extract_sources(self, results: Dict[str, Any]):
        # Implementation of _extract_sources method
        pass

    def _generate_chat_response(self, messages: List[Dict[str, str]], results: Dict[str, Any]):
        # Implementation of _generate_chat_response method
        pass 