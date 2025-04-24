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
from qa_system.embedding_system import EmbeddingSystem

class QuerySystem:
    """Handles semantic search and response generation."""
    
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
            self.vector_store = VectorStore(config_path)
            self.embedding_system = EmbeddingSystem(config_path)
            
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
            if not query_text or not query_text.strip():
                raise ValueError("Empty query provided")
                
            start_time = time.time()
            stats = {
                'start_time': datetime.utcnow().isoformat(),
                'query_length': len(query_text)
            }
            
            # Generate query embedding
            query_embedding = self.embedding_system.generate_embedding(query_text)
            
            # Search vector store
            search_results = self.vector_store.query_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Extract relevant chunks
            chunks = []
            sources = []
            for result in search_results['matches']:
                chunks.append(result['metadata']['chunk_text'])
                sources.append({
                    'file_path': result['metadata']['path'],
                    'chunk_number': result['metadata']['chunk_number'],
                    'similarity': result['similarity']
                })
            
            # Generate response using Gemini
            context = "\n\n".join(chunks)
            prompt = f"""Based on the following context, answer the question: {query_text}

Context:
{context}

Answer:"""
            
            response = self.model.generate_content(prompt)
            
            # Calculate confidence score based on source similarities
            confidence = sum(source['similarity'] for source in sources) / len(sources) if sources else 0.0
            
            # Update statistics
            stats.update({
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': time.time() - start_time,
                'chunks_used': len(chunks),
                'confidence': confidence
            })
            
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
            self.logger.error(
                f"Error processing query: {str(e)}",
                extra={
                    'component': 'query_system',
                    'operation': 'query',
                    'query_text': query_text,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
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
                raise ValueError("Empty message list provided")
                
            start_time = time.time()
            stats = {
                'start_time': datetime.utcnow().isoformat(),
                'message_count': len(messages)
            }
            
            # Get the last user message as the query
            last_user_message = None
            for message in reversed(messages):
                if message['role'] == 'user':
                    last_user_message = message['content']
                    break
            
            if not last_user_message:
                raise ValueError("No user message found in conversation")
            
            # Generate query embedding
            query_embedding = self.embedding_system.generate_embedding(last_user_message)
            
            # Search vector store
            search_results = self.vector_store.query_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Extract relevant chunks
            chunks = []
            sources = []
            for result in search_results['matches']:
                chunks.append(result['metadata']['chunk_text'])
                sources.append({
                    'file_path': result['metadata']['path'],
                    'chunk_number': result['metadata']['chunk_number'],
                    'similarity': result['similarity']
                })
            
            # Generate response using Gemini
            context = "\n\n".join(chunks)
            chat = self.model.start_chat(history=[
                {'role': msg['role'], 'parts': [msg['content']]}
                for msg in messages[:-1]  # Exclude last message
            ])
            
            prompt = f"""Based on the following context and chat history, respond to: {last_user_message}

Context:
{context}

Response:"""
            
            response = chat.send_message(prompt)
            
            # Calculate confidence score based on source similarities
            confidence = sum(source['similarity'] for source in sources) / len(sources) if sources else 0.0
            
            # Update statistics
            stats.update({
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': time.time() - start_time,
                'chunks_used': len(chunks),
                'confidence': confidence
            })
            
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
            self.logger.error(
                f"Error processing chat: {str(e)}",
                extra={
                    'component': 'query_system',
                    'operation': 'chat',
                    'message_count': len(messages),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise 