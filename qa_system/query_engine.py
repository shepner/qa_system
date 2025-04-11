"""
Query processing and answer generation using Google Gemini with thinking capabilities
"""
from typing import List, Dict, Any, Optional
import os
import json
import logging
from pathlib import Path
import google.generativeai as genai

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class QueryEngine:
    """Processes queries and generates answers using stored documents."""
    
    def __init__(self, config: Dict[str, Any], vector_store: Optional[VectorStore] = None):
        """Initialize the query engine.
        
        Args:
            config: Configuration dictionary containing settings
            vector_store: Optional VectorStore instance. If not provided, one will be created.
        """
        self.config = config
        self.vector_store = vector_store or VectorStore(config)
        
        # Load system prompt from config
        self.system_prompt = self._load_prompt_template()
        
        # Initialize model
        self._initialize_model()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the local config directory."""
        try:
            config_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "config"
            prompt_path = config_dir / "prompts.py"
            
            # Import the prompts module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("prompts", prompt_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load prompt template from {prompt_path}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module.ANSWER_GENERATION_PROMPT
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {str(e)}")
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")
        
    def _initialize_model(self):
        """Initialize the language model based on configuration."""
        try:
            # Get API key from config or environment
            api_key = (
                self.config.get("API_KEY") or
                self.config.get("SECURITY", {}).get("API_KEY") or
                os.getenv("API_KEY")
            )
            
            if not api_key:
                raise ValueError("No API key found in config or environment")
                
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model with configured parameters
            model_config = self.config.get("QUERY_ENGINE", {})
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config={
                    "temperature": model_config.get("TEMPERATURE", 0.7),
                    "top_p": model_config.get("TOP_P", 0.95),
                    "max_output_tokens": model_config.get("MAX_TOKENS", 2048)
                }
            )
            
            logger.info("Successfully initialized Gemini model")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
        
    async def generate_answer(self, question: str, context: List[str]) -> Dict[str, Any]:
        """Generate an answer for the given question using the provided context.
        
        Args:
            question: The user's question
            context: List of relevant document chunks
            
        Returns:
            Dictionary containing answer, sources, confidence score, and reasoning path
        """
        try:
            # Format the prompt with the question and context
            formatted_prompt = self.system_prompt.format(
                question=question,
                context="\n".join(context)
            )
            
            # Get response from model
            response = await self.model.generate_content(formatted_prompt)
            
            try:
                # Parse JSON response
                result = json.loads(response.text)
                return {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.0),
                    "reasoning_path": result.get("reasoning_path", [])
                }
            except json.JSONDecodeError:
                logger.error("Failed to parse model response as JSON")
                return {
                    "answer": response.text,
                    "sources": [],
                    "confidence": 0.0,
                    "reasoning_path": []
                }
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
            
    async def process_query(self, question: str) -> Dict[str, Any]:
        """Process a query and generate an answer using relevant context.
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing answer, sources, confidence and reasoning
        """
        try:
            # Search for relevant documents
            matches = await self.vector_store.search(question, k=self.config.get("TOP_K_RESULTS", 5))
            
            if not matches:
                return {
                    "answer": "I could not find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "reasoning": "No relevant documents found in knowledge base."
                }
            
            # Extract content and sources
            context = "\n\n".join(match["content"] for match in matches)
            sources = [match["metadata"].get("source", "Unknown") for match in matches]
            
            # Generate answer using context
            response = await self.generate_answer(question, context)
            response["sources"] = sources
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if hasattr(self, 'vector_store'):
                await self.vector_store.cleanup()
            logger.info("Query engine cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise 