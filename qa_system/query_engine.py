"""
Query processing and answer generation using Google Gemini with thinking capabilities
"""
from typing import List, Dict, Any
import google.generativeai as genai
import os
import json
from pathlib import Path
import numpy as np

from .vector_store import VectorStore

class QueryEngine:
    """Handles query processing and answer generation using Gemini with thinking capabilities."""
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore):
        """Initialize the query engine.
        
        Args:
            config: Configuration dictionary
            vector_store: Vector store instance
        """
        self.config = config
        self.vector_store = vector_store
        
        # Initialize Gemini client with API key from security config
        api_key = config.get("SECURITY", {}).get("API_KEY") or config.get("API_KEY")
        genai.configure(api_key=api_key)
        self.generation_model = genai.GenerativeModel('models/gemini-2.5-pro-exp-03-25')
        
        # Query processing settings from new config structure
        query_config = config.get("QUERY_ENGINE", {})
        self.max_context_docs = query_config.get("MAX_CONTEXT_DOCS", 10)
        self.min_relevance_score = query_config.get("MIN_RELEVANCE_SCORE", 0.4)
        self.enable_query_expansion = query_config.get("ENABLE_QUERY_EXPANSION", False)
        self.max_tokens = query_config.get("MAX_TOKENS", 2048)
        self.temperature = query_config.get("TEMPERATURE", 0.3)
        self.top_p = query_config.get("TOP_P", 0.95)
        self.response_mode = query_config.get("RESPONSE_MODE", "comprehensive")
        
    async def process_query(self, question: str) -> Dict:
        """Process a question and generate an answer.
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer and sources
        """
        # Expand query for better semantic matching
        expanded_queries = await self._expand_query(question) if self.enable_query_expansion else [question]
        
        # Collect relevant documents from all expanded queries
        all_relevant_docs = []
        for query in expanded_queries:
            # Generate question embedding
            query_embedding = await self._get_embedding(query)
            
            # Find relevant documents
            docs = await self.vector_store.similarity_search(
                query_embedding,
                k=self.max_context_docs
            )
            all_relevant_docs.extend(docs)
        
        # Filter and deduplicate documents
        relevant_docs = self._filter_and_dedupe_docs(all_relevant_docs)
        
        # Prepare context for Gemini
        context = self._prepare_context(relevant_docs)
        
        # Generate answer using Gemini
        answer = await self._generate_answer(question, context, relevant_docs)
        
        return {
            "question": question,
            "answer": answer["answer"],
            "sources": answer["sources"],
            "confidence": answer.get("confidence", None)
        }
        
    async def _expand_query(self, question: str) -> List[str]:
        """Expand the query to capture different semantic aspects."""
        prompt = f"""Given this question, generate 2-3 alternative phrasings that capture different semantic aspects.
Focus on identifying key concepts and their relationships.

Question: {question}

Return ONLY a JSON array of strings with no additional text or explanation.
Example format: ["first phrasing", "second phrasing", "third phrasing"]"""
        
        try:
            response = self.generation_model.generate_content(prompt)
            text = response.text.strip()
            
            # Handle common formatting issues
            if not text.startswith('['):
                # Try to find JSON array in the response
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    text = text[start_idx:end_idx]
                else:
                    # If no JSON array found, create one with the original question
                    return [question]
                
            expanded = json.loads(text)
            if not isinstance(expanded, list):
                return [question]
            
            # Add original question and return unique queries
            expanded.append(question)
            return list(set(expanded))
        except Exception as e:
            print(f"Query expansion failed: {str(e)}")
            # Fallback to original question
            return [question]
            
    def _filter_and_dedupe_docs(self, docs: List[Dict]) -> List[Dict]:
        """Filter and deduplicate documents based on relevance and content."""
        # Sort by relevance score
        docs = sorted(docs, key=lambda x: 1 - (x["distance"] if x["distance"] is not None else 0), reverse=True)
        
        # Filter by minimum relevance score
        docs = [doc for doc in docs if 1 - (doc["distance"] if doc["distance"] is not None else 0) >= self.min_relevance_score]
        
        # Deduplicate based on content
        seen_contents = set()
        unique_docs = []
        for doc in docs:
            content = doc["content"]
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        return unique_docs[:self.max_context_docs]
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        result = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return result['embedding']
        
    def _prepare_context(self, relevant_docs: List[Dict]) -> str:
        """Prepare context from relevant documents."""
        # Sort documents by relevance score
        sorted_docs = sorted(
            relevant_docs,
            key=lambda x: 1 - (x["distance"] if x["distance"] is not None else 0),
            reverse=True
        )
        
        context_parts = []
        for doc in sorted_docs:
            relevance_score = 1 - (doc["distance"] if doc["distance"] is not None else 0)
            context_parts.append(
                f"Content from {doc['metadata']['filename']} (relevance: {relevance_score:.2f}):\n{doc['content']}\n"
            )
            
        return "\n".join(context_parts)
        
    async def _generate_answer(self, question: str, context: str, relevant_docs: List[Dict]) -> Dict:
        """Generate answer using Gemini with structured thinking."""
        prompt = f"""You are a knowledgeable research assistant with access to specific context information.
Your goal is to provide well-reasoned, comprehensive answers based on the available information.

Think through this task step by step:

1. First, carefully analyze the provided context:
   - Identify key concepts and their relationships
   - Note any conflicting or complementary information
   - Consider the reliability and relevance of each source

2. Then, break down the question:
   - Identify the main concepts and requirements
   - Consider any implicit assumptions or related aspects
   - Determine what information is needed to provide a complete answer

3. Finally, formulate a clear and comprehensive answer:
   - Start with a direct response to the main question
   - Provide supporting evidence and explanations
   - Address any uncertainties or limitations
   - Include relevant examples or analogies if helpful

Context:
{context}

Question: {question}

Think through your response carefully and provide:
1. A clear, well-reasoned answer that:
   - Directly addresses the question
   - Explains your reasoning
   - Provides relevant examples or analogies
   - Acknowledges any uncertainties or limitations

2. The source documents used, including:
   - Filename
   - Relevance to specific parts of your answer
   - Any conflicting or complementary information between sources

3. A confidence score (0-1) based on:
   - How directly the context addresses the question
   - The completeness and coherence of the information
   - The reliability and relevance of the sources
   - The presence of any significant gaps or uncertainties

If you cannot answer based on the context, explain:
- What information is missing
- What assumptions would be needed
- What additional research might be helpful

Format your response as JSON with these keys: "answer", "sources", "confidence", "reasoning_path"
"""
        try:
            # Generate content with thinking capabilities
            response = self.generation_model.generate_content(prompt)
            
            if not response:
                return {
                    "answer": "Error: No response from the model",
                    "sources": [],
                    "confidence": 0,
                    "reasoning_path": []
                }

            # Parse response
            try:
                parsed_response = json.loads(response.text)
            except json.JSONDecodeError:
                # Extract JSON from text if needed
                start_idx = response.text.find("{")
                end_idx = response.text.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response.text[start_idx:end_idx]
                    try:
                        parsed_response = json.loads(json_str)
                    except json.JSONDecodeError:
                        parsed_response = {
                            "answer": response.text,
                            "sources": [],
                            "confidence": 0.5,
                            "reasoning_path": []
                        }
                else:
                    parsed_response = {
                        "answer": response.text,
                        "sources": [],
                        "confidence": 0.5,
                        "reasoning_path": []
                    }
            
            # Clean up and format sources
            sources = []
            for doc in relevant_docs:
                filepath = Path(doc["metadata"]["filename"])
                if filepath.is_file() and not filepath.name.startswith('.'):
                    sources.append({
                        "filename": filepath.name,
                        "chunk_index": doc["metadata"]["chunk_index"],
                        "relevance_score": 1 - (doc["distance"] if doc["distance"] is not None else 0)
                    })
            
            return {
                "answer": parsed_response.get("answer", response.text),
                "sources": sources,
                "confidence": parsed_response.get("confidence", 0.5),
                "reasoning_path": parsed_response.get("reasoning_path", [])
            }
            
        except Exception as e:
            print(f"Error in _generate_answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": 0,
                "reasoning_path": []
            } 