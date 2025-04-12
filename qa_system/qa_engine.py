"""
QA Engine for processing questions and generating answers.
"""
from typing import Dict, Any, Optional, List
from .query_engine import QueryEngine
from .vector_store import VectorStore
from .document_processor import DocumentProcessor
import logging
from pathlib import Path
import os
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from .document_store import DocumentStore
from .embedding_generator import EmbeddingGenerator

# Get logger for this module
logger = logging.getLogger(__name__)

@dataclass
class QAMetrics:
    """Metrics for QA operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    total_documents: int = 0
    total_chunks: int = 0

class QAEngineInterface(ABC):
    """Interface defining QA Engine capabilities."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the QA Engine components."""
        pass
        
    @abstractmethod
    async def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a question."""
        pass
        
    @abstractmethod
    def list_documents(self) -> List[Dict]:
        """List all documents in the vector store."""
        pass

class QAEngine(QAEngineInterface):
    """QA Engine that handles question answering using a query engine and vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the QA Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vector_store = VectorStore(config)
        self.document_store = DocumentStore(config)
        self.document_processor = DocumentProcessor(config)  # Initialize DocumentProcessor
        self.embedding_generator = EmbeddingGenerator(config)  # Initialize EmbeddingGenerator
        self.query_engine = None
        self._initialized = False
        self.metrics = QAMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit = config.get('RATE_LIMIT', {
            'requests_per_minute': 60,
            'burst_limit': 10
        })
        self.request_timestamps = []
        
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits.
        
        Returns:
            bool: True if request can proceed, False if rate limited
        """
        now = time.time()
        minute_ago = now - 60
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check rate limits
        if len(self.request_timestamps) >= self.rate_limit['requests_per_minute']:
            return False
            
        # Check burst limit (requests in last second)
        recent_requests = len([ts for ts in self.request_timestamps if ts > now - 1])
        if recent_requests >= self.rate_limit['burst_limit']:
            return False
            
        self.request_timestamps.append(now)
        return True
        
    def _validate_request(self, question: str) -> None:
        """Validate incoming request parameters.
        
        Args:
            question: The question to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
            
        if len(question) > self.config.get('MAX_QUESTION_LENGTH', 1000):
            raise ValueError("Question exceeds maximum length")
            
    def _update_metrics(self, start_time: float, success: bool) -> None:
        """Update operation metrics.
        
        Args:
            start_time: Operation start timestamp
            success: Whether operation succeeded
        """
        duration = time.time() - start_time
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            
        # Update average response time
        if self.metrics.total_requests == 1:
            self.metrics.avg_response_time = duration
        else:
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + duration)
                / self.metrics.total_requests
            )
            
    async def initialize(self) -> None:
        """Initialize the QA Engine components."""
        try:
            logger.info("Initializing QA Engine components...")
            
            # Initialize document store first
            if not self.document_store:
                logger.info("Initializing document store...")
                self.document_store = DocumentStore(self.config)
            
            # Initialize vector store
            if not self._initialized:
                logger.info("Initializing vector store...")
                await self.vector_store.initialize()
                
            # Initialize query engine with vector store
            if not self.query_engine:
                logger.info("Initializing query engine...")
                self.query_engine = QueryEngine(self.config, self.vector_store)
                
            self._initialized = True
            logger.info("QA Engine initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize QA Engine: {str(e)}", exc_info=True)
            raise RuntimeError(f"QA Engine initialization failed: {str(e)}")

    async def get_answer(self, question: str) -> str:
        """Get answer for a question using relevant documents."""
        try:
            # Get relevant documents
            relevant_docs = await self._get_relevant_documents(question)
            if not relevant_docs:
                return "No relevant documents found to answer the question."

            # Format documents for the prompt
            context = "\n\n".join(relevant_docs)
            
            # Generate answer using the language model
            prompt = self._format_prompt(question, context)
            answer = await self._generate_answer(prompt)
            
            return answer
        except Exception as e:
            self.logger.error(f"Error getting answer: {str(e)}")
            return f"Error getting answer: {str(e)}"

    async def _get_relevant_documents(self, question: str) -> List[str]:
        """Get documents relevant to the question."""
        try:
            # Get document embeddings from vector store
            doc_ids = await self.vector_store.search(question, self.config.max_documents)
            
            # Get full documents
            docs = []
            for doc_id in doc_ids:
                doc = await self.document_store.get_document(doc_id)
                if doc:
                    docs.append(doc)
            
            return docs
        except Exception as e:
            self.logger.error(f"Error getting relevant documents: {str(e)}")
            return []

    def list_documents(self) -> List[Dict]:
        """List all documents in the vector store.
        
        Returns:
            List of document metadata
            
        Raises:
            RuntimeError: If engine not initialized
        """
        if not self._initialized:
            return []
        return self.vector_store.list_documents() if self.vector_store else []

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dict containing system metrics
        """
        if not self._initialized:
            raise RuntimeError("QA Engine not initialized. Call initialize() first.")
            
        try:
            # Update document counts
            docs = self.list_documents()
            self.metrics.total_documents = len(docs)
            self.metrics.total_chunks = await self.vector_store.get_index_size()
            
            return asdict(self.metrics)
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise

    async def add_document(self, file_path: str) -> Dict[str, Any]:
        """Add a document to the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document metadata including chunks and embeddings
        """
        if not self._initialized:
            logger.info("QA Engine not initialized, initializing now...")
            await self.initialize()

        try:
            # Process document into chunks using existing processor instance
            doc_metadata = await self.document_processor.process_document(file_path)
            
            # If document processor returns None (unsupported file type), return early
            if doc_metadata is None:
                return {
                    "status": "skipped",
                    "reason": "unsupported_file_type",
                    "file_path": file_path,
                    "chunks": 0
                }
            
            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(doc_metadata['chunks'])} chunks from {file_path}")
            chunks_with_embeddings = await self.embedding_generator.generate_document_embeddings(doc_metadata["chunks"])
            
            # Extract chunks and embeddings for vector store
            chunks = [chunk["text"] for chunk in chunks_with_embeddings]
            embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
            
            # Store embeddings in vector store
            logger.info(f"Storing {len(chunks)} embeddings for {file_path}")
            # Merge all necessary metadata fields
            metadata = {
                "document_id": doc_metadata["metadata"]["document_id"],
                "id": doc_metadata["metadata"]["id"],
                "path": doc_metadata["metadata"]["path"],  # Ensure path is included
                "file_type": doc_metadata["metadata"]["file_type"],  # Include file_type as it's required
                **doc_metadata["metadata"]  # Include any additional metadata
            }
            await self.vector_store.store_embeddings(
                doc_id=doc_metadata["metadata"]["id"],
                embeddings=embeddings,
                chunks=chunks,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "file_path": file_path,
                "chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {str(e)}")
            raise

    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the vector store.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if successful
        """
        if not self.vector_store:
            raise RuntimeError("QA Engine not initialized. Call initialize() first.")
            
        return await self.vector_store.remove_document(doc_id)

    async def add_documents(self, path: str) -> Dict[str, Any]:
        """Add documents from a file or directory path.
        
        Args:
            path: Path to a file or directory
            
        Returns:
            Dict containing processing statistics
        """
        if not self._initialized:
            raise RuntimeError("QAEngine not initialized")
            
        stats = {
            "processed": 0,
            "skipped": 0,
            "unchanged": 0,
            "chunks": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                # Process single file
                result = await self.add_document(str(path_obj))
                self._update_stats(stats, result)
            elif path_obj.is_dir():
                # Recursively process directory
                for file_path in path_obj.rglob("*"):
                    if file_path.is_file() and not self._should_ignore(file_path):
                        try:
                            result = await self.add_document(str(file_path))
                            self._update_stats(stats, result)
                        except Exception as e:
                            logger.error(f"Failed to process file {file_path}: {str(e)}")
                            stats["skipped"] += 1
            else:
                raise ValueError(f"Path does not exist: {path}")
                
            stats["end_time"] = datetime.now().isoformat()
            return stats
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            stats["error"] = str(e)
            stats["end_time"] = datetime.now().isoformat()
            return stats
            
    def _update_stats(self, stats: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update statistics with results from processing a file.
        
        Args:
            stats: Statistics dict to update
            result: Result dict from processing a file
        """
        stats["processed"] += result.get("processed", 0)
        stats["skipped"] += result.get("skipped", 0)
        stats["unchanged"] += result.get("unchanged", 0)
        stats["chunks"] += result.get("chunks", 0)
        
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file should be ignored during processing.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file should be ignored, False otherwise
        """
        try:
            if not self.document_processor:
                logger.warning("Document processor not initialized, ignoring file")
                return True
                
            return self.document_processor.should_exclude(str(file_path))
            
        except Exception as e:
            logger.error(f"Error checking if file should be ignored: {str(e)}")
            return True  # Fail safe - ignore file if there's an error
        
    async def get_status(self) -> Dict[str, Any]:
        """Get system status information.
        
        Returns:
            Dict containing status information
        """
        if not self._initialized:
            raise RuntimeError("QAEngine not initialized")
            
        try:
            # Get document stats
            docs = await self.vector_store.list_documents()
            total_docs = len(docs)
            
            # Get vector store stats
            index_size = await self.vector_store.get_index_size()
            
            # Get last updated time from most recent doc
            last_updated = None
            if docs:
                last_updated = max(doc.get("last_updated", "") for doc in docs)
                
            return {
                "total_documents": total_docs,
                "total_chunks": index_size,
                "index_size": index_size,
                "last_updated": last_updated,
                "initialized": self._initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                "error": str(e),
                "initialized": self._initialized
            }

    async def ask(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
        """Get an answer to a question, considering conversation history.
        
        Args:
            question: The question to answer
            conversation_history: Optional list of previous conversation messages
            
        Returns:
            Dict containing answer, sources, and confidence score
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Enhance question with conversation context if available
            enhanced_question = question
            if conversation_history:
                context = self._build_context_from_history(conversation_history)
                enhanced_question = f"{context}\n\nCurrent question: {question}"
            
            # Get answer from query engine
            response = await self.query_engine.get_answer(
                enhanced_question,
                self.vector_store
            )
            
            # Format response
            return {
                "answer": response.answer,
                "sources": [
                    {
                        "filename": source.filename,
                        "relevance_score": source.relevance_score,
                        "excerpt": source.excerpt
                    }
                    for source in response.sources
                ],
                "confidence": response.confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise

    def _build_context_from_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Build context string from conversation history.
        
        Args:
            conversation_history: List of previous conversation messages
            
        Returns:
            String containing formatted conversation context
        """
        # Only use last 5 messages to keep context manageable
        recent_history = conversation_history[-5:]
        
        context_parts = []
        for msg in recent_history:
            if "user" in msg:
                context_parts.append(f"User: {msg['user']}")
            if "assistant" in msg:
                context_parts.append(f"Assistant: {msg['assistant']}")
                
        if not context_parts:
            return ""
            
        return "Previous conversation:\n" + "\n".join(context_parts)

    async def get_index_size(self) -> int:
        """Get the total number of chunks in the vector store.
        
        Returns:
            Integer count of chunks
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            return await self.vector_store.get_index_size()
        except Exception as e:
            logger.error(f"Error getting index size: {str(e)}")
            return 0

    async def cleanup(self) -> None:
        """Clean up QA Engine resources."""
        logger.info("Cleaning up QA Engine resources...")
        try:
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                await self.vector_store.cleanup()
                self.vector_store = None
            else:
                logger.debug("Vector store already cleaned up or not initialized")
            
            if hasattr(self, 'query_engine') and self.query_engine is not None:
                await self.query_engine.cleanup()
                self.query_engine = None
            else:
                logger.debug("Query engine already cleaned up or not initialized")
                
            if hasattr(self, 'document_store') and self.document_store is not None:
                await self.document_store.cleanup()
                self.document_store = None
            else:
                logger.debug("Document store already cleaned up or not initialized")
                
            self._initialized = False
            logger.info("QA Engine cleanup complete")
            
        except Exception as e:
            # Log error but don't re-raise to allow other cleanup processes to continue
            logger.error(f"Error during QA Engine cleanup: {str(e)}", exc_info=True) 