import logging
from typing import List, Optional, Any, Dict
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import QASystemError, QueryError, EmbeddingError

class Source:
    def __init__(self, document: str, chunk: str, similarity: float, context: str = '', metadata: Optional[dict] = None):
        self.document = document
        self.chunk = chunk
        self.similarity = similarity
        self.context = context
        self.metadata = metadata or {}

class QueryResponse:
    def __init__(self, text: str, sources: List[Source], confidence: float = 1.0, processing_time: float = 0.0, error: Optional[str] = None, success: bool = True):
        self.text = text
        self.sources = sources
        self.confidence = confidence
        self.processing_time = processing_time
        self.error = error
        self.success = success

class QueryProcessor:
    """
    Handles semantic search queries and generates contextual responses using embeddings and the vector store.
    Supports dependency injection for embedding_generator and vector_store for testability.
    """
    def __init__(self, config, embedding_generator=None, vector_store=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called QueryProcessor.__init__(config={config})")
        self.config = config
        self.embedding_generator = embedding_generator if embedding_generator is not None else EmbeddingGenerator(config)
        self.vector_store = vector_store if vector_store is not None else ChromaVectorStore(config)

    def process_query(self, query: str) -> QueryResponse:
        """
        Process a semantic search query and return a response object.
        Args:
            query: The user query string
        Returns:
            QueryResponse object with answer, sources, and metadata
        """
        import time
        start_time = time.time()
        try:
            if not query or not isinstance(query, str):
                raise QASystemError("Query must be a non-empty string.")
            # Generate embedding for the query
            embedding_result = self.embedding_generator.generate_embeddings([query], metadata={"task_type": "RETRIEVAL_QUERY"})
            vectors = embedding_result.get('vectors', [])
            if not vectors or not isinstance(vectors, list):
                raise EmbeddingError("Failed to generate embedding for query.")
            query_vector = vectors[0]
            # Query the vector store
            results = self.vector_store.query(query_vector)
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] if 'distances' in results else []
            # Build sources list
            sources = []
            for i, doc_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ''
                meta = metadatas[i] if i < len(metadatas) else {}
                sim = 1.0 - distances[i] if i < len(distances) else 0.0
                context = doc[:200]  # Truncate for context
                sources.append(Source(document=meta.get('path', doc_id), chunk=doc, similarity=sim, context=context, metadata=meta))
            # Generate response text (simple: return top doc chunk or summary)
            response_text = sources[0].chunk if sources else "No relevant documents found."
            confidence = sources[0].similarity if sources else 0.0
            processing_time = time.time() - start_time
            return QueryResponse(
                text=response_text,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                error=None,
                success=True
            )
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return QueryResponse(
                text="",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                error=str(e),
                success=False
            )
