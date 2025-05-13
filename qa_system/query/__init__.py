import logging
from typing import List, Optional, Any, Dict
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import QASystemError, QueryError, EmbeddingError
from dotenv import load_dotenv
import os
from google import genai
import time

# Add a simple English stopword set for keyword filtering
STOPWORDS = {
    "the", "is", "at", "which", "on", "and", "a", "an", "to", "me", "what", "explain", "please", "of", "for", "in", "with", "by", "as", "it", "that", "this", "be", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "shall", "may", "might", "must", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our", "their"
}

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
    Handles semantic search queries and generates contextual responses using embeddings, vector store, and Gemini LLM.
    Implements hybrid scoring (semantic + metadata boosts) and config-driven parameters.
    """
    def __init__(self, config, embedding_generator=None, vector_store=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called QueryProcessor.__init__(config={config})")
        self.config = config
        self.embedding_generator = embedding_generator if embedding_generator is not None else EmbeddingGenerator(config)
        self.vector_store = vector_store if vector_store is not None else ChromaVectorStore(config)
        # Gemini setup
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model = self.config.get_nested('QUERY.MODEL_NAME', 'gemini-2.0-flash')

    def _apply_hybrid_scoring(self, sources: List[Source]) -> List[Source]:
        # Configurable boosts
        recency_boost = float(self.config.get_nested('QUERY.RECENCY_BOOST', default=1.0))
        tag_boost = float(self.config.get_nested('QUERY.TAG_BOOST', default=1.5))  # More aggressive by default
        source_boost = float(self.config.get_nested('QUERY.SOURCE_BOOST', default=1.0))
        now = time.time()
        # Fetch preferred sources ONCE
        preferred_sources = self.config.get_nested('QUERY.PREFERRED_SOURCES', default=[])
        for src in sources:
            boost = 1.0
            # Recency boost (if 'date' in metadata)
            date = src.metadata.get('date')
            if date:
                try:
                    # Assume date is ISO8601 or epoch seconds
                    if isinstance(date, (int, float)):
                        age_days = (now - float(date)) / 86400
                    else:
                        from dateutil.parser import parse
                        dt = parse(date)
                        age_days = (now - dt.timestamp()) / 86400
                    if age_days < 365:
                        boost *= recency_boost
                except Exception:
                    pass
            # Tag boost (if tags overlap with query keywords)
            tags = src.metadata.get('tags', [])
            if tags and hasattr(self, '_last_query_keywords'):
                if any(tag.lower() in self._last_query_keywords for tag in tags):
                    boost *= tag_boost
            # Source boost (if preferred source substring in document path)
            if preferred_sources and any(pref in src.document for pref in preferred_sources):
                boost *= source_boost
            src.similarity *= boost
        # Re-rank by new similarity
        return sorted(sources, key=lambda s: s.similarity, reverse=True)

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """
        Deduplicate sources by document path. Keep only the highest-similarity chunk per document.
        """
        seen = {}
        for src in sources:
            doc = src.document
            if doc not in seen or src.similarity > seen[doc].similarity:
                seen[doc] = src
        return list(seen.values())

    def process_query(self, query: str) -> QueryResponse:
        """
        Process a semantic search query and return a response object.
        Implements hybrid scoring and Gemini LLM response generation.
        Args:
            query: The user query string
        Returns:
            QueryResponse object with answer, sources, and metadata
        """
        start_time = time.time()
        try:
            if not query or not isinstance(query, str):
                self.logger.info(f"process_query called with invalid query: {query!r} (type: {type(query)})")
                raise QASystemError("Query must be a non-empty string.")
            self.logger.info(f"process_query called with query: {query!r}")
            # Generate embedding for the query
            embedding_result = self.embedding_generator.generate_embeddings([query], metadata={"task_type": "RETRIEVAL_QUERY"})
            vectors = embedding_result.get('vectors', [])
            if not vectors or not isinstance(vectors, list):
                self.logger.info(f"Embedding generation failed or returned no vectors for query: {query!r}")
                raise EmbeddingError("Failed to generate embedding for query.")
            query_vector = vectors[0]
            # Query the vector store
            top_k = int(self.config.get_nested('QUERY.TOP_K', default=40))
            results = self.vector_store.query(query_vector, top_k=top_k)
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] if 'distances' in results else []
            # Log concise list of sources after vector search
            source_paths = [meta.get('path', doc_id) for meta, doc_id in zip(metadatas, ids)]
            self.logger.debug(f"Initial sources after vector search for query '{query}': {source_paths}")
            # Detailed logging for vector store results (no document excerpts, no document content, no full results dump)
            self.logger.debug(f"Vector store returned {len(ids)} results.")
            for i, doc_id in enumerate(ids):
                self.logger.debug(f"  id: {doc_id}")
                self.logger.debug(f"  distance: {distances[i] if i < len(distances) else 'N/A'}")
                # Only log safe metadata fields (never log 'documents' or any content)
                safe_meta = {k: v for k, v in (metadatas[i] if i < len(metadatas) else {}).items() if k != 'chunk' and k != 'document' and k != 'text'}
                self.logger.debug(f"  metadata: {safe_meta}")
            # Build sources list
            sources = []
            for i, doc_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ''
                meta = metadatas[i] if i < len(metadatas) else {}
                sim = 1.0 - distances[i] if i < len(distances) else 0.0
                context = doc[:200]  # Truncate for context
                sources.append(Source(document=meta.get('path', doc_id), chunk=doc, similarity=sim, context=context, metadata=meta))
            # Log similarity before hybrid scoring
            self.logger.debug("Similarities before hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources before hybrid scoring: {[src.document for src in sources]}")
            # Hybrid scoring: extract keywords from query for tag boost
            import re
            all_words = re.findall(r"\w+", query.lower())
            self._last_query_keywords = set(w for w in all_words if w not in STOPWORDS)
            self.logger.info(f"Extracted keywords from query (stopwords removed): {sorted(self._last_query_keywords)}")
            if not self._last_query_keywords:
                self.logger.info(f"No keywords extracted from query after stopword removal: {query!r}. This may indicate an empty, non-alphanumeric, or all-stopword query.")
            sources = self._apply_hybrid_scoring(sources)
            # Log similarity after hybrid scoring
            self.logger.debug("Similarities after hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources after hybrid scoring: {[src.document for src in sources]}")
            # Deduplicate sources by document path (keep highest similarity chunk per doc)
            deduped_sources = self._deduplicate_sources(sources)
            # Minimum similarity threshold
            min_similarity = float(self.config.get_nested('QUERY.MIN_SIMILARITY', default=0.2))
            filtered_sources = [src for src in deduped_sources if src.similarity >= min_similarity]
            if not filtered_sources:
                self.logger.warning(f"No sources above similarity threshold {min_similarity}. Returning top result anyway.")
                filtered_sources = deduped_sources[:1] if deduped_sources else []
            # Log concise list of sources after deduplication/filtering
            self.logger.debug(f"Sources after deduplication/filtering: {[src.document for src in filtered_sources]}")
            # Check for relevant document presence: look for any extracted keyword in metadata fields or content
            relevant_keywords = list(self._last_query_keywords)
            found_relevant = False
            for src in filtered_sources:
                # Gather all searchable fields
                fields = []
                # tags (may be list or string)
                tags = src.metadata.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',') if t.strip()]
                fields.extend([t.lower() for t in tags])
                # path
                path = src.metadata.get('path', '')
                if path:
                    fields.append(path.lower())
                # filename_stem
                filename_stem = src.metadata.get('filename_stem', '')
                if filename_stem:
                    fields.append(filename_stem.lower())
                # url
                url = src.metadata.get('url', '')
                if url:
                    fields.append(url.lower())
                # document path and chunk text
                fields.append(src.document.lower())
                fields.append(src.chunk.lower())
                # Check if any keyword is in any field
                for kw in relevant_keywords:
                    if any(kw in f for f in fields):
                        found_relevant = True
                        break
                if found_relevant:
                    break
            if not found_relevant:
                self.logger.warning(
                    f"Relevant document for keywords {relevant_keywords} not found in any of the following fields: tags, path, filename_stem, url, document, chunk. Extracted keywords: {sorted(self._last_query_keywords)}."
                )
            # Assemble context for Gemini (at most one chunk per document, unless more needed to fill context window)
            context_window = int(self.config.get_nested('QUERY.CONTEXT_WINDOW', default=4096))
            max_tokens = int(self.config.get_nested('QUERY.MAX_TOKENS', default=512))
            temperature = float(self.config.get_nested('QUERY.TEMPERATURE', default=0.2))
            context_text = ""
            tokens_used = 0
            used_docs = set()
            context_chunks = []
            for src in filtered_sources:
                if src.document in used_docs:
                    continue
                chunk = src.chunk
                tokens_used += len(chunk.split())
                if tokens_used > context_window:
                    break
                context_text += f"\n---\n{chunk}"
                context_chunks.append(src.document)
                used_docs.add(src.document)
            self.logger.debug("Context documents for Gemini:")
            for doc in context_chunks:
                self.logger.debug(f"  - {doc}")
            prompt = (
                "You are an expert assistant. Use the following context to answer the user's question. "
                "If the answer is not directly in the context, use your best judgment to provide a helpful, accurate answer. "
                "Cite sources if you use information from a specific document.\n\n"
                f"Question: {query}\n\nContext:{context_text}"
            )
            gemini_response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            response_text = getattr(gemini_response, 'text', None) or str(gemini_response)
            confidence = filtered_sources[0].similarity if filtered_sources else 0.0
            processing_time = time.time() - start_time
            return QueryResponse(
                text=response_text,
                sources=filtered_sources,
                confidence=confidence,
                processing_time=processing_time,
                error=None,
                success=True
            )
        except TypeError as e:
            if "unexpected keyword argument 'generation_config'" in str(e):
                gemini_response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )
                response_text = getattr(gemini_response, 'text', None) or str(gemini_response)
                confidence = filtered_sources[0].similarity if filtered_sources else 0.0
                processing_time = time.time() - start_time
                return QueryResponse(
                    text=response_text,
                    sources=filtered_sources,
                    confidence=confidence,
                    processing_time=processing_time,
                    error=None,
                    success=True
                )
            else:
                self.logger.error(f"Query processing failed: {e}")
                return QueryResponse(
                    text="",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error=str(e),
                    success=False
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
