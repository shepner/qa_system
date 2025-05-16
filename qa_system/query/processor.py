import logging
import os
import time
import uuid
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import QASystemError, EmbeddingError
from .models import QueryResponse
from .scoring import apply_scoring, deduplicate_sources
from .keywords import derive_keywords
from .source_utils import build_sources_from_vector_results
from .source_filter import filter_sources
from .context_builder import build_context_window
from .prompts import build_llm_prompt
from .gemini_llm import GeminiLLM

"""
@file: processor.py
Semantic Query Processor for QA System

This module implements the QueryProcessor class, which handles semantic search queries and generates contextual responses using document embeddings, a vector store, and a large language model (LLM). It supports hybrid scoring (semantic + metadata boosts), tag/keyword matching, and config-driven parameters for flexible, high-quality question answering over a document corpus.

Main Components:
- QueryProcessor: Main class for processing queries and generating responses.
- Hybrid search: Combines tag/keyword and semantic search for candidate document selection.
- Scoring and filtering: Applies hybrid scoring, deduplication, and similarity filtering.
- Context window: Builds context for LLM prompt from filtered sources.
- LLM integration: Calls Gemini LLM to generate a final answer.

Usage:
    processor = QueryProcessor(config)
    response = processor.process_query("What is data governance?")
    print(response.text)

Dependencies:
- EmbeddingGenerator, ChromaVectorStore, GeminiLLM, and related QA system modules.
- Configuration must provide paths and parameters for vector store and LLM.

Raises:
- QASystemError, EmbeddingError for various error conditions.

"""

class QueryProcessor:
    """
    Handles semantic search queries and generates contextual responses using embeddings, vector store, and Gemini LLM.
    Implements hybrid scoring (semantic + metadata boosts) and config-driven parameters.

    Args:
        config (Any): Configuration object with required settings for embeddings, vector store, and LLM.
        embedding_generator (Optional[EmbeddingGenerator]): Custom embedding generator (for testing or extension).
        vector_store (Optional[ChromaVectorStore]): Custom vector store (for testing or extension).
        llm (Optional[GeminiLLM]): Custom LLM interface (for testing or extension).

    Attributes:
        logger (logging.Logger): Logger for this processor.
        config (Any): Configuration object.
        embedding_generator (EmbeddingGenerator): Embedding generator instance.
        vector_store (ChromaVectorStore): Vector store instance.
        llm (GeminiLLM): LLM interface instance.
    """
    def __init__(self, config, embedding_generator=None, vector_store=None, llm=None):
        """
        Initialize the QueryProcessor.

        Args:
            config (Any): Configuration object with required settings.
            embedding_generator (Optional[EmbeddingGenerator]): Custom embedding generator.
            vector_store (Optional[ChromaVectorStore]): Custom vector store.
            llm (Optional[GeminiLLM]): Custom LLM interface.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called QueryProcessor.__init__(config={config})")
        self.config = config
        self.embedding_generator = embedding_generator if embedding_generator is not None else EmbeddingGenerator(config)
        self.vector_store = vector_store if vector_store is not None else ChromaVectorStore(config)
        self.llm = llm if llm is not None else GeminiLLM(config)

    def process_query(self, query: str, system_prompt: str = None) -> QueryResponse:
        """
        Process a user query and generate a contextual response using hybrid search and LLM.

        Args:
            query (str): The user query string to process.
            system_prompt (str, optional): Optional system prompt to guide the LLM.

        Returns:
            QueryResponse: Structured response containing the answer, sources, confidence, timing, and error info.

        Raises:
            QASystemError: If the query is invalid or empty.
            EmbeddingError: If embedding generation fails.
            Exception: For other unexpected errors (returned in QueryResponse.error).

        Side Effects:
            - Logs query processing steps and errors.
            - Tracks timing and query/session IDs for correlation.
        """
        # Start timing for performance metrics
        start_time = time.time()
        # Generate a unique query/session id for correlation
        query_id = str(uuid.uuid4())
        try:
            # --- Input validation ---
            if not query or not isinstance(query, str):
                self.logger.info(f"process_query called with invalid query: {query!r} (type: {type(query)}) [query_id={query_id}]")
                raise QASystemError("Query must be a non-empty string.")
            self.logger.info(f"process_query called with query: {query!r} [query_id={query_id}]")

            # --- Derive keywords and tag-matching keywords from query ---
            self._last_query_keywords = derive_keywords(
                self,
                query=query,
                mode='keywords',
                logger=self.logger
            )
            self.logger.info(f"Keywords derived from query: {sorted(self._last_query_keywords)} [query_id={query_id}]")
            if not self._last_query_keywords:
                self.logger.info(f"No keywords derived from query: {query!r}. [query_id={query_id}]")
            self._last_tag_matching_keywords = derive_keywords(
                self,
                query=query,
                mode='tags',
                logger=self.logger
            )
            self.logger.info(f"Tag-matching keywords derived from query: {sorted(self._last_tag_matching_keywords)} [query_id={query_id}]")

            # --- Hybrid-first: Gather candidate docs by tag/keyword ---
            candidate_metas = []
            seen_doc_ids = set()
            # Build id->meta mapping for fast lookup
            all_metas = self.vector_store.list_metadata()
            id_to_meta = {meta.get('id') or meta.get('path'): meta for meta in all_metas}
            # Tag-matching
            for tag in self._last_tag_matching_keywords:
                self.logger.info(f"[query_id={query_id}] Tag-matching: searching for tag '{tag}'")
                tag_ids = self.vector_store.search_metadata(tag, metadata_keys=["tags"])
                for doc_id in tag_ids:
                    if doc_id and doc_id not in seen_doc_ids:
                        meta = id_to_meta.get(doc_id)
                        if meta:
                            candidate_metas.append(meta)
                            seen_doc_ids.add(doc_id)
            # Keyword-matching
            for kw in self._last_query_keywords:
                self.logger.info(f"[query_id={query_id}] Keyword-matching: searching for keyword '{kw}'")
                kw_ids = self.vector_store.search_metadata(kw)
                for doc_id in kw_ids:
                    if doc_id and doc_id not in seen_doc_ids:
                        meta = id_to_meta.get(doc_id)
                        if meta:
                            candidate_metas.append(meta)
                            seen_doc_ids.add(doc_id)
            self.logger.info(f"Hybrid-first: Found {len(candidate_metas)} unique candidate docs by tag/keyword matching. [query_id={query_id}]")

            # --- Optionally, add top_k semantic search for recall boost ---
            embedding_result = self.embedding_generator.generate_embeddings([query], metadata={"task_type": "RETRIEVAL_QUERY"})
            vectors = embedding_result.get('vectors', [])
            if not vectors or not isinstance(vectors, list):
                self.logger.info(f"Query embedding generation failed or returned no vectors for query: {query!r}")
                raise EmbeddingError("Failed to generate query embedding.")
            query_vector = vectors[0]
            top_k = int(self.config.get_nested('QUERY.TOP_K', default=40))
            results = self.vector_store.query(query_vector, top_k=top_k)
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] if 'distances' in results else []
            self.logger.info(f"Hybrid-first: Using {len(ids)} docs from top_k semantic search for scoring and context.")

            # --- Build Source objects from vector search results only ---
            docs_root = self.config.get_nested('FILE_SCANNER.DOCUMENT_PATH', './docs')
            docs_root = os.path.abspath(docs_root)
            sources = build_sources_from_vector_results(
                ids=ids,
                docs=docs,
                metadatas=metadatas,
                distances=distances,
                docs_root=docs_root,
                context_length=200
            )

            # --- Apply hybrid scoring and deduplicate sources ---
            sources = apply_scoring(self, sources)
            sources = deduplicate_sources(sources, logger=self.logger)

            # --- Filter sources based on similarity and tag-matching ---
            filtered_sources, tag_matched_sources = filter_sources(
                sources,
                tag_matching_keywords=self._last_tag_matching_keywords,
                min_similarity=float(self.config.get_nested('QUERY.MIN_SIMILARITY', default=0.2)),
                tag_min_similarity=float(self.config.get_nested('QUERY.TAG_MATCH_MIN_SIMILARITY', default=0.1)),
                logger=self.logger
            )
            self.logger.info(f"Sources included due to tag-matching: {[src.document for src in tag_matched_sources]}")

            # --- Build context window for LLM prompt ---
            context_window = int(self.config.get_nested('QUERY.CONTEXT_WINDOW', default=4096))
            context_text, tokens_used, context_chunks = build_context_window(
                filtered_sources,
                context_window=context_window
            )
            self.logger.debug("Context documents for Gemini:")
            for doc in context_chunks:
                self.logger.debug(f"  - {doc}")

            # --- Build LLM prompt and call LLM ---
            prompt = build_llm_prompt(query, context_text, system_prompt=system_prompt)
            try:
                response_text = self.llm.generate_response(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=float(self.config.get_nested('QUERY.TEMPERATURE', default=0.2)),
                    max_output_tokens=int(self.config.get_nested('QUERY.MAX_TOKENS', default=512))
                )
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}")
                response_text = ""

            # --- Prepare and return QueryResponse ---
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

        # --- Error handling ---
        except TypeError as e:
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