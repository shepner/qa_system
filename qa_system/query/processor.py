import logging
import os
import time
import fnmatch
import difflib
import re
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from google import genai
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import QASystemError, QueryError, EmbeddingError
from .models import Source, QueryResponse
from .constants import STOPWORDS
from .scoring import apply_scoring, deduplicate_sources, extract_tag_matching_keywords
from .source_utils import build_sources_from_vector_results
from .keywords import extract_keywords
from .source_filter import filter_sources
from .context_builder import build_context_window
from .prompts import build_llm_prompt

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

    def process_query(self, query: str) -> QueryResponse:
        # ... existing code ...
        start_time = time.time()
        try:
            if not query or not isinstance(query, str):
                self.logger.info(f"process_query called with invalid query: {query!r} (type: {type(query)})")
                raise QASystemError("Query must be a non-empty string.")
            self.logger.info(f"process_query called with query: {query!r}")
            embedding_result = self.embedding_generator.generate_embeddings([query], metadata={"task_type": "RETRIEVAL_QUERY"})
            vectors = embedding_result.get('vectors', [])
            if not vectors or not isinstance(vectors, list):
                self.logger.info(f"Embedding generation failed or returned no vectors for query: {query!r}")
                raise EmbeddingError("Failed to generate embedding for query.")
            query_vector = vectors[0]
            top_k = int(self.config.get_nested('QUERY.TOP_K', default=40))
            results = self.vector_store.query(query_vector, top_k=top_k)
            ids = results.get('ids', [[]])[0]
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0] if 'distances' in results else []
            source_paths = [meta.get('path', doc_id) for meta, doc_id in zip(metadatas, ids)]
            self.logger.debug(f"Initial sources after vector search for query '{query}': {source_paths}")
            self.logger.debug(f"Vector store returned {len(ids)} results.")
            for i, doc_id in enumerate(ids):
                self.logger.debug(f"  id: {doc_id}")
                self.logger.debug(f"  distance: {distances[i] if i < len(distances) else 'N/A'}")
                safe_meta = {k: v for k, v in (metadatas[i] if i < len(metadatas) else {}).items() if k != 'chunk' and k != 'document' and k != 'text'}
                self.logger.debug(f"  metadata: {safe_meta}")
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
            self.logger.debug("Similarities before hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources before hybrid scoring: {[src.document for src in sources]}")
            self._last_query_keywords = extract_keywords(query, STOPWORDS)
            self.logger.info(f"Extracted keywords from query (stopwords removed): {sorted(self._last_query_keywords)}")
            if not self._last_query_keywords:
                self.logger.info(f"No keywords extracted from query after stopword removal: {query!r}. This may indicate an empty, non-alphanumeric, or all-stopword query.")
            all_tags = self.vector_store.get_all_tags()
            self._last_tag_matching_keywords = extract_tag_matching_keywords(
                query,
                all_tags=all_tags,
                stopwords=STOPWORDS,
                logger=self.logger
            )
            self.logger.info(f"Tag-matching keywords from query: {sorted(self._last_tag_matching_keywords)}")
            sources = apply_scoring(self, sources)
            self.logger.debug("Similarities after hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources after hybrid scoring: {[src.document for src in sources]}")
            deduped_sources = deduplicate_sources(sources, logger=self.logger)
            filtered_sources, tag_matched_sources = filter_sources(
                deduped_sources,
                tag_matching_keywords=self._last_tag_matching_keywords,
                min_similarity=float(self.config.get_nested('QUERY.MIN_SIMILARITY', default=0.2)),
                tag_min_similarity=float(self.config.get_nested('QUERY.TAG_MATCH_MIN_SIMILARITY', default=0.1)),
                logger=self.logger
            )
            self.logger.info(f"Sources included due to tag-matching: {[src.document for src in tag_matched_sources]}")
            relevant_keywords = list(self._last_query_keywords)
            found_relevant = False
            for src in filtered_sources:
                fields = []
                tags = src.metadata.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',') if t.strip()]
                fields.extend([t.lower() for t in tags])
                path = src.metadata.get('path', '')
                if path:
                    fields.append(path.lower())
                filename_stem = src.metadata.get('filename_stem', '')
                if filename_stem:
                    fields.append(filename_stem.lower())
                url = src.metadata.get('url', '')
                if url:
                    fields.append(url.lower())
                fields.append(src.document.lower())
                fields.append(src.chunk.lower())
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
            context_window = int(self.config.get_nested('QUERY.CONTEXT_WINDOW', default=4096))
            context_text, tokens_used, context_chunks = build_context_window(
                filtered_sources,
                context_window=context_window
            )
            self.logger.debug("Context documents for Gemini:")
            for doc in context_chunks:
                self.logger.debug(f"  - {doc}")
            prompt = build_llm_prompt(query, context_text)
            gemini_response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                generation_config={
                    "temperature": float(self.config.get_nested('QUERY.TEMPERATURE', default=0.2)),
                    "max_output_tokens": int(self.config.get_nested('QUERY.MAX_TOKENS', default=512))
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