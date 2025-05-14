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
        # ... existing code ...
        recency_boost = float(self.config.get_nested('QUERY.RECENCY_BOOST', default=1.0))
        tag_boost = float(self.config.get_nested('QUERY.TAG_BOOST', default=1.5))
        source_boost = float(self.config.get_nested('QUERY.SOURCE_BOOST', default=1.0))
        now = time.time()
        preferred_sources = self.config.get_nested('QUERY.PREFERRED_SOURCES', default=[])
        for src in sources:
            original_similarity = src.similarity
            src.similarity = max(0.0, min(1.0, src.similarity))
            boost = 1.0
            date = src.metadata.get('date')
            if date:
                try:
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
            tags = src.metadata.get('tags', [])
            if tags and hasattr(self, '_last_query_keywords'):
                if any(tag.lower() in self._last_query_keywords for tag in tags):
                    boost *= tag_boost
            matched = False
            for pref in preferred_sources:
                if fnmatch.fnmatch(src.document, pref):
                    matched = True
                    break
            self.logger.debug(f"Hybrid scoring: src.document={src.document}, preferred_sources={preferred_sources}, matched={matched}")
            if matched:
                self.logger.debug(f"SOURCE BOOST APPLIED: {src.document} matched {preferred_sources}, boost={source_boost}")
                boost *= source_boost
            final_similarity = src.similarity * boost
            self.logger.debug(f"Scoring: {src.document} | original={original_similarity:.4f} | clamped={src.similarity:.4f} | boost={boost:.2f} | final={final_similarity:.4f}")
            src.original_similarity = original_similarity
            src.boost = boost
            src.similarity = final_similarity
        return sorted(sources, key=lambda s: s.similarity, reverse=True)

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        seen = {}
        for src in sources:
            doc = src.document
            if doc not in seen or src.similarity > seen[doc].similarity:
                seen[doc] = src
        return list(seen.values())

    def _get_all_tags(self):
        if hasattr(self, '_all_tags_cache'):
            return self._all_tags_cache
        all_tags = set()
        for meta in self.vector_store.list_documents():
            tags = meta.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',') if t.strip()]
            all_tags.update(t.lower() for t in tags)
        self._all_tags_cache = all_tags
        return all_tags

    def _extract_tag_matching_keywords(self, query: str, fuzzy_cutoff: float = 0.75) -> set:
        all_tags = self._get_all_tags()
        all_words = re.findall(r"\w+", query.lower())
        concept_keywords = [w for w in all_words if w not in STOPWORDS]
        concept_to_tags = {}
        matched_tags = set()
        for concept in concept_keywords:
            exact_matches = [tag for tag in all_tags if tag == concept]
            fuzzy_matches = difflib.get_close_matches(concept, all_tags, n=5, cutoff=fuzzy_cutoff)
            all_matches = set(exact_matches + fuzzy_matches)
            if all_matches:
                concept_to_tags[concept] = sorted(all_matches)
                matched_tags.update(all_matches)
        self.logger.info(f"Concept keywords from query: {concept_keywords}")
        for concept, tags in concept_to_tags.items():
            self.logger.info(f"Tag matches for '{concept}': {tags}")
        self.logger.info(f"Final tag-matching keywords: {sorted(matched_tags)}")
        return matched_tags

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
            sources = []
            docs_root = self.config.get_nested('FILE_SCANNER.DOCUMENT_PATH', './docs')
            docs_root = os.path.abspath(docs_root)
            for i, doc_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ''
                meta = metadatas[i] if i < len(metadatas) else {}
                sim = 1.0 - distances[i] if i < len(distances) else 0.0
                context = doc[:200]
                abs_path = os.path.abspath(meta.get('path', doc_id))
                if abs_path.startswith(docs_root):
                    rel_path = os.path.relpath(abs_path, docs_root)
                else:
                    rel_path = abs_path
                clamped_sim = max(0.0, min(1.0, sim))
                sources.append(Source(document=rel_path, chunk=doc, similarity=clamped_sim, context=context, metadata=meta, original_similarity=sim))
            self.logger.debug("Similarities before hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources before hybrid scoring: {[src.document for src in sources]}")
            all_words = re.findall(r"\w+", query.lower())
            self._last_query_keywords = set(w for w in all_words if w not in STOPWORDS)
            self.logger.info(f"Extracted keywords from query (stopwords removed): {sorted(self._last_query_keywords)}")
            if not self._last_query_keywords:
                self.logger.info(f"No keywords extracted from query after stopword removal: {query!r}. This may indicate an empty, non-alphanumeric, or all-stopword query.")
            self._last_tag_matching_keywords = self._extract_tag_matching_keywords(query)
            self.logger.info(f"Tag-matching keywords from query: {sorted(self._last_tag_matching_keywords)}")
            sources = self._apply_hybrid_scoring(sources)
            self.logger.debug("Similarities after hybrid scoring:")
            for src in sources:
                self.logger.debug(f"  {src.document}: {src.similarity:.3f}")
            self.logger.debug(f"Sources after hybrid scoring: {[src.document for src in sources]}")
            deduped_sources = self._deduplicate_sources(sources)
            min_similarity = float(self.config.get_nested('QUERY.MIN_SIMILARITY', default=0.2))
            tag_min_similarity = float(self.config.get_nested('QUERY.TAG_MATCH_MIN_SIMILARITY', default=0.1))
            filtered_sources = []
            tag_matched_sources = []
            for src in deduped_sources:
                tags = src.metadata.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',') if t.strip()]
                tag_match = any(t.lower() in self._last_tag_matching_keywords for t in tags)
                if src.similarity >= min_similarity:
                    filtered_sources.append(src)
                elif tag_match and src.similarity >= tag_min_similarity:
                    tag_matched_sources.append(src)
            if not filtered_sources:
                self.logger.warning(f"No sources above similarity threshold {min_similarity}. Returning top result anyway.")
                filtered_sources = deduped_sources[:1] if deduped_sources else []
            for src in tag_matched_sources:
                if src not in filtered_sources:
                    filtered_sources.append(src)
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