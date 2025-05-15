"""
keywords.py
Utilities for extracting keywords from queries, with stopword removal.
"""
import re
from typing import Set, List


def derive_keywords(processor, query: str, mode: str = 'keywords', logger: None = None) -> set:
    """
    Derive keywords or tags from the query using different modes.
    Modes:
        - 'keywords': Extract keywords from the query (ignoring tags).
        - 'tags': Use Gemini LLM to extract relevant tags for the query, restricted to those in the vector store.
    Args:
        processor: QueryProcessor instance (must have .vector_store and .llm)
        query: The user query string
        mode: Extraction mode ('keywords' or 'tags')
        logger: Optional logger
    Returns:
        Set[str]: Set of keywords or tags relevant to the query
    """
    if mode == 'keywords':
        # Use Gemini LLM to extract keywords from the query
        system_prompt = (
            "You are a keyword extraction assistant. "
            "Given a user query, extract the most relevant keywords (not tags or phrases, just keywords). "
            "Output only a comma-separated list of keywords, or an empty string if none are relevant. "
            "Example output: keyword1,keyword2,keyword3"
        )
        user_prompt = f"Query: {query}"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            response = processor.llm.generate_response(
                user_prompt=full_prompt,
                temperature=0.0,
                max_output_tokens=32
            )
            if logger:
                logger.info(f"Gemini LLM keyword extraction response: {response}")
            if not response:
                return set()
            response = response.strip().strip('[]')
            keywords = set()
            for kw in response.split(','):
                t = kw.strip().strip('"\'')
                if t:
                    keywords.add(t)
            return keywords
        except Exception as e:
            if logger:
                logger.error(f"Failed to extract keywords using Gemini LLM: {e}")
            return set()
    elif mode == 'tags':
        all_tags = ", ".join(processor.vector_store.get_all_tags())
        if not all_tags:
            if logger:
                logger.warning("No tags found in vector store.")
            return set()
        # Build system prompt for tag extraction
        system_prompt = (
            "You are a tag extraction assistant. "
            "Given a user query and a list of allowed tags, return only the tags from the allowed list that are relevant to the query. "
            f"Allowed tags: {all_tags}. "
            "Do not invent new tags. Do not explain your reasoning. "
            "Output only a comma-separated list of relevant tags from the allowed list, or an empty string if none are relevant. "
            "Example output: tag1,tag2,tag3"
        )
        user_prompt = f"Query: {query}\nAllowed tags: {all_tags}"
        if logger:
            logger.debug(f"system_prompt type: {type(system_prompt)}, value: {system_prompt}")
            logger.debug(f"user_prompt type: {type(user_prompt)}, value: {user_prompt}")
        system_prompt = str(system_prompt)
        user_prompt = str(user_prompt)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if logger:
            logger.debug(f"full_prompt type: {type(full_prompt)}, value: {full_prompt}")
        try:
            response = processor.llm.generate_response(
                user_prompt=full_prompt,
                temperature=0.0,
                max_output_tokens=64
            )
            if logger:
                logger.info(f"Gemini LLM tag extraction response: {response}")
            if not response:
                return set()
            response = response.strip().strip('[]')
            tags = set()
            allowed_tags = set(t.strip() for t in all_tags.split(","))
            for tag in response.split(','):
                t = tag.strip().strip('"\'')
                if t and t in allowed_tags:
                    tags.add(t)
            return tags
        except Exception as e:
            if logger:
                logger.error(f"Failed to extract tags using Gemini LLM: {e}")
            return set()
    else:
        if logger:
            logger.error(f"Unknown derive_keywords mode: {mode}")
        return set() 