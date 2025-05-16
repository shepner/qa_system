"""
keywords.py
Utilities for extracting keywords or tags from queries using an LLM.
"""


def derive_keywords(processor, query: str, mode: str = 'keywords', logger: None = None) -> set:
    """
    Extract keywords or tags from a user query using an LLM.

    Modes:
        - 'keywords': Uses the LLM to extract relevant keywords from the query for search purposes.
        - 'tags': Uses the LLM to extract relevant tags from a predefined set (from the vector store), restricting output to only allowed tags.

    Args:
        processor: QueryProcessor instance (must have .vector_store and .llm attributes).
        query: The user query string.
        mode: Extraction mode ('keywords' or 'tags'). Defaults to 'keywords'.
        logger: Optional logger for debug and error messages.

    Returns:
        Set[str]: A set of extracted keywords or tags relevant to the query.
                  Returns an empty set if extraction fails or the LLM response is invalid.

    Notes:
        - Both modes use the LLM for extraction, but with different prompts and constraints.
        - The function expects the LLM to return a comma-separated string of keywords or tags.
        - Handles and logs errors gracefully, returning an empty set on failure.
    """
    if mode == 'keywords':
        # Use Gemini LLM to extract keywords from the query
        system_prompt = (
            "You are a keyword extraction assistant. "
            "Given a user query, generate a list of the most relevant keywords (not tags or phrases, just keywords) that would be useful for searching a knowledge base. "
            "Output only a comma-separated list of keywords, or an empty string if none are relevant. "
            "Example output: keyword1,keyword2,keyword3"
        )
        user_prompt = f"Query: {query}"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if logger:
            logger.debug(f"[KEYWORDS] system_prompt type: {type(system_prompt)}, value: {system_prompt}")
            logger.debug(f"[KEYWORDS] user_prompt type: {type(user_prompt)}, value: {user_prompt}")
            logger.debug(f"[KEYWORDS] full_prompt type: {type(full_prompt)}, value: {full_prompt}")
        try:
            response = processor.llm.generate_response(
                user_prompt=full_prompt,
                temperature=0.0,
                max_output_tokens=32
            )
            if logger:
                logger.info(f"[KEYWORDS] Gemini LLM keyword extraction response: {response}")
                logger.debug(f"[KEYWORDS] Gemini LLM keyword extraction response type: {type(response)}")
            if not response or not isinstance(response, str):
                if logger:
                    logger.warning(f"[KEYWORDS] LLM keyword extraction returned non-string or empty response: {type(response)}: {response}")
                return set()
            if any(x in response for x in ["=", "None", "candidates=", "usage_metadata=", "prompt_token_count", "response_id", "model_version"]):
                if logger:
                    logger.warning(f"[KEYWORDS] LLM keyword extraction returned unexpected format: {response}")
                return set()
            response = response.strip().strip('[]')
            if logger:
                logger.debug(f"[KEYWORDS] Stripped response: {response}")
            keywords = set()
            for kw in response.split(','):
                t = kw.strip().strip('"\'')
                if t:
                    keywords.add(t)
            if logger:
                logger.debug(f"[KEYWORDS] Parsed keywords: {keywords}")
            return keywords
        except Exception as e:
            if logger:
                logger.error(f"[KEYWORDS] Failed to extract keywords using Gemini LLM: {e}")
            return set()
    elif mode == 'tags':
        all_tags = ", ".join(processor.vector_store.get_all_tags())
        if not all_tags:
            if logger:
                logger.warning("[TAGS] No tags found in vector store.")
            return set()
        # Build system prompt for tag extraction
        system_prompt = (
            "You are a tag extraction assistant. "
            "Given a user query and a list of allowed tags, return only the best-fit tags from the allowed list that are relevant to the query. "
            f"Allowed tags: {all_tags}. "
            "Do not invent new tags. Do not explain your reasoning. "
            "Output only a comma-separated list of relevant tags from the allowed list, or an empty string if none are relevant. "
            "Example output: tag1,tag2,tag3"
        )
        user_prompt = f"Query: {query}\nAllowed tags: {all_tags}"
        if logger:
            logger.debug(f"[TAGS] system_prompt type: {type(system_prompt)}, value: {system_prompt}")
            logger.debug(f"[TAGS] user_prompt type: {type(user_prompt)}, value: {user_prompt}")
        system_prompt = str(system_prompt)
        user_prompt = str(user_prompt)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if logger:
            logger.debug(f"[TAGS] full_prompt type: {type(full_prompt)}, value: {full_prompt}")
        try:
            response = processor.llm.generate_response(
                user_prompt=full_prompt,
                temperature=0.0,
                max_output_tokens=64
            )
            if logger:
                logger.info(f"[TAGS] Gemini LLM tag extraction response: {response}")
                logger.debug(f"[TAGS] Gemini LLM tag extraction response type: {type(response)}")
            if not response or not isinstance(response, str):
                if logger:
                    logger.warning(f"[TAGS] LLM tag extraction returned non-string or empty response: {type(response)}: {response}")
                return set()
            if any(x in response for x in ["=", "None", "candidates=", "usage_metadata=", "prompt_token_count", "response_id", "model_version"]):
                if logger:
                    logger.warning(f"[TAGS] LLM tag extraction returned unexpected format: {response}")
                return set()
            response = response.strip().strip('[]')
            if logger:
                logger.debug(f"[TAGS] Stripped response: {response}")
            tags = set()
            allowed_tags = set(t.strip() for t in all_tags.split(","))
            for tag in response.split(','):
                t = tag.strip().strip('"\'')
                if t and t in allowed_tags:
                    tags.add(t)
            if logger:
                logger.debug(f"[TAGS] Parsed tags: {tags}")
            return tags
        except Exception as e:
            if logger:
                logger.error(f"[TAGS] Failed to extract tags using Gemini LLM: {e}")
            return set()
    else:
        if logger:
            logger.error(f"Unknown derive_keywords mode: {mode}")
        return set() 