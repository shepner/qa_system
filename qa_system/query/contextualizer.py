"""
@file: contextualizer.py
Module to generate contextual background data from user interactions.

Takes user interaction text as input and generates contextual summaries using an LLM.
"""

import logging
from .gemini_llm import GeminiLLM
from qa_system.config import get_config

logger = logging.getLogger(__name__)

def generate_context(text: str) -> str:
    """
    Use the LLM to generate a contextual summary from user interaction text.
    """
    logger.debug(f"Processing text: {text[:100]}...")
    prompt = (
        "Given the following logfile of interactions with the user, generate concise contextual background information without including the original text.\n"
        "Mainly focus on the 'Question' sections with less emphasis on the 'Answer' sections.  The timestamp is likely of lesser importance.\n"
        "I want to capture pertinent information about the user's intent as well as information about the organization and its operations that would be helpful to answer other similar questions in the future.\n\n"
        f"User interaction:\n{text}"
    )
    logger.info(f"Prompt sent to LLM:\n{prompt}")
    config = get_config()
    llm = GeminiLLM(config)
    context = llm.generate_response(user_prompt=prompt)
    logger.info(f"Output from LLM:\n{context}")
    return context 