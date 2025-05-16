"""
@file: prompts.py
Utilities for constructing prompts for Large Language Models (LLMs) using a user question and supporting context.

This module provides helper functions to build well-structured prompts for LLMs, ensuring that the model receives clear instructions, relevant context, and optional system-level guidance. Prompts are formatted to encourage accurate, helpful, and well-cited responses from the LLM.
"""

from typing import Optional

def build_llm_prompt(
    question: str,
    context_text: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Construct a prompt for a Large Language Model (LLM) using the user's question and supporting context.

    Args:
        question (str): The user's question to be answered by the LLM.
        context_text (str): Contextual information or source material to inform the answer.
        system_prompt (Optional[str], optional): Additional system-level instructions or persona for the LLM. Defaults to None.

    Returns:
        str: A formatted prompt string to send to the LLM.

    Example:
        >>> build_llm_prompt(
        ...     question="What is the capital of France?",
        ...     context_text="France is a country in Europe. Its capital is Paris.",
        ...     system_prompt="You are a helpful assistant."
        ... )
        'You are a helpful assistant.\n\nYou are an expert assistant. Use the following context to answer the user's question. If the answer is not directly in the context, use your best judgment to provide a helpful, accurate answer. Cite sources if you use information from a specific document.\n\nQuestion: What is the capital of France?\n\nContext:France is a country in Europe. Its capital is Paris.'
    """
    default_instructions = (
        "You are an expert assistant. Use the following context to answer the user's question. "
        "If the answer is not directly in the context, use your best judgment to provide a helpful, accurate answer. "
        "Cite sources if you use information from a specific document.\n\n"
    )
    prompt_instructions = (
        f"{system_prompt}\n\n{default_instructions}" if system_prompt else default_instructions
    )
    return (
        f"{prompt_instructions}Question: {question}\n\nContext:{context_text}"
    ) 