"""
prompts.py
Utilities for constructing LLM prompts from question and context.
"""

def build_llm_prompt(question: str, context_text: str, system_prompt: str = None) -> str:
    """
    Build the prompt for the LLM using the question and context.
    Args:
        question: The user's question
        context_text: The context string built from sources
        system_prompt: Optional system prompt/instructions
    Returns:
        str: The prompt to send to the LLM
    """
    base = (
        "You are an expert assistant. Use the following context to answer the user's question. "
        "If the answer is not directly in the context, use your best judgment to provide a helpful, accurate answer. "
        "Cite sources if you use information from a specific document.\n\n"
    )
    if system_prompt:
        base = system_prompt + "\n\n" + base
    return (
        f"{base}Question: {question}\n\nContext:{context_text}"
    ) 