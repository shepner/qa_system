---
title: Prompt Templates Configuration
description: Documentation for the QA System's prompt templates configuration
tags: [configuration, prompts, templates]
version: 1.0.0
last_updated: 2024-03-27
---

# Prompt Templates Configuration

## Overview

The `prompts.yaml` configuration file defines the templates used by the QA system for generating structured responses to user queries. These templates ensure consistent, high-quality answers by providing clear instructions to the language model about the expected format and analysis process.

## File Location

```
config/prompts.yaml
```

## Template Structure

### Answer Generation Template

The answer generation template consists of several key components:

#### System Message
Defines the AI assistant's role and general behavioral guidelines. Example:
```yaml
system_message: |
  You are a knowledgeable AI assistant tasked with providing accurate, 
  well-structured answers based on the provided context.
```

#### Task Description

Outlines the specific task and requirements for answer generation. Example:
```yaml
task_description: |
  Analyze the provided context and generate a comprehensive answer to the user's question.
  Ensure all claims are supported by the context and cite relevant sources.
```

#### Analysis Steps
Defines the structured approach for processing questions:
```yaml
analysis_steps:
  - Understand the core question and identify key concepts
  - Review provided context for relevant information
  - Synthesize information into a coherent answer
  - Validate answer against source material
  - Format response according to specified structure
```

#### Full Template
The complete template combines these elements with placeholders for dynamic content:
```yaml
template: |
  {system_message}
  
  Question: {question}
  
  Context:
  {context}
  
  Task: {task_description}
  
  Follow these steps:
  {analysis_steps}
```

## Response Format

Answers must be formatted as JSON with the following structure:
```json
{
  "answer": "The detailed response to the question",
  "sources": [
    {
      "document_id": "unique_id",
      "relevance_score": 0.95,
      "excerpt": "Relevant text from source"
    }
  ],
  "confidence_score": 0.85
}
```

## Usage

To load and use templates in code:

```python
from prompt_loader import PromptLoader

# Initialize the prompt loader
prompt_loader = PromptLoader(config_dir="config")

# Load the answer generation template
template = prompt_loader.load_template("answer_generation")

# Format the template with specific content
formatted_prompt = template.format(
    question=user_question,
    context=relevant_documents,
)
```

## Customization

Templates can be customized by:
1. Modifying the system message to adjust the AI's role
2. Updating the task description for specific use cases
3. Adding or modifying analysis steps
4. Extending the response format with additional fields

## Best Practices

1. Keep system messages concise and focused
2. Use clear, unambiguous language in task descriptions
3. Break down complex analysis into discrete steps
4. Maintain consistent formatting across templates
5. Version control template changes
6. Test template changes with various question types
7. Document any custom modifications

## Related Files

- `qa_system/prompt_loader.py`: Implementation of template loading
- `qa_system/qa_engine.py`: Usage of templates in answer generation
- `config/qa_config.yaml`: General QA system configuration

## Version History

- 1.0.0 (2024-03-27): Initial documentation
  - Defined basic template structure
  - Added usage examples and best practices
  - Established response format standards 