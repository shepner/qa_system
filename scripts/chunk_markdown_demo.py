import sys
import yaml
import json
from qa_system.document_processors.markdown_processor import MarkdownDocumentProcessor

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
class C:
    def get_nested(self, k, d=None):
        return config["DOCUMENT_PROCESSING"].get(k.split(".")[-1], d)

processor = MarkdownDocumentProcessor(C())

file_path = './docs/Library/Artifacts/Guidance/Strategies/User Endpoint Security Strategy.md'
result = processor.process(file_path)

for i, c in enumerate(result["chunks"]):
    header = c["metadata"].get("section_header", "")
    text = c["text"][:120].replace("\n", " ") + ("..." if len(c["text"]) > 120 else "")
    print(f"Chunk {i+1:02d} | Section: {header!r}\n{text}\n{'-'*80}") 