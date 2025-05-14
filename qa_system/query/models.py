from typing import Optional

class Source:
    def __init__(self, document: str, chunk: str, similarity: float, context: str = '', metadata: Optional[dict] = None, original_similarity: float = None, boost: float = None):
        self.document = document
        self.chunk = chunk
        self.similarity = similarity
        self.context = context
        self.metadata = metadata or {}
        self.original_similarity = original_similarity
        self.boost = boost

class QueryResponse:
    def __init__(self, text: str, sources: list, confidence: float = 1.0, processing_time: float = 0.0, error: Optional[str] = None, success: bool = True):
        self.text = text
        self.sources = sources
        self.confidence = confidence
        self.processing_time = processing_time
        self.error = error
        self.success = success 