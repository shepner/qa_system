from .base_processor import BaseDocumentProcessor

class TextDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for plain text (.txt) files. Extracts metadata, chunks text, and returns results.
    """
    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing text file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            # Merge/override with extracted metadata
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = self.chunk_text(text)
        metadata['chunk_count'] = len(chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in chunks)
        return {
            'metadata': metadata,
            'chunks': chunks
        } 