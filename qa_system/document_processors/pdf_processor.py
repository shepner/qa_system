from .base_processor import BaseDocumentProcessor

class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF (.pdf) files. Extracts metadata, extracts text per page, chunks text, and returns results.
    Attempts to preserve page boundaries in chunking.
    """
    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing PDF file: {file_path}")
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Please install it with 'pip install pypdf'.")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            all_chunks = []
            page_texts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                page_texts.append(text)
                # Chunk per page, then further chunk if needed
                if len(text) > self.chunk_size:
                    all_chunks.extend(self.chunk_text(text))
                else:
                    all_chunks.append(text)
        metadata['chunk_count'] = len(all_chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in all_chunks)
        metadata['page_count'] = len(page_texts)
        return {
            'metadata': metadata,
            'chunks': all_chunks,
            'page_texts': page_texts  # for traceability
        } 