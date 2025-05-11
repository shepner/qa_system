from .base_processor import BaseDocumentProcessor
import re
import csv
import io

class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF (.pdf) files. Extracts metadata, extracts text per page, chunks text, and returns results.
    Attempts to preserve page boundaries in chunking.
    Skips password-protected PDFs with a warning.
    Extracts URLs from all text and stores as CSV string in metadata.
    """
    def _list_to_csv(self, items):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(items)
        return output.getvalue().strip()

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
            try:
                reader = pypdf.PdfReader(f)
            except Exception as e:
                # Handle cryptography errors for encrypted PDFs
                if 'cryptography' in str(e) and 'AES' in str(e):
                    self.logger.warning(f"File {file_path} is encrypted (AES) and cannot be processed: {e}. Skipping.")
                    metadata['skipped'] = True
                    metadata['skip_reason'] = 'encrypted-pdf-unsupported-crypto'
                    return {
                        'metadata': metadata,
                        'chunks': [],
                        'page_texts': []
                    }
                raise
            if reader.is_encrypted:
                self.logger.warning(f"File {file_path} is password-protected and will be skipped.")
                metadata['skipped'] = True
                metadata['skip_reason'] = 'password-protected'
                return {
                    'metadata': metadata,
                    'chunks': [],
                    'page_texts': []
                }
            all_chunks = []
            page_texts = []
            all_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                page_texts.append(text)
                all_text.append(text)
                if len(text) > self.chunk_size:
                    all_chunks.extend(self.chunk_text(text))
                else:
                    all_chunks.append(text)
        # Extract URLs from all text
        urls = set()
        full_text = '\n'.join(all_text)
        for m in re.finditer(r'(https?://[^\s)\]\'"<>]+|ftp://[^\s)\]\'"<>]+)', full_text):
            urls.add(m.group(1))
        metadata['urls'] = self._list_to_csv(sorted(urls))
        metadata['chunk_count'] = len(all_chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in all_chunks)
        metadata['page_count'] = len(page_texts)
        return {
            'metadata': metadata,
            'chunks': all_chunks,
            'page_texts': page_texts  # for traceability
        } 