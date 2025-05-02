from .base_processor import BaseDocumentProcessor

class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for Markdown (.md) files. Extracts metadata, chunks text, and returns results.
    Attempts to preserve markdown structure in chunking.
    """
    def process(self, file_path, metadata=None):
        self.logger.debug(f"Processing markdown file: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Optionally, split on markdown headers for chunking, then further chunk if needed
        import re
        header_chunks = re.split(r'(^#+ .*$)', text, flags=re.MULTILINE)
        # Merge headers with their content
        merged_chunks = []
        i = 0
        while i < len(header_chunks):
            if header_chunks[i].startswith('#'):
                header = header_chunks[i]
                content = header_chunks[i+1] if i+1 < len(header_chunks) else ''
                merged_chunks.append(header + '\n' + content)
                i += 2
            else:
                if header_chunks[i].strip():
                    merged_chunks.append(header_chunks[i])
                i += 1
        # Now further chunk if any merged chunk is too large
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self.chunk_text(chunk))
            else:
                final_chunks.append(chunk)
        metadata['chunk_count'] = len(final_chunks)
        metadata['total_tokens'] = sum(len(chunk) for chunk in final_chunks)
        return {
            'metadata': metadata,
            'chunks': final_chunks
        } 