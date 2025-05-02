import pytest
from pathlib import Path
from qa_system.document_processors.markdown_processor import MarkdownDocumentProcessor

class DummyConfig:
    def get_nested(self, key, default=None):
        if key == 'DOCUMENT_PROCESSING.MAX_CHUNK_SIZE':
            return 50
        if key == 'DOCUMENT_PROCESSING.MIN_CHUNK_SIZE':
            return 10
        if key == 'DOCUMENT_PROCESSING.CHUNK_OVERLAP':
            return 10
        if key == 'DOCUMENT_PROCESSING.PRESERVE_SENTENCES':
            return True
        return default

def test_markdown_processor_basic(tmp_path):
    file = tmp_path / 'sample.md'
    file.write_text('# Header 1\nThis is a test.\n## Header 2\nAnother section!\nNo header here.')
    proc = MarkdownDocumentProcessor(DummyConfig())
    result = proc.process(str(file))
    assert 'metadata' in result
    assert 'chunks' in result
    assert result['metadata']['filename_full'] == 'sample.md'
    assert result['metadata']['file_type'] == 'md'
    assert result['metadata']['chunk_count'] == len(result['chunks'])
    assert all(isinstance(chunk, str) for chunk in result['chunks'])
    assert sum(len(chunk) for chunk in result['chunks']) == result['metadata']['total_tokens']
    # Should split on headers
    assert any(chunk.startswith('# Header 1') for chunk in result['chunks'])
    assert any(chunk.startswith('## Header 2') for chunk in result['chunks'])

def test_markdown_processor_empty_file(tmp_path):
    file = tmp_path / 'empty.md'
    file.write_text('')
    proc = MarkdownDocumentProcessor(DummyConfig())
    result = proc.process(str(file))
    assert result['chunks'] == [''] or result['chunks'] == []
    assert result['metadata']['chunk_count'] == len(result['chunks'])

def test_markdown_processor_metadata_override(tmp_path):
    file = tmp_path / 'meta.md'
    file.write_text('# Title\nHello world.')
    proc = MarkdownDocumentProcessor(DummyConfig())
    custom_meta = {'custom': 'value'}
    result = proc.process(str(file), metadata=custom_meta)
    assert result['metadata']['custom'] == 'value'
    assert result['metadata']['filename_full'] == 'meta.md' 