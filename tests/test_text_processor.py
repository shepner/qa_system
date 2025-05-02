import pytest
from pathlib import Path
from qa_system.document_processors.text_processor import TextDocumentProcessor

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

def test_text_processor_basic(tmp_path):
    file = tmp_path / 'sample.txt'
    file.write_text('This is a test. This is another sentence! And a third one?')
    proc = TextDocumentProcessor(DummyConfig())
    result = proc.process(str(file))
    assert 'metadata' in result
    assert 'chunks' in result
    assert result['metadata']['filename_full'] == 'sample.txt'
    assert result['metadata']['file_type'] == 'txt'
    assert result['metadata']['chunk_count'] == len(result['chunks'])
    assert all(isinstance(chunk, str) for chunk in result['chunks'])
    assert sum(len(chunk) for chunk in result['chunks']) == result['metadata']['total_tokens']

def test_text_processor_empty_file(tmp_path):
    file = tmp_path / 'empty.txt'
    file.write_text('')
    proc = TextDocumentProcessor(DummyConfig())
    result = proc.process(str(file))
    assert result['chunks'] == [''] or result['chunks'] == []
    assert result['metadata']['chunk_count'] == len(result['chunks'])

def test_text_processor_metadata_override(tmp_path):
    file = tmp_path / 'meta.txt'
    file.write_text('Hello world.')
    proc = TextDocumentProcessor(DummyConfig())
    custom_meta = {'custom': 'value'}
    result = proc.process(str(file), metadata=custom_meta)
    assert result['metadata']['custom'] == 'value'
    assert result['metadata']['filename_full'] == 'meta.txt' 