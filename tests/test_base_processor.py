import pytest
import tempfile
import os
from pathlib import Path
from qa_system.document_processors.base_processor import BaseDocumentProcessor
from qa_system.exceptions import ProcessingError

class DummyConfig:
    def get_nested(self, key, default=None):
        # Simulate config for chunking
        if key == 'DOCUMENT_PROCESSING.MAX_CHUNK_SIZE':
            return 50
        if key == 'DOCUMENT_PROCESSING.MIN_CHUNK_SIZE':
            return 10
        if key == 'DOCUMENT_PROCESSING.CHUNK_OVERLAP':
            return 10
        if key == 'DOCUMENT_PROCESSING.PRESERVE_SENTENCES':
            return True
        return default

class DummyProcessor(BaseDocumentProcessor):
    def process(self, file_path, metadata=None):
        # Just return metadata and chunked text for testing
        with open(file_path) as f:
            text = f.read()
        meta = self.extract_metadata(file_path)
        chunks = self.chunk_text(text)
        return {'metadata': meta, 'chunks': chunks}

def test_extract_metadata(tmp_path):
    file = tmp_path / 'test.txt'
    file.write_text('hello world')
    proc = BaseDocumentProcessor(DummyConfig())
    meta = proc.extract_metadata(str(file))
    assert meta['filename_full'] == 'test.txt'
    assert meta['file_type'] == 'txt'
    assert meta['path'].endswith('test.txt')
    assert 'created_at' in meta and 'last_modified' in meta

def test_chunk_text_sentence_aware():
    config = DummyConfig()
    proc = BaseDocumentProcessor(config)
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes."
    chunks = proc.chunk_text(text)
    # With max_chunk_size=50, should split into at least 2 chunks
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) >= 2
    # Each chunk should be at least min_chunk_size or only chunk
    assert all(len(c) >= 10 or len(chunks) == 1 for c in chunks)

def test_chunk_text_overlap():
    config = DummyConfig()
    proc = BaseDocumentProcessor(config)
    # 3 sentences, each 20 chars, max_chunk_size=50, overlap=10
    text = "A. " * 10 + "B. " * 10 + "C. " * 10
    text = text.strip()
    chunks = proc.chunk_text(text)
    # Overlap should ensure some repeated content
    if len(chunks) > 1:
        assert any(chunks[i][-10:] in chunks[i+1] for i in range(len(chunks)-1))

def test_process_not_implemented():
    proc = BaseDocumentProcessor(DummyConfig())
    with pytest.raises(ProcessingError):
        proc.run('fake.txt')

def test_logging_integration(tmp_path, caplog):
    file = tmp_path / 'fail.txt'
    file.write_text('fail')
    class FailingProcessor(BaseDocumentProcessor):
        def process(self, file_path, metadata=None):
            raise ValueError('fail!')
    proc = FailingProcessor(DummyConfig())
    with pytest.raises(ProcessingError):
        proc.run(str(file))
    assert 'Processing failed: fail!' in caplog.text 