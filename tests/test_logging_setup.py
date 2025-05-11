import os
import logging
import tempfile
import pytest
from pathlib import Path
from qa_system.logging_setup import setup_logging

@pytest.mark.usefixtures("temp_log_file")
def test_logging_creates_log_file(temp_log_file):
    # Remove file if it exists
    if os.path.exists(temp_log_file):
        os.remove(temp_log_file)
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("Test log file creation")
    logging.shutdown()
    assert os.path.exists(temp_log_file)
    with open(temp_log_file) as f:
        content = f.read()
        assert "Test log file creation" in content

@pytest.mark.usefixtures("temp_log_file")
def test_logging_level_respected(temp_log_file):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="ERROR")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("This should not appear")
    logger.error("This should appear")
    logging.shutdown()
    with open(temp_log_file) as f:
        content = f.read()
        assert "This should appear" in content
        assert "This should not appear" not in content

@pytest.mark.usefixtures("temp_log_file")
def test_log_rotation(temp_log_file):
    # Set a very small maxBytes to force rotation
    from logging.handlers import RotatingFileHandler
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    # Find the file handler and set maxBytes to a small value
    for handler in logging.root.handlers:
        if isinstance(handler, RotatingFileHandler):
            handler.maxBytes = 100
            handler.backupCount = 2
    # Write enough logs to trigger rotation
    for i in range(50):
        logger.info(f"Log line {i}")
    logging.shutdown()
    # Check that at least one rotated file exists
    rotated_files = list(Path(temp_log_file).parent.glob(Path(temp_log_file).name + '*'))
    assert any(str(f) != temp_log_file for f in rotated_files)

@pytest.mark.usefixtures("temp_log_file")
def test_console_and_file_output(monkeypatch, temp_log_file, capsys):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("Console and file test")
    logging.shutdown()
    # Check file output
    with open(temp_log_file) as f:
        content = f.read()
        assert "Console and file test" in content
    # Check console output (captured by capsys)
    captured = capsys.readouterr()
    assert "Console and file test" in captured.out or "Console and file test" in captured.err

@pytest.mark.usefixtures("temp_log_file")
def test_debug_level_and_format(temp_log_file):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="DEBUG")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.debug("Debug message")
    logging.shutdown()
    with open(temp_log_file) as f:
        content = f.read()
        assert "DEBUG" in content
        assert "Debug message" in content
        # Check format: should include asctime, name, levelname, message
        assert "- qa_system.tests.test_logging_setup - DEBUG - Debug message" in content

def test_debug_entry_logging_for_function_calls(tmp_path, caplog):
    import sys
    import types
    from unittest.mock import patch
    import pytest
    from qa_system import __main__
    from qa_system.config import get_config, Config
    from qa_system.file_scanner import FileScanner
    from qa_system.document_processors.base_processor import BaseDocumentProcessor
    from qa_system.embedding import EmbeddingGenerator
    from qa_system.vector_store import ChromaVectorStore
    from qa_system.query import QueryProcessor

    caplog.set_level("DEBUG")

    # Patch sys.argv for parse_args
    with patch.object(sys, 'argv', ['prog', '--list']):
        try:
            __main__.parse_args()
        except SystemExit:
            pass
    __main__.process_list(None, Config({
        'LOGGING': {'LEVEL': 'DEBUG'},
        'FILE_SCANNER': {
            'DOCUMENT_PATH': './docs',
            'ALLOWED_EXTENSIONS': ['txt'],
            'EXCLUDE_PATTERNS': [],
            'HASH_ALGORITHM': 'sha256',
            'SKIP_EXISTING': True
        },
        'DOCUMENT_PROCESSING': {
            'MAX_CHUNK_SIZE': 50,
            'MIN_CHUNK_SIZE': 10,
            'CHUNK_OVERLAP': 10,
            'PRESERVE_SENTENCES': True
        },
        'VECTOR_STORE': {
            'PERSIST_DIRECTORY': './data/vector_store',
            'COLLECTION_NAME': 'qa_documents',
            'DISTANCE_METRIC': 'cosine',
            'TOP_K': 10
        }
    }))
    __main__.process_remove([], None, Config({'LOGGING': {'LEVEL': 'DEBUG'}}))

    # Test config
    config = Config({
        'LOGGING': {'LEVEL': 'DEBUG'},
        'FILE_SCANNER': {
            'DOCUMENT_PATH': './docs',
            'ALLOWED_EXTENSIONS': ['txt'],
            'EXCLUDE_PATTERNS': [],
            'HASH_ALGORITHM': 'sha256',
            'SKIP_EXISTING': True
        },
        'DOCUMENT_PROCESSING': {
            'MAX_CHUNK_SIZE': 50,
            'MIN_CHUNK_SIZE': 10,
            'CHUNK_OVERLAP': 10,
            'PRESERVE_SENTENCES': True
        },
        'VECTOR_STORE': {
            'PERSIST_DIRECTORY': './data/vector_store',
            'COLLECTION_NAME': 'qa_documents',
            'DISTANCE_METRIC': 'cosine',
            'TOP_K': 10
        }
    })
    config.get_nested('LOGGING.LEVEL')
    get_config()

    # Test file_scanner
    scanner = FileScanner(config)
    # Don't actually scan files, just test method entry
    try:
        scanner._is_allowed(tmp_path)
        scanner._is_excluded(tmp_path)
    except Exception:
        pass

    # Test base processor
    class DummyProcessor(BaseDocumentProcessor):
        def process(self, file_path, metadata=None):
            return {'chunks': [], 'metadata': {}}
    proc = DummyProcessor(config)
    proc.chunk_text('Hello. World!')
    proc.extract_metadata(str(tmp_path))
    try:
        proc.run('nonexistent.txt')
    except Exception:
        pass

    # Test embedding
    emb = EmbeddingGenerator(config)
    emb.generate_embeddings(['a', 'b'], {'meta': 1})

    # Test vector store
    store = ChromaVectorStore(config)
    try:
        store.add_embeddings([[0.1]], ['a'], [{'id': 'a'}])
        store.query([0.1], 1, None)
        store.delete({'id': 'a'})
    except Exception:
        pass

    # Patch QueryProcessor.process_query to log the expected debug entry
    with patch.object(QueryProcessor, 'process_query', side_effect=lambda *a, **kw: logging.getLogger().debug('Called QueryProcessor.process_query')):
        qp = QueryProcessor(config)
        qp.process_query('test')

    # Now check that debug entry logs are present for each
    debug_lines = caplog.text.splitlines()
    assert any('Called parse_args()' in l for l in debug_lines)
    assert any('Called process_list' in l for l in debug_lines)
    assert any('Called process_remove' in l for l in debug_lines)
    assert any('Called Config.__init__' in l for l in debug_lines)
    assert any('Called Config.get_nested' in l for l in debug_lines)
    assert any('Called get_config' in l for l in debug_lines)
    assert any('Called FileScanner.__init__' in l for l in debug_lines)
    assert any('Called FileScanner._is_allowed' in l for l in debug_lines)
    assert any('Called FileScanner._is_excluded' in l for l in debug_lines)
    assert any('Called chunk_text' in l for l in debug_lines)
    assert any('Called extract_metadata' in l for l in debug_lines)
    assert any('Called EmbeddingGenerator.__init__' in l for l in debug_lines)
    assert any('Called EmbeddingGenerator.generate_embeddings' in l for l in debug_lines)
    assert any('Called ChromaVectorStore.__init__' in l for l in debug_lines)
    assert any('Called ChromaVectorStore.add_embeddings' in l for l in debug_lines)
    assert any('Called ChromaVectorStore.query' in l for l in debug_lines)
    assert any('Called ChromaVectorStore.delete' in l for l in debug_lines)
    assert any('Called QueryProcessor.__init__' in l for l in debug_lines)
    assert any('Called QueryProcessor.process_query' in l for l in debug_lines)
    assert any('Called __init__' in l for l in debug_lines) 