import pytest
from unittest.mock import patch, MagicMock
import sys
from qa_system.__main__ import process_add_files, process_list, process_remove, process_query, main
from qa_system.exceptions import QASystemError

class DummyConfig:
    def get_nested(self, key, default=None):
        return default

@patch('qa_system.document_processors.FileScanner')
@patch('qa_system.vector_store.ChromaVectorStore')
@patch('qa_system.embedding.EmbeddingGenerator')
def test_process_add_files_file_not_found(mock_generator, mock_store, mock_scanner):
    config = DummyConfig()
    config.get_nested = lambda key, default=None: {
        'FILE_SCANNER.DOCUMENT_PATH': '.',
        'FILE_SCANNER.ALLOWED_EXTENSIONS': ['txt'],
        'FILE_SCANNER.EXCLUDE_PATTERNS': [],
        'FILE_SCANNER.HASH_ALGORITHM': 'sha256',
        'VECTOR_STORE.PERSIST_DIRECTORY': '.',
        'EMBEDDING.MODEL': 'test-model'
    }.get(key, default)
    
    # Make the scanner raise FileNotFoundError
    mock_scanner.return_value.scan_files.side_effect = FileNotFoundError("File not found")
    
    # The error should be caught and return 1
    result = process_add_files(['nonexistent.txt'], config)
    assert result == 1

def test_process_add_files_empty_list():
    config = DummyConfig()
    config.get_nested = lambda key, default=None: {
        'FILE_SCANNER.DOCUMENT_PATH': '.',
        'FILE_SCANNER.ALLOWED_EXTENSIONS': ['txt'],
        'FILE_SCANNER.EXCLUDE_PATTERNS': [],
        'FILE_SCANNER.HASH_ALGORITHM': 'sha256'
    }.get(key, default)
    result = process_add_files([], config)
    assert result == 1  # Empty list should be treated as an error

@patch('qa_system.document_processors.FileScanner')
@patch('qa_system.vector_store.ChromaVectorStore')
@patch('qa_system.embedding.EmbeddingGenerator')
def test_process_add_files_scanner_error(mock_scanner, mock_store, mock_generator):
    mock_scanner.side_effect = Exception("Scanner error")
    result = process_add_files(['test.txt'], DummyConfig())
    assert result == 1

def test_process_list_empty():
    result = process_list(None, DummyConfig())
    assert result == 0

@patch('qa_system.document_processors.ListHandler')
def test_process_list_with_documents(mock_handler):
    mock_handler.return_value.list_documents.return_value = [
        {
            'path': 'test.txt',
            'metadata': {
                'file_type': 'txt',
                'chunk_count': 5,
                'last_modified': '2024-03-20'
            }
        }
    ]
    result = process_list(None, DummyConfig())
    assert result == 0

@patch('qa_system.document_processors.ListHandler')
def test_process_list_error(mock_handler):
    mock_handler.side_effect = Exception("List error")
    result = process_list(None, DummyConfig())
    assert result == 1

def test_process_remove_no_paths():
    result = process_remove([], None, DummyConfig())
    assert result == 1

@patch('qa_system.query.QueryProcessor')
def test_process_query_single(mock_processor):
    mock_response = MagicMock()
    mock_response.text = "Answer text"
    mock_response.sources = [
        MagicMock(document="doc1.txt", similarity=0.9),
        MagicMock(document="doc2.txt", similarity=0.8)
    ]
    mock_processor.return_value.process_query.return_value = mock_response
    result = process_query("test query", DummyConfig())
    assert result == 0

@patch('qa_system.query.QueryProcessor')
def test_process_query_error(mock_processor):
    mock_processor.side_effect = Exception("Query error")
    result = process_query("test query", DummyConfig())
    assert result == 1

@patch('qa_system.query.QueryProcessor')
@patch('builtins.input', side_effect=['test query', 'exit'])
def test_process_query_interactive(mock_input, mock_processor):
    mock_response = MagicMock()
    mock_response.text = "Answer text"
    mock_response.sources = [
        MagicMock(document="doc1.txt", similarity=0.9)
    ]
    mock_processor.return_value.process_query.return_value = mock_response
    result = process_query(None, DummyConfig())
    assert result == 0

@patch('qa_system.query.QueryProcessor')
@patch('builtins.input', side_effect=['test query', KeyboardInterrupt])
def test_process_query_interactive_keyboard_interrupt(mock_input, mock_processor):
    mock_response = MagicMock()
    mock_processor.return_value.process_query.return_value = mock_response
    result = process_query(None, DummyConfig())
    assert result == 0

@patch('qa_system.__main__.parse_args')
def test_main_success(mock_parse_args):
    mock_args = MagicMock()
    mock_args.debug = False
    mock_args.config = None
    mock_args.add = None
    mock_args.list = True
    mock_args.remove = None
    mock_args.query = None
    mock_args.filter = None
    mock_parse_args.return_value = mock_args
    
    result = main()
    assert result == 0

@patch('qa_system.__main__.parse_args')
def test_main_qa_system_error(mock_parse_args):
    mock_parse_args.side_effect = QASystemError("Test error")
    result = main()
    assert result == 1

@patch('qa_system.__main__.parse_args')
def test_main_unexpected_error(mock_parse_args):
    mock_parse_args.side_effect = Exception("Unexpected error")
    result = main()
    assert result == 1 