import argparse
import logging
import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path

from qa_system.__main__ import (
    parse_args,
    process_add_files,
    process_list,
    process_remove,
    process_query,
    main
)
from qa_system.config import Config
from qa_system.exceptions import QASystemError

# Test data
TEST_CONFIG = Config({
    'LOGGING': {
        'LEVEL': 'INFO',
        'LOG_FILE': 'logs/test.log'
    }
})

@pytest.fixture
def mock_logger():
    with patch('qa_system.__main__.logger') as mock:
        yield mock

@pytest.fixture
def mock_config():
    with patch('qa_system.__main__.get_config', return_value=TEST_CONFIG) as mock:
        yield mock

@pytest.fixture
def mock_setup_logging():
    with patch('qa_system.__main__.setup_logging') as mock:
        yield mock

class TestParseArgs:
    def test_add_operation(self):
        """Test parsing add operation arguments"""
        with patch('sys.argv', ['qa_system', '--add', 'file1.txt', '--add', 'file2.txt']):
            args = parse_args()
            assert args.add == ['file1.txt', 'file2.txt']
            assert not args.list
            assert not args.remove
            assert args.query is None

    def test_list_operation(self):
        """Test parsing list operation arguments"""
        with patch('sys.argv', ['qa_system', '--list', '--filter', '*.pdf']):
            args = parse_args()
            assert args.list
            assert args.filter == '*.pdf'
            assert not args.add
            assert not args.remove
            assert args.query is None

    def test_remove_operation(self):
        """Test parsing remove operation arguments"""
        with patch('sys.argv', ['qa_system', '--remove', 'file1.txt', '--remove', 'file2.txt']):
            args = parse_args()
            assert args.remove == ['file1.txt', 'file2.txt']
            assert not args.list
            assert not args.add
            assert args.query is None

    def test_query_operation(self):
        """Test parsing query operation arguments"""
        with patch('sys.argv', ['qa_system', '--query', 'test query']):
            args = parse_args()
            assert args.query == 'test query'
            assert not args.list
            assert not args.add
            assert not args.remove

    def test_query_interactive_mode(self):
        """Test parsing query operation without query text"""
        with patch('sys.argv', ['qa_system', '--query']):
            args = parse_args()
            assert args.query == ''
            assert not args.list
            assert not args.add
            assert not args.remove

    def test_debug_flag(self):
        """Test parsing debug flag"""
        with patch('sys.argv', ['qa_system', '--list', '--debug']):
            args = parse_args()
            assert args.debug

    def test_custom_config(self):
        """Test parsing custom config path"""
        with patch('sys.argv', ['qa_system', '--list', '--config', 'custom_config.yaml']):
            args = parse_args()
            assert args.config == 'custom_config.yaml'

class TestProcessAddFiles:
    @pytest.fixture
    def mock_components(self):
        with patch('qa_system.document_processors.FileScanner') as mock_scanner, \
             patch('qa_system.vector_store.ChromaVectorStore') as mock_store, \
             patch('qa_system.embedding.EmbeddingGenerator') as mock_generator, \
             patch('qa_system.document_processors.get_processor_for_file_type') as mock_processor_factory:
            yield {
                'scanner': mock_scanner,
                'store': mock_store,
                'generator': mock_generator,
                'processor_factory': mock_processor_factory
            }

    def test_successful_processing(self, mock_components, mock_logger):
        """Test successful file processing"""
        # Setup mock returns
        mock_scanner = Mock()
        mock_store = Mock()
        mock_generator = Mock()
        mock_processor = Mock()
        
        mock_components['scanner'].return_value = mock_scanner
        mock_components['store'].return_value = mock_store
        mock_components['generator'].return_value = mock_generator
        mock_components['processor_factory'].return_value = mock_processor
        
        mock_scanner.scan_files.return_value = [{
            'path': 'test.txt',
            'needs_processing': True
        }]
        
        mock_processor.process.return_value = {
            'chunks': ['chunk1', 'chunk2'],
            'metadata': {'file_type': 'txt'}
        }
        
        mock_generator.generate_embeddings.return_value = {
            'vectors': [[0.1, 0.2], [0.3, 0.4]],
            'texts': ['chunk1', 'chunk2'],
            'metadata': [{'file_type': 'txt'}, {'file_type': 'txt'}]
        }
        
        # Call function
        result = process_add_files(['test.txt'], TEST_CONFIG)
        
        # Verify
        assert result == 0
        mock_scanner.scan_files.assert_called_once_with('test.txt')
        mock_processor.process.assert_called_once()
        mock_generator.generate_embeddings.assert_called_once()
        mock_store.add_embeddings.assert_called_once()
        mock_logger.info.assert_called()

    def test_skip_processed_file(self, mock_components, mock_logger):
        """Test skipping already processed file"""
        mock_scanner = Mock()
        mock_components['scanner'].return_value = mock_scanner
        mock_scanner.scan_files.return_value = [{
            'path': 'test.txt',
            'needs_processing': False
        }]
        
        result = process_add_files(['test.txt'], TEST_CONFIG)
        
        assert result == 0
        mock_logger.info.assert_any_call('Skipping already processed file: test.txt')

    def test_processing_error(self, mock_components, mock_logger):
        """Test handling of processing error"""
        mock_scanner = Mock()
        mock_components['scanner'].return_value = mock_scanner
        mock_scanner.scan_files.side_effect = Exception("Processing failed")
        
        result = process_add_files(['test.txt'], TEST_CONFIG)
        
        assert result == 1
        mock_logger.error.assert_called_once()

class TestProcessList:
    @pytest.fixture
    def mock_list_handler(self):
        with patch('qa_system.document_processors.ListHandler') as mock:
            yield mock

    def test_successful_listing(self, mock_list_handler, capsys):
        """Test successful document listing"""
        mock_handler = Mock()
        mock_list_handler.return_value = mock_handler
        mock_handler.list_documents.return_value = [{
            'path': 'test.txt',
            'metadata': {
                'file_type': 'txt',
                'chunk_count': 2,
                'last_modified': '2024-03-20'
            }
        }]
        
        result = process_list('*.txt', TEST_CONFIG)
        
        assert result == 0
        captured = capsys.readouterr()
        assert 'test.txt' in captured.out
        assert 'txt' in captured.out
        assert '2' in captured.out

    def test_empty_listing(self, mock_list_handler, capsys):
        """Test listing with no documents"""
        mock_handler = Mock()
        mock_list_handler.return_value = mock_handler
        mock_handler.list_documents.return_value = []
        
        result = process_list(None, TEST_CONFIG)
        
        assert result == 0
        captured = capsys.readouterr()
        assert 'No documents found' in captured.out

    def test_listing_error(self, mock_list_handler, mock_logger):
        """Test handling of listing error"""
        mock_handler = Mock()
        mock_list_handler.return_value = mock_handler
        mock_handler.list_documents.side_effect = Exception("Listing failed")
        
        result = process_list(None, TEST_CONFIG)
        
        assert result == 1
        mock_logger.error.assert_called_once()

class TestProcessRemove:
    @pytest.fixture
    def mock_remove_handler(self):
        with patch('qa_system.document_processors.RemoveHandler') as mock:
            yield mock

    def test_successful_removal(self, mock_remove_handler, capsys):
        """Test successful document removal"""
        mock_handler = Mock()
        mock_remove_handler.return_value = mock_handler
        mock_handler.remove_documents.return_value = {
            'removed': ['test.txt'],
            'failed': {},
            'not_found': []
        }
        
        result = process_remove(['test.txt'], None, TEST_CONFIG)
        
        assert result == 0
        captured = capsys.readouterr()
        assert 'Successfully removed' in captured.out
        assert 'test.txt' in captured.out

    def test_failed_removal(self, mock_remove_handler, capsys):
        """Test partially failed document removal"""
        mock_handler = Mock()
        mock_remove_handler.return_value = mock_handler
        mock_handler.remove_documents.return_value = {
            'removed': ['test1.txt'],
            'failed': {'test2.txt': 'Access denied'},
            'not_found': ['test3.txt']
        }
        
        result = process_remove(['test1.txt', 'test2.txt', 'test3.txt'], None, TEST_CONFIG)
        
        assert result == 1
        captured = capsys.readouterr()
        assert 'Successfully removed' in captured.out
        assert 'Failed to remove' in captured.out
        assert 'No matches found' in captured.out

    def test_no_paths_provided(self, mock_remove_handler, mock_logger):
        """Test removal with no paths provided"""
        result = process_remove([], None, TEST_CONFIG)
        
        assert result == 1
        mock_logger.error.assert_called_once()

class TestProcessQuery:
    @pytest.fixture
    def mock_query_processor(self):
        with patch('qa_system.query.QueryProcessor') as mock:
            yield mock

    def test_single_query(self, mock_query_processor, capsys):
        """Test processing single query"""
        mock_processor = Mock()
        mock_query_processor.return_value = mock_processor
        mock_response = Mock(
            text="Test response",
            sources=[Mock(document='doc1.txt', similarity=0.9)]
        )
        mock_processor.process_query.return_value = mock_response
        
        result = process_query("test query", TEST_CONFIG)
        
        assert result == 0
        captured = capsys.readouterr()
        assert 'Test response' in captured.out
        assert 'doc1.txt' in captured.out
        assert '0.90' in captured.out

    def test_interactive_mode(self, mock_query_processor, capsys):
        """Test interactive query mode"""
        mock_processor = Mock()
        mock_query_processor.return_value = mock_processor
        mock_response = Mock(
            text="Test response",
            sources=[Mock(document='doc1.txt', similarity=0.9)]
        )
        mock_processor.process_query.return_value = mock_response
        
        with patch('builtins.input', side_effect=['test query', 'exit']):
            result = process_query(None, TEST_CONFIG)
        
        assert result == 0
        captured = capsys.readouterr()
        assert 'Test response' in captured.out
        assert 'doc1.txt' in captured.out

    def test_query_error(self, mock_query_processor, mock_logger):
        """Test handling of query error"""
        mock_processor = Mock()
        mock_query_processor.return_value = mock_processor
        mock_processor.process_query.side_effect = Exception("Query failed")
        
        result = process_query("test query", TEST_CONFIG)
        
        assert result == 1
        mock_logger.error.assert_called_once()

class TestMain:
    def test_successful_execution(self, mock_config, mock_setup_logging):
        """Test successful main execution"""
        with patch('qa_system.__main__.process_list', return_value=0) as mock_list:
            with patch('sys.argv', ['qa_system', '--list']):
                result = main()
                
                assert result == 0
                mock_config.assert_called_once()
                mock_setup_logging.assert_called_once()
                mock_list.assert_called_once()

    def test_qa_system_error(self, mock_config, mock_setup_logging, mock_logger):
        """Test handling of QASystemError"""
        with patch('qa_system.__main__.process_list', side_effect=QASystemError("Test error")):
            with patch('sys.argv', ['qa_system', '--list']):
                result = main()
                
                assert result == 1
                mock_logger.error.assert_called_once_with("Test error")

    def test_unexpected_error(self, mock_config, mock_setup_logging, mock_logger):
        """Test handling of unexpected error"""
        with patch('qa_system.__main__.process_list', side_effect=Exception("Unexpected error")):
            with patch('sys.argv', ['qa_system', '--list']):
                result = main()
                
                assert result == 1
                mock_logger.critical.assert_called_once() 