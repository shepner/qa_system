import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import os
from embed_files.document_processors.pdf_processor import PDFDocumentProcessor
from embed_files.embedding_system import EmbeddingGenerator

@pytest.fixture
def sample_config_dict():
    return {
        'DOCUMENT_PROCESSING': {
            'MAX_CHUNK_SIZE': 1500,
            'CHUNK_OVERLAP': 300,
            'PDF_HEADER_RECOGNITION': {
                'ENABLED': True,
                'PATTERNS': ['^[A-Z][^.]*$']
            }
        },
        'DOCUMENT_PATH': '/test/docs',
        'SECURITY': {
            'API_KEY': 'test_key'
        }
    }

@pytest.fixture
def sample_config_object(sample_config_dict):
    config = Mock()
    config.get_nested = lambda *args: sample_config_dict.get(args[0], {})
    config.config_path = '/path/to/config.yaml'
    config.DOCUMENT_PROCESSING = sample_config_dict['DOCUMENT_PROCESSING']
    config.SECURITY = sample_config_dict['SECURITY']
    return config

def test_init_with_config_dict(sample_config_dict):
    processor = PDFDocumentProcessor(sample_config_dict)
    assert processor.max_chunk_size == 1500
    assert processor.chunk_overlap == 300
    assert processor.header_patterns == ['^[A-Z][^.]*$']
    assert isinstance(processor.embedding_generator, EmbeddingGenerator)

def test_init_with_config_object(sample_config_object):
    processor = PDFDocumentProcessor(sample_config_object)
    assert processor.max_chunk_size == 1500
    assert processor.chunk_overlap == 300
    assert processor.header_patterns == ['^[A-Z][^.]*$']
    assert isinstance(processor.embedding_generator, EmbeddingGenerator)
    # Verify EmbeddingGenerator was initialized with config_path
    assert processor.embedding_generator.config_path == '/path/to/config.yaml'

def test_init_with_none_config():
    with pytest.raises(ValueError, match="Config cannot be None"):
        PDFDocumentProcessor(None)

def test_init_with_invalid_config_type():
    with pytest.raises(ValueError, match="Config must be either a dictionary or a Config object"):
        PDFDocumentProcessor("invalid_config")

def test_init_with_missing_document_processing():
    invalid_config = {'SECURITY': {}}
    with pytest.raises(ValueError, match="DOCUMENT_PROCESSING section must be a dictionary"):
        PDFDocumentProcessor(invalid_config)

@patch('fitz.open')
def test_process_pdf_with_text(mock_fitz_open, sample_config_dict, tmp_path):
    # Create a mock PDF document
    mock_doc = Mock()
    mock_doc.__len__ = lambda _: 2
    mock_page1, mock_page2 = Mock(), Mock()
    mock_page1.get_text.return_value = "Page 1 content"
    mock_page2.get_text.return_value = "Page 2 content"
    mock_doc.__iter__ = lambda _: iter([mock_page1, mock_page2])
    mock_doc.metadata = {
        'title': 'Test PDF',
        'author': 'Test Author',
        'subject': 'Test Subject'
    }
    mock_fitz_open.return_value.__enter__.return_value = mock_doc

    # Create a test PDF file
    test_pdf = tmp_path / "test.pdf"
    test_pdf.touch()

    processor = PDFDocumentProcessor(sample_config_dict)
    metadata = {
        'path': str(test_pdf),
        'file_type': 'pdf',
        'filename': test_pdf.name
    }

    result = processor.process(str(test_pdf), metadata)

    assert result['title'] == 'Test PDF'
    assert result['author'] == 'Test Author'
    assert result['subject'] == 'Test Subject'
    assert result['page_count'] == 2
    assert result['has_text_content'] is True
    assert len(result['chunks']) > 0
    assert 'embeddings' in result

@patch('fitz.open')
def test_process_pdf_without_text(mock_fitz_open, sample_config_dict, tmp_path):
    # Create a mock PDF document with no text content
    mock_doc = Mock()
    mock_doc.__len__ = lambda _: 1
    mock_page = Mock()
    mock_page.get_text.return_value = ""
    mock_doc.__iter__ = lambda _: iter([mock_page])
    mock_doc.metadata = {}
    mock_fitz_open.return_value.__enter__.return_value = mock_doc

    # Create a test PDF file
    test_pdf = tmp_path / "empty.pdf"
    test_pdf.touch()

    processor = PDFDocumentProcessor(sample_config_dict)
    metadata = {
        'path': str(test_pdf),
        'file_type': 'pdf',
        'filename': test_pdf.name
    }

    result = processor.process(str(test_pdf), metadata)

    assert result['has_text_content'] is False
    assert result['chunks'] == []
    assert 'embeddings' not in result

def test_process_invalid_pdf(sample_config_dict, tmp_path):
    # Create an invalid PDF file
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("Not a PDF file")

    processor = PDFDocumentProcessor(sample_config_dict)
    metadata = {
        'path': str(invalid_pdf),
        'file_type': 'pdf',
        'filename': invalid_pdf.name
    }

    with pytest.raises(ValueError, match="Invalid PDF file"):
        processor.process(str(invalid_pdf), metadata) 