import pytest
from qa_system.document_processors import (
    get_processor_for_file_type,
    ListHandler,
    TextDocumentProcessor,
    MarkdownDocumentProcessor,
    PDFDocumentProcessor
)
from PIL import Image

class DummyConfig:
    def get_nested(self, key, default=None):
        return default

def test_get_processor_for_file_type_txt():
    processor = get_processor_for_file_type('test.txt', DummyConfig())
    assert isinstance(processor, TextDocumentProcessor)

def test_get_processor_for_file_type_md():
    processor = get_processor_for_file_type('test.md', DummyConfig())
    assert isinstance(processor, MarkdownDocumentProcessor)

def test_get_processor_for_file_type_pdf():
    processor = get_processor_for_file_type('test.pdf', DummyConfig())
    assert isinstance(processor, PDFDocumentProcessor)

def test_get_processor_for_file_type_unknown():
    processor = get_processor_for_file_type('test.unknown', DummyConfig())
    result = processor.process()
    assert result == {'chunks': [], 'metadata': {}}

def test_get_processor_for_file_type_no_extension():
    processor = get_processor_for_file_type('testfile', DummyConfig())
    result = processor.process()
    assert result == {'chunks': [], 'metadata': {}}

def test_list_handler_init():
    handler = ListHandler(DummyConfig())
    assert handler is not None

def test_list_handler_list_documents_no_filter():
    handler = ListHandler(DummyConfig())
    result = handler.list_documents()
    assert result == []

def test_list_handler_list_documents_with_filter():
    handler = ListHandler(DummyConfig())
    result = handler.list_documents(filter_pattern='*.txt')
    assert result == []

def test_get_processor_for_file_type_csv(tmp_path):
    csv_file = tmp_path / 'test.csv'
    csv_file.write_text('col1,col2\n1,2\n3,4')
    from qa_system.document_processors import get_processor_for_file_type
    from qa_system.document_processors.csv_processor import CSVDocumentProcessor
    processor = get_processor_for_file_type(str(csv_file), DummyConfig())
    assert isinstance(processor, CSVDocumentProcessor)
    result = processor.process(str(csv_file))
    assert 'header_fields' in result['metadata']
    assert 'row_count' in result['metadata']

def test_get_processor_for_file_type_vision(tmp_path):
    img_file = tmp_path / 'test.png'
    # Create a simple 1x1 PNG
    img = Image.new('RGB', (1, 1), color='white')
    img.save(str(img_file))
    from qa_system.document_processors import get_processor_for_file_type
    from qa_system.document_processors.vision_processor import VisionDocumentProcessor
    processor = get_processor_for_file_type(str(img_file), DummyConfig())
    assert isinstance(processor, VisionDocumentProcessor)
    result = processor.process(str(img_file))
    assert 'image_dimensions' in result['metadata']
    assert 'image_format' in result['metadata']
    assert 'vision_labels' in result['metadata']
    assert 'ocr_text' in result['metadata']
    assert 'processing_timestamp' in result['metadata'] 