import pytest
from qa_system.document_processors import (
    get_processor_for_file_type,
    ListHandler,
    RemoveHandler,
    TextDocumentProcessor,
    MarkdownDocumentProcessor,
    PDFDocumentProcessor
)

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

def test_remove_handler_init():
    handler = RemoveHandler(DummyConfig())
    assert handler is not None

def test_remove_handler_remove_documents_no_filter():
    handler = RemoveHandler(DummyConfig())
    result = handler.remove_documents(['test.txt'])
    assert result == {'removed': [], 'failed': {}, 'not_found': []}

def test_remove_handler_remove_documents_with_filter():
    handler = RemoveHandler(DummyConfig())
    result = handler.remove_documents(['test.txt'], filter_pattern='*.txt')
    assert result == {'removed': [], 'failed': {}, 'not_found': []} 