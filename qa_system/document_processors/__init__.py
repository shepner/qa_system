from qa_system.file_scanner import FileScanner
import logging
from .text_processor import TextDocumentProcessor
from .markdown_processor import MarkdownDocumentProcessor
from .pdf_processor import PDFDocumentProcessor

def get_processor_for_file_type(path, config):
    logger = logging.getLogger(__name__)
    logger.debug(f"Called get_processor_for_file_type(path={path}, config={config})")
    ext = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    if ext == 'txt':
        return TextDocumentProcessor(config)
    if ext == 'md':
        return MarkdownDocumentProcessor(config)
    if ext == 'pdf':
        return PDFDocumentProcessor(config)
    class DummyProcessor:
        def process(self):
            logger.debug("Called DummyProcessor.process()")
            return {'chunks': [], 'metadata': {}}
    return DummyProcessor()

class ListHandler:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called ListHandler.__init__(config={config})")
    def list_documents(self, filter_pattern=None):
        self.logger.debug(f"Called ListHandler.list_documents(filter_pattern={filter_pattern})")
        return []

class RemoveHandler:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called RemoveHandler.__init__(config={config})")
    def remove_documents(self, paths, filter_pattern=None):
        self.logger.debug(f"Called RemoveHandler.remove_documents(paths={paths}, filter_pattern={filter_pattern})")
        return {'removed': [], 'failed': {}, 'not_found': []}
