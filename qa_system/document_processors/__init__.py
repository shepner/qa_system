from qa_system.file_scanner import FileScanner

def get_processor_for_file_type(path, config):
    class DummyProcessor:
        def process(self):
            return {'chunks': [], 'metadata': {}}
    return DummyProcessor()

class ListHandler:
    def __init__(self, config): pass
    def list_documents(self, filter_pattern=None): return []

class RemoveHandler:
    def __init__(self, config): pass
    def remove_documents(self, paths, filter_pattern=None):
        return {'removed': [], 'failed': {}, 'not_found': []}
