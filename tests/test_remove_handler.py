import pytest
from unittest.mock import MagicMock, patch
from qa_system.remove_handler import RemoveHandler
from qa_system.exceptions import DocumentNotFoundError, RemovalError, ValidationError
import os

class DummyConfig:
    def get_nested(self, key, default=None):
        if key.startswith('VECTOR_STORE'):
            return {
                'PERSIST_DIRECTORY': '.',
                'COLLECTION_NAME': 'qa_documents',
                'DISTANCE_METRIC': 'cosine',
                'TOP_K': 10
            }
        return default

def make_docs(paths):
    return [{'id': f'id_{i}', 'path': os.path.abspath(p)} for i, p in enumerate(paths)]

@pytest.fixture
def handler():
    config = DummyConfig()
    with patch('qa_system.remove_handler.ChromaVectorStore') as MockVS:
        vs = MockVS.return_value
        vs.list_metadata.return_value = make_docs([
            '/docs/a.pdf', '/docs/b.pdf', '/docs/c.md', '/docs/d.txt'
        ])
        vs.delete.return_value = None
        return RemoveHandler(config)

def test_find_matches(handler):
    matches = handler.find_matches('*.pdf')
    assert all(doc['path'].endswith('.pdf') for doc in matches)
    assert len(matches) == 2

def test_remove_documents_success(handler):
    with patch.object(handler.vector_store, 'list_metadata', side_effect=[
        make_docs(['/docs/a.pdf', '/docs/b.pdf', '/docs/c.md', '/docs/d.txt']),
        []
    ]):
        result = handler.remove_documents('*.pdf')
        assert 'removed' in result
        assert set(result['removed']) == {'/docs/a.pdf', '/docs/b.pdf'}
        assert not result['failed']
        assert not result['not_found']

def test_remove_documents_not_found(handler):
    handler.vector_store.list_metadata.return_value = make_docs(['/docs/x.md'])
    result = handler.remove_documents('*.pdf')
    assert result['not_found']
    assert not result['removed']

def test_remove_documents_batch_failure(handler):
    def fail_delete(*args, **kwargs):
        raise Exception('delete failed')
    handler.vector_store.delete.side_effect = fail_delete
    with pytest.raises(RemovalError):
        handler.remove_documents('*.pdf')

def test_verify_removal(handler):
    handler.vector_store.list_metadata.return_value = make_docs(['/docs/x.md'])
    assert handler.verify_removal(['/docs/a.pdf'])
    handler.vector_store.list_metadata.return_value = make_docs(['/docs/a.pdf'])
    assert not handler.verify_removal(['/docs/a.pdf'])

def test_cleanup_failed_removal(handler):
    handler.vector_store.delete.return_value = None
    handler.cleanup_failed_removal('id_1')
    handler.vector_store.delete.assert_called_with({'id': 'id_1'}, require_confirmation=False) 