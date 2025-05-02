# Ensure all tests in this file run in isolated processes to avoid ChromaDB singleton issues
import pytest
import tempfile
import shutil
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import VectorStoreError, ConnectionError, QueryError
from types import SimpleNamespace

# Mark all tests in this file to run in a forked process
pytestmark = pytest.mark.forked

@pytest.fixture
def temp_vector_config(tmp_path):
    # Provide a minimal config object with get_nested
    class DummyConfig:
        def get_nested(self, key, default=None):
            if key == 'VECTOR_STORE':
                return {
                    'PERSIST_DIRECTORY': str(tmp_path),
                    'COLLECTION_NAME': 'test_collection',
                    'DISTANCE_METRIC': 'cosine',
                    'TOP_K': 3
                }
            return default
    return DummyConfig()

def test_initialization(temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    assert store.collection_name == 'test_collection'
    assert store.top_k == 3

def test_add_and_query_embeddings(temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]]
    texts = ["chunk one", "chunk two"]
    metadatas = [
        {"path": "file1.txt", "chunk_index": 0},
        {"path": "file2.txt", "chunk_index": 1}
    ]
    store.add_embeddings(embeddings, texts, metadatas)
    # Query for a similar vector
    result = store.query([0.1, 0.2, 0.3], top_k=1)
    assert 'ids' in result
    assert len(result['ids'][0]) >= 1

def test_delete_embeddings(temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    embeddings = [[0.1, 0.2, 0.3]]
    texts = ["chunk one"]
    metadatas = [{"path": "file1.txt", "chunk_index": 0}]
    store.add_embeddings(embeddings, texts, metadatas)
    # Delete by metadata
    store.delete({"path": "file1.txt"})
    # After deletion, query should return no results for that metadata
    result = store.query([0.1, 0.2, 0.3], top_k=1, filter_criteria={"path": "file1.txt"})
    assert result['ids'][0] == []

def test_error_handling_on_bad_init(monkeypatch, temp_vector_config):
    # Simulate chromadb.Client raising an error
    import qa_system.vector_store
    monkeypatch.setattr(qa_system.vector_store.chromadb, 'Client', lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(ConnectionError):
        ChromaVectorStore(temp_vector_config)

def test_error_handling_on_add(monkeypatch, temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    monkeypatch.setattr(store.collection, 'add', lambda *a, **kw: (_ for _ in ()).throw(Exception("fail add")))
    with pytest.raises(VectorStoreError):
        store.add_embeddings([[0.1, 0.2, 0.3]], ["chunk"], [{"path": "file.txt"}])

def test_error_handling_on_query(monkeypatch, temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    monkeypatch.setattr(store.collection, 'query', lambda *a, **kw: (_ for _ in ()).throw(Exception("fail query")))
    with pytest.raises(QueryError):
        store.query([0.1, 0.2, 0.3])

def test_error_handling_on_delete(monkeypatch, temp_vector_config):
    store = ChromaVectorStore(temp_vector_config)
    monkeypatch.setattr(store.collection, 'delete', lambda *a, **kw: (_ for _ in ()).throw(Exception("fail delete")))
    with pytest.raises(VectorStoreError):
        store.delete({"path": "file.txt"}) 