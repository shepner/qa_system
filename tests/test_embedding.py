import pytest
from qa_system.embedding import EmbeddingGenerator

class DummyConfig:
    def get_nested(self, key, default=None):
        if key == 'EMBEDDING_MODEL.MODEL_NAME':
            return 'embedding-001'
        if key == 'EMBEDDING_MODEL.BATCH_SIZE':
            return 4
        if key == 'EMBEDDING_MODEL.MAX_LENGTH':
            return 3072
        if key == 'EMBEDDING_MODEL.DIMENSIONS':
            return 8
        return default

def test_generate_embeddings_basic():
    config = DummyConfig()
    generator = EmbeddingGenerator(config)
    texts = [f"chunk {i}" for i in range(6)]
    metadata = {"file_type": "txt", "filename": "test.txt"}
    result = generator.generate_embeddings(texts, metadata)
    assert 'vectors' in result and 'texts' in result and 'metadata' in result
    assert len(result['vectors']) == len(texts)
    assert len(result['vectors'][0]) == 8
    assert result['texts'] == texts
    assert all(m == metadata for m in result['metadata'])

def test_generate_embeddings_empty():
    config = DummyConfig()
    generator = EmbeddingGenerator(config)
    result = generator.generate_embeddings([], {"file_type": "txt"})
    assert result['vectors'] == []
    assert result['texts'] == []
    assert result['metadata'] == []

def test_generate_embeddings_batching():
    config = DummyConfig()
    generator = EmbeddingGenerator(config)
    # 10 texts, batch_size=4, should still return 10 vectors
    texts = [f"chunk {i}" for i in range(10)]
    metadata = {"file_type": "txt"}
    result = generator.generate_embeddings(texts, metadata)
    assert len(result['vectors']) == 10
    assert all(len(vec) == 8 for vec in result['vectors']) 