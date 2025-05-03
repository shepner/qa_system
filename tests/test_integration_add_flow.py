import pytest
from pathlib import Path

from qa_system.file_scanner import FileScanner
from qa_system.document_processors.text_processor import TextDocumentProcessor
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.config import Config

@pytest.fixture
def temp_config(tmp_path):
    # Minimal config for integration test
    return Config({
        'FILE_SCANNER': {
            'DOCUMENT_PATH': str(tmp_path),
            'ALLOWED_EXTENSIONS': ['txt'],
            'EXCLUDE_PATTERNS': [],
            'HASH_ALGORITHM': 'sha256',
            'SKIP_EXISTING': True,
        },
        'DOCUMENT_PROCESSING': {
            'MAX_CHUNK_SIZE': 1000,
            'MIN_CHUNK_SIZE': 200,
            'CHUNK_OVERLAP': 100,
            'PRESERVE_SENTENCES': True,
        },
        'VECTOR_STORE': {
            'TYPE': 'chroma',
            'PERSIST_DIRECTORY': str(tmp_path / "vector_store"),
            'COLLECTION_NAME': 'test_collection',
            'DISTANCE_METRIC': 'cosine',
            'TOP_K': 5,
        },
        'EMBEDDING_MODEL': {
            'MODEL_NAME': 'embedding-001',
            'BATCH_SIZE': 4,
            'MAX_LENGTH': 3072,
            'DIMENSIONS': 8,
        }
    })

def test_add_flow_integration(tmp_path, temp_config):
    # 1. Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document. It has several sentences. This is for integration testing.")

    # 2. Initialize components
    scanner = FileScanner(temp_config)
    processor = TextDocumentProcessor(temp_config)
    embedding_generator = EmbeddingGenerator(temp_config)
    vector_store = ChromaVectorStore(temp_config)

    # 3. Scan for files
    files = scanner.scan_files()
    assert any(f['path'].endswith("test.txt") for f in files)
    file_info = next(f for f in files if f['path'].endswith("test.txt"))

    # 4. Process the file
    metadata = {
        "path": str(test_file),
        "file_type": "txt",
        "filename": "test.txt",
        "checksum": file_info.get('hash') or file_info.get('checksum')
    }
    processed = processor.process(str(test_file), metadata)
    assert 'chunks' in processed
    assert len(processed['chunks']) > 0

    # 5. Generate embeddings
    embeddings = embedding_generator.generate_embeddings(processed['chunks'], processed['metadata'])
    assert 'vectors' in embeddings
    assert len(embeddings['vectors']) == len(processed['chunks'])

    # 6. Store in vector DB
    vector_store.add_embeddings(embeddings['vectors'], processed['chunks'], [processed['metadata']] * len(processed['chunks']))

    # 7. Query the vector DB to ensure the embedding is present
    result = vector_store.query(embeddings['vectors'][0], top_k=1)
    assert result['ids'][0], "No results found in vector DB"

    # 8. Re-scan: Should still find the file (no deduplication in this integration test)
    files_again = scanner.scan_files()
    file_info_again = next(f for f in files_again if f['path'].endswith("test.txt"))
    assert file_info_again is not None 