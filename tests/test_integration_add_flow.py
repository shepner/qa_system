import pytest
from pathlib import Path
from PIL import Image

from qa_system.file_scanner import FileScanner
from qa_system.document_processors.text_processor import TextDocumentProcessor
from qa_system.embedding import EmbeddingGenerator
from qa_system.vector_store import ChromaVectorStore
from qa_system.config import Config
from qa_system.document_processors import get_processor_for_file_type

@pytest.fixture
def temp_config(tmp_path):
    # Minimal config for integration test
    return Config({
        'FILE_SCANNER': {
            'DOCUMENT_PATH': str(tmp_path),
            'ALLOWED_EXTENSIONS': ['txt', 'csv', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
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
    vector_store.add_embeddings(embeddings['vectors'], [chunk['text'] for chunk in processed['chunks']], [processed['metadata']] * len(processed['chunks']))

    # 7. Query the vector DB to ensure the embedding is present
    result = vector_store.query(embeddings['vectors'][0], top_k=1)
    assert result['ids'][0], "No results found in vector DB"

    # 8. Re-scan: Should still find the file (no deduplication in this integration test)
    files_again = scanner.scan_files()
    file_info_again = next(f for f in files_again if f['path'].endswith("test.txt"))
    assert file_info_again is not None

def test_add_flow_with_csv_and_image(tmp_path, temp_config):
    # Create a CSV file
    csv_file = tmp_path / 'test.csv'
    csv_file.write_text('a,b\n1,2\n3,4')
    # Create an image file
    img_file = tmp_path / 'test.jpg'
    img = Image.new('RGB', (2, 2), color='blue')
    img.save(str(img_file))
    # Scan files
    scanner = FileScanner(temp_config)
    files = scanner.scan_files(str(tmp_path))
    # Should find both files
    found = {f['path'] for f in files}
    assert any('test.csv' in f for f in found)
    assert any('test.jpg' in f for f in found)
    # Process CSV
    csv_proc = get_processor_for_file_type(str(csv_file), temp_config)
    csv_result = csv_proc.process(str(csv_file))
    meta = csv_result['metadata']
    assert 'header_fields' in meta
    assert 'row_count' in meta
    assert 'chunk_count' in meta
    assert 'total_tokens' in meta
    assert 'urls' in meta
    # Check at least one chunk for all required fields
    if csv_result['chunks']:
        chunk = csv_result['chunks'][0]
        assert 'text' in chunk
        cmeta = chunk['metadata']
        for field in [
            'chunk_index', 'start_offset', 'end_offset', 'tags', 'urls', 'url_contexts', 'topics', 'summary']:
            assert field in cmeta
    # Process image
    img_proc = get_processor_for_file_type(str(img_file), temp_config)
    img_result = img_proc.process(str(img_file))
    meta = img_result['metadata']
    for field in [
        'image_dimensions', 'image_format', 'color_profile', 'vision_labels', 'ocr_text',
        'face_detection', 'safe_search', 'feature_confidence', 'processing_timestamp', 'error_states',
        'chunk_count', 'total_tokens']:
        assert field in meta
    # Check at least one chunk for all required fields
    if img_result['chunks']:
        chunk = img_result['chunks'][0]
        assert 'text' in chunk
        cmeta = chunk['metadata']
        for field in [
            'chunk_index', 'start_offset', 'end_offset', 'chunk_type', 'urls', 'url_contexts', 'topics', 'summary']:
            assert field in cmeta 