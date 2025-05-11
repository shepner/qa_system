"""Test configuration and fixtures."""

import os
import pytest
from pathlib import Path
import yaml

@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "test_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file."""
    log_file = tmp_path / "test.log"
    yield str(log_file)
    if log_file.exists():
        log_file.unlink()

@pytest.fixture
def mock_file_scanner(mocker):
    """Mock FileScanner class."""
    return mocker.patch("qa_system.document_processors.FileScanner")

@pytest.fixture
def mock_processor_factory(mocker):
    """Mock processor factory function."""
    return mocker.patch("qa_system.document_processors.get_processor_for_file_type")

@pytest.fixture
def mock_embedding_generator(mocker):
    """Mock EmbeddingGenerator class."""
    return mocker.patch("qa_system.embedding.EmbeddingGenerator")

@pytest.fixture
def mock_vector_store(mocker):
    """Mock ChromaVectorStore class."""
    return mocker.patch("qa_system.vector_store.ChromaVectorStore")

@pytest.fixture
def mock_list_handler(mocker):
    """Mock ListHandler class."""
    return mocker.patch("qa_system.document_processors.ListHandler")

@pytest.fixture
def mock_query_processor(mocker):
    """Mock QueryProcessor class."""
    return mocker.patch("qa_system.query.QueryProcessor") 