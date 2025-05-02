import os
import tempfile
import yaml
import pytest
from qa_system.config import get_config, Config

CONFIG_YAML = """
LOGGING:
  LEVEL: "DEBUG"
  LOG_FILE: "test.log"
SECURITY:
  GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS}
  GOOGLE_CLOUD_PROJECT: ${GOOGLE_CLOUD_PROJECT}
  GOOGLE_VISION_API_KEY: ${GOOGLE_VISION_API_KEY}
FILE_SCANNER:
  DOCUMENT_PATH: "./docs"
  ALLOWED_EXTENSIONS:
    - "txt"
    - "md"
    - "pdf"
  EXCLUDE_PATTERNS:
    - ".*"
    - "__pycache__"
    - "*.pyc"
  HASH_ALGORITHM: "sha256"
  SKIP_EXISTING: true
DOCUMENT_PROCESSING:
  MAX_CHUNK_SIZE: 3072
  MIN_CHUNK_SIZE: 1024
  CHUNK_OVERLAP: 768
  CONCURRENT_TASKS: 6
  BATCH_SIZE: 50
  PRESERVE_SENTENCES: true
VECTOR_STORE:
  TYPE: "chroma"
  PERSIST_DIRECTORY: "./data/vector_store"
  COLLECTION_NAME: "qa_documents"
  DISTANCE_METRIC: "cosine"
  TOP_K: 40
"""

def test_get_config_and_nested_access(monkeypatch):
    # Set environment variables for SECURITY section
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key")

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp.write(CONFIG_YAML)
        tmp.flush()
        config = get_config(tmp.name)

    # Test nested access
    assert config.get_nested("LOGGING.LEVEL") == "DEBUG"
    assert config.get_nested("LOGGING.LOG_FILE") == "test.log"
    assert config.get_nested("FILE_SCANNER.DOCUMENT_PATH") == "./docs"
    assert config.get_nested("DOCUMENT_PROCESSING.BATCH_SIZE") == 50
    assert config.get_nested("VECTOR_STORE.TYPE") == "chroma"
    # Test default value
    assert config.get_nested("NONEXISTENT.PATH", default="default") == "default"
    # Test environment variable substitution
    assert config.get_nested("SECURITY.GOOGLE_APPLICATION_CREDENTIALS") == "/tmp/creds.json"
    assert config.get_nested("SECURITY.GOOGLE_CLOUD_PROJECT") == "test-project"
    assert config.get_nested("SECURITY.GOOGLE_VISION_API_KEY") == "test-key"


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_config("/nonexistent/path/config.yaml")


def test_invalid_yaml(tmp_path):
    bad_yaml = "LOGGING: [unclosed_list"
    file_path = tmp_path / "bad.yaml"
    file_path.write_text(bad_yaml)
    with pytest.raises(Exception):
        get_config(str(file_path)) 