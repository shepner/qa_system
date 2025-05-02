import os
import shutil
import pytest
from pathlib import Path
from qa_system.file_scanner import FileScanner
from qa_system.exceptions import ValidationError, QASystemError

class DummyConfig:
    def __init__(self, data):
        self._data = data
    def get_nested(self, path, default=None):
        return self._data.get(path, default)
    def get(self, key, default=None):
        return self._data.get(key, default)

def make_config(tmp_path, extra=None):
    base = {
        'DOCUMENT_PATH': str(tmp_path),
        'ALLOWED_EXTENSIONS': ['txt', 'md'],
        'EXCLUDE_PATTERNS': ['__pycache__', '*.pyc'],
        'HASH_ALGORITHM': 'sha256',
        'SKIP_EXISTING': True,
    }
    if extra:
        base.update(extra)
    return DummyConfig({'FILE_SCANNER': base})

def test_scan_files_basic(tmp_path):
    # Create files
    (tmp_path / 'a.txt').write_text('hello')
    (tmp_path / 'b.md').write_text('world')
    (tmp_path / 'c.pdf').write_text('nope')
    (tmp_path / '__pycache__').mkdir()
    (tmp_path / '__pycache__' / 'd.txt').write_text('skip')
    config = make_config(tmp_path)
    scanner = FileScanner(config)
    results = scanner.scan_files()
    paths = {Path(f['path']).name for f in results}
    assert 'a.txt' in paths
    assert 'b.md' in paths
    assert 'c.pdf' not in paths
    assert 'd.txt' not in paths
    for f in results:
        assert 'hash' in f and len(f['hash']) == 64
        assert f['size'] > 0

def test_scan_files_empty(tmp_path):
    config = make_config(tmp_path)
    scanner = FileScanner(config)
    results = scanner.scan_files()
    assert results == []

def test_scan_files_invalid_path(tmp_path):
    config = make_config(tmp_path)
    scanner = FileScanner(config)
    with pytest.raises(ValidationError):
        scanner.scan_files(path=str(tmp_path / 'not_a_dir'))

def test_scan_files_unsupported_hash(tmp_path):
    (tmp_path / 'a.txt').write_text('data')
    config = make_config(tmp_path, {'HASH_ALGORITHM': 'notarealhash'})
    scanner = FileScanner(config)
    with pytest.raises(QASystemError):
        scanner.scan_files()

def test_scan_files_exclude_pattern(tmp_path):
    (tmp_path / 'foo.txt').write_text('x')
    (tmp_path / 'bar.md').write_text('y')
    (tmp_path / 'skip.txt').write_text('z')
    config = make_config(tmp_path, {'EXCLUDE_PATTERNS': ['skip.*']})
    scanner = FileScanner(config)
    results = scanner.scan_files()
    names = {Path(f['path']).name for f in results}
    assert 'foo.txt' in names
    assert 'bar.md' in names
    assert 'skip.txt' not in names 