import os
import fnmatch
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from qa_system.exceptions import QASystemError, ValidationError

logger = logging.getLogger(__name__)

class FileScanner:
    """
    Scans directories for files to be embedded, applying inclusion/exclusion rules and hashing.
    """
    def __init__(self, config: Any):
        self.config = config.get_nested('FILE_SCANNER') if hasattr(config, 'get_nested') else config.get('FILE_SCANNER', {})
        self.document_path = Path(self.config.get('DOCUMENT_PATH', './docs'))
        self.allowed_extensions = set(self.config.get('ALLOWED_EXTENSIONS', []))
        self.exclude_patterns = self.config.get('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = self.config.get('HASH_ALGORITHM', 'sha256')
        self.skip_existing = self.config.get('SKIP_EXISTING', True)

    def scan_files(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scan for files to process, applying extension and exclusion rules.
        Args:
            path: Optional override for root directory to scan.
        Returns:
            List of dicts with file metadata (path, hash, size, etc.)
        Raises:
            ValidationError: If the path is invalid or inaccessible.
        """
        root = Path(path) if path else self.document_path
        if not root.exists() or not root.is_dir():
            logger.error(f"Scan path does not exist or is not a directory: {root}")
            raise ValidationError(f"Scan path does not exist or is not a directory: {root}")
        found_files = []
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            if not self._is_allowed(file_path):
                continue
            if self._is_excluded(file_path):
                continue
            file_info = {
                'path': str(file_path.resolve()),
                'size': file_path.stat().st_size,
                'hash': self._compute_hash(file_path),
            }
            found_files.append(file_info)
        logger.info(f"Scanned {root}: found {len(found_files)} files for processing.")
        return found_files

    def _is_allowed(self, file_path: Path) -> bool:
        return file_path.suffix.lstrip('.').lower() in self.allowed_extensions

    def _is_excluded(self, file_path: Path) -> bool:
        rel_path = str(file_path.relative_to(self.document_path))
        # Check all parts of the relative path for exclusion
        parts = rel_path.split(os.sep)
        for pattern in self.exclude_patterns:
            # Exclude if any part of the path matches the pattern
            if any(fnmatch.fnmatch(part, pattern) for part in parts):
                return True
            # Also check the full relative path and file name
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                return True
        return False

    def _compute_hash(self, file_path: Path) -> str:
        hash_func = getattr(hashlib, self.hash_algorithm, None)
        if not hash_func:
            logger.error(f"Unsupported hash algorithm: {self.hash_algorithm}")
            raise QASystemError(f"Unsupported hash algorithm: {self.hash_algorithm}")
        h = hash_func()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest() 