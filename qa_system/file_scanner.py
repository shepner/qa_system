"""
@file: file_scanner.py
FileScanner module for scanning directories and files with inclusion/exclusion rules and hashing.

This module provides the FileScanner class, which is responsible for scanning a directory tree or a single file,
applying extension and exclusion rules, and computing file hashes for downstream processing (e.g., document embedding).

Key Features:
- Configurable document root, allowed extensions, and exclusion patterns
- Hashing with configurable algorithm
- Skips files based on extension and exclusion rules
- Returns file metadata (path, size, checksum)

Example usage:
    config = {
        'FILE_SCANNER': {
            'DOCUMENT_PATH': './docs',
            'ALLOWED_EXTENSIONS': ['md', 'txt', 'pdf'],
            'EXCLUDE_PATTERNS': ['*.tmp', '.*', '__pycache__'],
            'HASH_ALGORITHM': 'sha256',
            'SKIP_EXISTING': True,
        }
    }
    scanner = FileScanner(config)
    files = scanner.scan_files()
"""
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

    Args:
        config: Configuration object or dict. Must contain a 'FILE_SCANNER' section with keys:
            - DOCUMENT_PATH (str): Root directory to scan (default: './docs')
            - ALLOWED_EXTENSIONS (list[str]): File extensions to include (e.g., ['md', 'txt'])
            - EXCLUDE_PATTERNS (list[str]): Patterns to exclude (e.g., ['*.tmp', '.*'])
            - HASH_ALGORITHM (str): Hash algorithm to use (default: 'sha256')
            - SKIP_EXISTING (bool): Whether to skip files that already exist (default: True)
    """
    def __init__(self, config: Any):
        """
        Initialize the FileScanner with configuration.
        """
        logger.info(f"Called FileScanner.__init__(config={config})")
        self.config = config.get_nested('FILE_SCANNER') if hasattr(config, 'get_nested') else config.get('FILE_SCANNER', {})
        self.document_path = Path(self.config.get('DOCUMENT_PATH', './docs')).resolve()
        self.allowed_extensions = set(self.config.get('ALLOWED_EXTENSIONS', []))
        self.exclude_patterns = self.config.get('EXCLUDE_PATTERNS', [])
        self.hash_algorithm = self.config.get('HASH_ALGORITHM', 'sha256')
        self.skip_existing = self.config.get('SKIP_EXISTING', True)

    def scan_files(self, path: Optional[str] = None, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scan for files to process, applying extension and exclusion rules.

        Args:
            path: Optional override for root directory or file to scan. If None, uses self.document_path.
            pattern: Optional glob pattern to match files within a directory (e.g., '*.png').
        Returns:
            List of dicts with file metadata (path, hash, size, etc.)
        Raises:
            ValidationError: If the path is invalid or inaccessible.
        """
        logger.debug(f"Called FileScanner.scan_files(path={path}, pattern={pattern})")
        root = Path(path) if path else self.document_path
        found_files = []
        if not root.exists():
            logger.error(f"Scan path does not exist: {root}")
            raise ValidationError(f"Scan path does not exist: {root}")
        if root.is_file():
            if not self._is_allowed(root):
                logger.info(f"File not allowed by extension: {root}")
                return []
            if self._is_excluded(root):
                logger.info(f"File excluded by pattern: {root}")
                return []
            file_info = {
                'path': str(root.resolve()),
                'size': root.stat().st_size,
                'checksum': self._compute_hash(root),
            }
            found_files.append(file_info)
        elif root.is_dir():
            if pattern:
                files_iter = root.rglob(pattern)
            else:
                files_iter = root.rglob('*')
            for file_path in files_iter:
                if not file_path.is_file():
                    continue
                if not self._is_allowed(file_path):
                    continue
                if self._is_excluded(file_path):
                    continue
                file_info = {
                    'path': str(file_path.resolve()),
                    'size': file_path.stat().st_size,
                    'checksum': self._compute_hash(file_path),
                }
                found_files.append(file_info)
        else:
            logger.error(f"Scan path is neither a file nor a directory: {root}")
            raise ValidationError(f"Scan path is neither a file nor a directory: {root}")
        logger.info(f"Scanned {root}: found {len(found_files)} files for processing.")
        return found_files

    def _is_allowed(self, file_path: Path) -> bool:
        """
        Check if the file extension is allowed.

        Args:
            file_path: Path object for the file.
        Returns:
            True if the file extension is in the allowed list, False otherwise.
        """
        logger.debug(f"Called FileScanner._is_allowed(file_path={file_path})")
        return file_path.suffix.lstrip('.').lower() in self.allowed_extensions

    def _is_excluded(self, file_path: Path) -> bool:
        """
        Check if the file matches any exclusion pattern.

        Args:
            file_path: Path object for the file.
        Returns:
            True if the file matches any exclusion pattern, False otherwise.
        """
        logger.debug(f"Called FileScanner._is_excluded(file_path={file_path})")
        try:
            rel_path = str(file_path.resolve().relative_to(self.document_path))
        except ValueError:
            logger.warning(f"File {file_path} is not under document_path {self.document_path}")
            return False
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
        """
        Compute the hash of a file using the configured hash algorithm.

        Args:
            file_path: Path object for the file.
        Returns:
            Hex digest of the file's hash.
        Raises:
            QASystemError: If the hash algorithm is unsupported.
        """
        logger.debug(f"Called FileScanner._compute_hash(file_path={file_path})")
        hash_func = getattr(hashlib, self.hash_algorithm, None)
        if not hash_func:
            logger.error(f"Unsupported hash algorithm: {self.hash_algorithm}")
            raise QASystemError(f"Unsupported hash algorithm: {self.hash_algorithm}")
        h = hash_func()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest() 