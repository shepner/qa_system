import unittest
from pathlib import Path
import tempfile
import os
import time
from datetime import datetime, timedelta
from qa_system.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "DOCUMENT_PROCESSING": {
                "ALLOWED_EXTENSIONS": [".txt", ".md"],
                "EXCLUDE_PATTERNS": [".*\\.tmp$"],
                "INCLUDE_PATTERNS": ["important.*\\.txt$"],
                "FILE_SIZE_LIMITS": {
                    "MIN_BYTES": 10,
                    "MAX_BYTES": 1024 * 1024  # 1MB
                },
                "DOCUMENT_AGE": {
                    "MAX_DAYS": 30,
                    "MIN_MINUTES": 5
                },
                "CONTENT_VALIDATION": {
                    "MIN_CHARS": 10,
                    "MAX_CHARS": 1000000,
                    "SKIP_BINARY": True
                }
            }
        }
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DocumentProcessor(self.test_config)

    def create_test_file(self, filename, content, mtime=None):
        path = Path(self.temp_dir) / filename
        path.write_text(content)
        if mtime:
            os.utime(path, (mtime.timestamp(), mtime.timestamp()))
        return path

    def test_valid_file_extension(self):
        path = self.create_test_file("test.txt", "Valid content")
        self.assertTrue(self.processor.is_valid_file(path))
        
        path = self.create_test_file("test.invalid", "Invalid extension")
        self.assertFalse(self.processor.is_valid_file(path))

    def test_file_size_limits(self):
        # Test file too small
        path = self.create_test_file("small.txt", "tiny")
        self.assertFalse(self.processor.is_valid_file(path))

        # Test valid size
        path = self.create_test_file("valid.txt", "x" * 100)
        self.assertTrue(self.processor.is_valid_file(path))

    def test_document_age(self):
        # Test too new
        now = datetime.now()
        path = self.create_test_file("new.txt", "New content", now)
        self.assertFalse(self.processor.is_valid_file(path))

        # Test valid age
        old_time = now - timedelta(minutes=10)
        path = self.create_test_file("old.txt", "Old content", old_time)
        self.assertTrue(self.processor.is_valid_file(path))

    def test_patterns(self):
        # Test exclude pattern
        path = self.create_test_file("test.tmp", "Temporary file")
        self.assertFalse(self.processor.is_valid_file(path))

        # Test include pattern overriding exclude
        path = self.create_test_file("important.txt", "Important content")
        self.assertTrue(self.processor.is_valid_file(path))

    def test_process_document(self):
        content = "Test document content"
        path = self.create_test_file("test.txt", content)
        time.sleep(5)  # Ensure file is old enough
        
        result = self.processor.process_document(path)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], content)
        self.assertIn("hash", result)
        self.assertIn("mime_type", result)
        self.assertIn("size", result)
        self.assertIn("mtime", result)

    def test_process_directory(self):
        # Create multiple test files
        self.create_test_file("valid1.txt", "Valid content 1")
        self.create_test_file("valid2.txt", "Valid content 2")
        self.create_test_file("invalid.tmp", "Invalid content")
        time.sleep(5)  # Ensure files are old enough

        results = list(self.processor.process_directory(self.temp_dir))
        self.assertEqual(len(results), 2)  # Only valid files should be processed

    def tearDown(self):
        # Clean up temporary directory
        for path in Path(self.temp_dir).glob("*"):
            path.unlink()
        Path(self.temp_dir).rmdir()

if __name__ == "__main__":
    unittest.main() 