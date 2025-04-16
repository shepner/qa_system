"""
Document processor for handling, validating, and preparing documents for embedding.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import hashlib
from datetime import datetime, timedelta
import magic
import re
import os
import logging
import asyncio
from .config import Config
from .embeddings import EmbeddingModel


class DocumentProcessor:
    def __init__(self, config: Config):
        """Initialize document processor with configuration."""
        self.config = config.get_nested("DOCUMENT_PROCESSING")
        if self.config is None:
            raise ValueError("DOCUMENT_PROCESSING configuration section is required")
        self.mime = magic.Magic(mime=True)
        self.logger = logging.getLogger(__name__)
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(config)
        # Store workspace root for relative path calculation
        self.workspace_root = Path(self.config.get("WORKSPACE_ROOT", os.getcwd()))

    def is_valid_file(self, file_path: Path) -> bool:
        """
        Validate if a file should be processed based on configured rules.
        """
        if not file_path.is_file():
            self.logger.debug(f"Skipping {file_path}: Not a file")
            return False

        extension = file_path.suffix.lower().strip()
        allowed_extensions = [
            (ext.lower().strip() if ext.startswith('.') else f'.{ext.lower().strip()}')
            for ext in self.config["ALLOWED_EXTENSIONS"]
        ]
        
        if extension not in allowed_extensions:
            self.logger.debug(f"Skipping {file_path}: Extension not allowed")
            return False

        # Check include patterns (these override exclude patterns)
        included = False
        for pattern in self.config["INCLUDE_PATTERNS"]:
            if Path(file_path).match(pattern):
                included = True
                break

        if not included:
            # Check exclude patterns
            for pattern in self.config["EXCLUDE_PATTERNS"]:
                if Path(file_path).match(pattern):
                    self.logger.debug(f"Skipping {file_path}: Matches exclude pattern")
                    return False

        return True

    def get_document_hash(self, content: str) -> str:
        """Generate document hash using configured algorithm."""
        algo = self.config["UPDATE_HANDLING"]["HASH_ALGORITHM"].lower()
        hasher = hashlib.new(algo)
        hasher.update(content.encode())
        return hasher.hexdigest()

    def get_document_metadata(self, file_path: Path, content: str, chunk_info: Dict[str, int]) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a document as specified in architecture.
        """
        try:
            relative_path = file_path.relative_to(self.workspace_root)
        except ValueError:
            relative_path = file_path

        checksum = hashlib.sha256(content.encode()).hexdigest()
        stats = file_path.stat()
        
        return {
            "path": str(file_path),
            "relative_path": str(relative_path),
            "directory": str(file_path.parent),
            "filename_full": file_path.name,
            "filename_stem": file_path.stem,
            "file_type": file_path.suffix.lower().strip('.'),
            "created_at": datetime.fromtimestamp(stats.st_ctime),
            "last_modified": datetime.fromtimestamp(stats.st_mtime),
            "chunk_count": chunk_info["chunk_count"],
            "total_tokens": chunk_info["total_tokens"],
            "checksum": checksum,
            "mime_type": self.mime.from_file(str(file_path)),
            "size_bytes": stats.st_size
        }

    async def process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single document and return its metadata, content, and embedding.
        """
        self.logger.info(f"Processing document: {file_path}")
        
        if not self.is_valid_file(file_path):
            return None

        try:
            content = file_path.read_text()
            chunk_info = await self.embedding_model.get_chunk_info(content)
            embedding = await self.embedding_model.generate_embeddings([content])
            metadata = self.get_document_metadata(file_path, content, chunk_info)
            
            result = {
                **metadata,
                "content": content,
                "embedding": embedding[0] if embedding else None
            }
            
            self.logger.info(f"Successfully processed {file_path}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    async def process_directory(self, dir_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Process all valid documents in a directory and generate their embeddings.
        """
        if not dir_path.is_dir():
            error_msg = f"Path {dir_path} is not a directory"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Processing directory: {dir_path}")
        processed_count = 0
        error_count = 0

        for file_path in dir_path.rglob("*"):
            if result := await self.process_document(file_path):
                processed_count += 1
                yield result
            else:
                error_count += 1

        self.logger.info(f"Directory processing complete. Processed: {processed_count}, Errors: {error_count}") 