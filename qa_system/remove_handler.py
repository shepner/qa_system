"""
@file: remove_handler.py
RemoveHandler module for safe deletion of files and their associated data from the system.

This module provides the RemoveHandler class, which implements pattern matching, batch removal,
verification, and error handling for document deletion from the vector store.

Classes:
    RemoveHandler: Handles finding and removing documents from the vector store based on patterns or paths.

Example usage:
    handler = RemoveHandler(config)
    result = handler.remove_documents(pattern="*.md")

Raises:
    RemovalError: If document removal fails unexpectedly.
    ValidationError: If input validation fails.
    DocumentNotFoundError: If no matching documents are found.
    VectorStoreError: For vector store operation errors.
"""

import logging
from typing import List, Union, Dict, Any
import os
import fnmatch
from qa_system.vector_store import ChromaVectorStore
from qa_system.exceptions import (
    DocumentNotFoundError, RemovalError, ValidationError, VectorStoreError
)

logger = logging.getLogger(__name__)

class RemoveHandler:
    """
    Handles safe removal of documents and their associated data from the vector store.

    Supports pattern matching, batch removal, verification, and error handling.
    """
    def __init__(self, config):
        """
        Initialize RemoveHandler with configuration.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.vector_store = ChromaVectorStore(config)
        remover_config = config.get_nested('DATA_REMOVER', {})
        self.recursive = remover_config.get('RECURSIVE', True)
        self.case_sensitive = remover_config.get('CASE_SENSITIVE', False)
        self.require_confirmation = remover_config.get('REQUIRE_CONFIRMATION', True)
        self.batch_size = remover_config.get('BATCH_SIZE', 20)
        self.verify_removal_flag = remover_config.get('VERIFY_REMOVAL', True)

    def find_matches(self, pattern: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Find documents matching the given pattern(s) in the vector store.

        Args:
            pattern: A string or list of strings representing file path patterns or globs.

        Returns:
            List of document metadata dicts matching the pattern(s).
        """
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        all_docs = self.vector_store.list_metadata()
        matches = []
        for pat in patterns:
            # Only normalize to absolute path if not a glob
            if any(c in pat for c in ['*', '?', '[']):
                pat_abs = pat
            else:
                pat_abs = os.path.abspath(os.path.expanduser(pat))
            for doc in all_docs:
                doc_path = doc.get('path', '')
                doc_path_abs = os.path.abspath(os.path.expanduser(doc_path))
                # Match absolute paths directly, or fallback to fnmatch for globs
                if pat_abs == doc_path_abs or fnmatch.fnmatch(doc_path_abs, pat_abs):
                    matches.append(doc)
        return matches

    def remove_documents(
        self,
        pattern: Union[str, List[str]] = None,
        paths: Union[str, List[str]] = None,
        recursive: bool = None,
        verify_removal: bool = None,
        require_confirmation: bool = None
    ) -> Dict[str, Any]:
        """
        Remove documents from the vector store matching the given pattern or paths.

        Args:
            pattern: File path pattern(s) or glob(s) to match for removal.
            paths: Alternative to pattern; direct file path(s) to remove.
            recursive: Whether to remove recursively (default from config).
            verify_removal: Whether to verify removal after deletion (default from config).
            require_confirmation: Whether to require confirmation before deletion (default from config).

        Returns:
            Dictionary with lists of 'removed', 'failed', 'not_found', and 'errors'.

        Raises:
            RemovalError: If an unexpected error occurs during removal.
            ValidationError: If input validation fails.
        """
        logger.info(f"Entered remove_documents with pattern={pattern}, paths={paths}, recursive={recursive}, verify_removal={verify_removal}, require_confirmation={require_confirmation}")
        # Normalize input: if pattern is None and paths is provided, use paths as pattern
        if pattern is None and paths is not None:
            if isinstance(paths, str):
                pattern = [paths]
            elif isinstance(paths, list):
                pattern = paths
            else:
                logger.warning("'paths' argument is not a string or list.")
                return {'removed': [], 'failed': [], 'not_found': [], 'errors': ["Invalid 'paths' argument type"]}
        elif pattern is not None:
            if isinstance(pattern, str):
                pattern = [pattern]
            elif not isinstance(pattern, list):
                logger.warning("'pattern' argument is not a string or list.")
                return {'removed': [], 'failed': [], 'not_found': [], 'errors': ["Invalid 'pattern' argument type"]}
        else:
            logger.warning("Both 'pattern' and 'paths' are None. Nothing to remove.")
            return {'removed': [], 'failed': [], 'not_found': [], 'errors': ["No pattern or paths provided"]}

        recursive = self.recursive if recursive is None else recursive
        verify_removal = self.verify_removal_flag if verify_removal is None else verify_removal
        require_confirmation = self.require_confirmation if require_confirmation is None else require_confirmation
        result = {'removed': [], 'failed': [], 'not_found': [], 'errors': []}
        try:
            matches = self.find_matches(pattern)
            logger.debug(f"After find_matches: found {len(matches)} matches for pattern {pattern}: {[doc.get('id') for doc in matches]}")
            if not matches:
                result['not_found'].append(pattern)
                return result
            # Collect all matching ids
            ids = [doc.get('id') for doc in matches if doc.get('id')]
            logger.debug(f"IDs to be deleted: {ids}")
            if ids:
                logger.info(f"Deleting ids: {ids}")
                self.vector_store.delete(ids=ids, require_confirmation=require_confirmation)
                if verify_removal:
                    still_exists = self.find_matches([doc.get('path') for doc in matches])
                    if still_exists:
                        result['failed'].extend([doc.get('path') for doc in still_exists])
                    else:
                        result['removed'].extend([doc.get('path') for doc in matches])
                else:
                    result['removed'].extend([doc.get('path') for doc in matches])
            else:
                logger.error("No valid ids found for deletion.")
                result['failed'].extend([doc.get('path') for doc in matches])
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            result['errors'].append({'exception': str(e), 'type': 'ValidationError'})
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            result['errors'].append({'exception': str(e), 'type': 'Unknown'})
            raise RemovalError(f"Failed to remove documents: {e}")
        return result

    def verify_removal(self, paths: List[str]) -> bool:
        """
        Verify that the given paths are no longer present in the vector store.

        Args:
            paths: List of file paths to check.

        Returns:
            True if none of the paths are present, False otherwise.
        """
        matches = self.find_matches(paths)
        return len(matches) == 0

    def cleanup_failed_removal(self, doc_id: str):
        """
        Attempt to clean up after a failed removal.

        Args:
            doc_id: Document ID to attempt to remove from the vector store.
        """
        try:
            self.vector_store.delete({'id': doc_id}, require_confirmation=False)
        except Exception as e:
            logger.error(f"Cleanup failed for {doc_id}: {e}") 