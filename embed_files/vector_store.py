"""
Vector store implementation for document embeddings and metadata.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from .config import Configuration


class VectorStore:
    def __init__(self, config: Configuration):
        """Initialize vector store with configuration."""
        self.config = config.get_nested("VECTOR_STORE")
        if self.config is None:
            raise ValueError("VECTOR_STORE configuration section is required")
        
        self.logger = logging.getLogger(__name__)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load_data()

    def load_data(self) -> None:
        """Load existing vector store data from disk."""
        store_path = Path(self.config["STORE_PATH"])
        if not store_path.exists():
            self.logger.info(f"Creating new vector store at {store_path}")
            store_path.mkdir(parents=True, exist_ok=True)
            return

        metadata_file = store_path / "metadata.json"
        embeddings_file = store_path / "embeddings.npz"

        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self.metadata)} documents")

            if embeddings_file.exists():
                data = np.load(embeddings_file)
                self.embeddings = {k: data[k] for k in data.files}
                self.logger.info(f"Loaded embeddings for {len(self.embeddings)} documents")

        except Exception as e:
            self.logger.error(f"Error loading vector store data: {str(e)}")
            raise

    def save_data(self) -> None:
        """Save vector store data to disk."""
        store_path = Path(self.config["STORE_PATH"])
        store_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save metadata
            with open(store_path / "metadata.json", 'w') as f:
                json.dump(self.metadata, f, default=str)

            # Save embeddings
            if self.embeddings:
                np.savez(store_path / "embeddings.npz", **self.embeddings)

            self.logger.info(f"Saved vector store data: {len(self.metadata)} documents")

        except Exception as e:
            self.logger.error(f"Error saving vector store data: {str(e)}")
            raise

    def add_document(self, doc_data: Dict[str, Any]) -> None:
        """
        Add or update a document in the vector store.
        """
        doc_path = doc_data["path"]
        
        try:
            # Extract and validate embedding
            embedding = doc_data.pop("embedding")
            if embedding is None:
                self.logger.warning(f"Skipping document {doc_path}: No embedding available")
                return

            # Store embedding and metadata
            self.embeddings[doc_path] = np.array(embedding)
            self.metadata[doc_path] = doc_data
            
            self.logger.info(f"Added document to vector store: {doc_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding document {doc_path}: {str(e)}")
            raise

    def remove_document(self, doc_path: str) -> None:
        """Remove a document from the vector store."""
        try:
            self.embeddings.pop(doc_path, None)
            self.metadata.pop(doc_path, None)
            self.logger.info(f"Removed document from vector store: {doc_path}")
        except Exception as e:
            self.logger.error(f"Error removing document {doc_path}: {str(e)}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        """
        if not self.embeddings:
            self.logger.warning("No documents in vector store")
            return []

        try:
            # Convert query to numpy array
            query_vector = np.array(query_embedding)

            # Calculate similarities
            similarities = {}
            for doc_path, doc_embedding in self.embeddings.items():
                similarity = np.dot(query_vector, doc_embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_embedding)
                )
                similarities[doc_path] = float(similarity)

            # Get top k results
            top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            results = []
            for doc_path, similarity in top_results:
                result = {
                    "similarity": similarity,
                    **self.metadata[doc_path]
                }
                results.append(result)

            self.logger.info(f"Search completed: Found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise

    def get_document_metadata(self, doc_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific document."""
        return self.metadata.get(doc_path)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents with their metadata."""
        return [self.metadata[doc_path] for doc_path in self.metadata] 