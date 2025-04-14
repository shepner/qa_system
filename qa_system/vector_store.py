"""
Vector store implementation for efficient document storage and retrieval.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store with configuration."""
        self.config = config
        self._client = None
        self._collection = None
        self.setup()

    def setup(self) -> None:
        """Set up the vector store client and collection."""
        persist_dir = Path(self.config["PERSIST_DIRECTORY"])
        persist_dir.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir),
            anonymized_telemetry=False
        )

        self._client = chromadb.Client(settings)
        self._collection = self._client.get_or_create_collection(
            name=self.config["COLLECTION_NAME"],
            metadata={"hnsw:space": self.config["DISTANCE_METRIC"]}
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents and their embeddings to the vector store."""
        batch_size = self.config["PERFORMANCE"]["BATCH_SIZE"]
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self._collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end] if metadatas else None
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using query embedding."""
        top_k = top_k or self.config["TOP_K"]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        self._collection.delete(ids=ids)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": len(self._collection.get()["ids"]),
            "collection_name": self.config["COLLECTION_NAME"],
            "distance_metric": self.config["DISTANCE_METRIC"]
        } 