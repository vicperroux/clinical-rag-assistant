"""Embedding generation using Mistral API."""

import numpy as np
from typing import List
from mistralai import Mistral


class EmbeddingGenerator:
    """Generates embeddings using Mistral API."""

    def __init__(self, api_key: str, model: str = "mistral-embed"):
        """
        Initialize embedding generator.

        Args:
            api_key: Mistral API key
            model: Embedding model name
        """
        self.client = Mistral(api_key=api_key)
        self.model = model

    def get_text_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for API calls

        Returns:
            NumPy array of embeddings
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(
                model=self.model,
                inputs=batch
            )
            all_vecs.extend([d.embedding for d in resp.data])
        return np.array(all_vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            NumPy array embedding
        """
        return self.get_text_embeddings([query])