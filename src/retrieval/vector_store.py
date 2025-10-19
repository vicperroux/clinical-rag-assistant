"""FAISS vector store for dense retrieval."""

import faiss
import numpy as np
import pickle
import re
from typing import List, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from ..models.types import Fragment


# Initialize NLTK components for BM25
try:
    stop_words = set(stopwords.words("english"))
except:
    stop_words = set()

stemmer = PorterStemmer()
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text with preprocessing for BM25.

    Args:
        text: Input text

    Returns:
        List of processed tokens
    """
    # Lowercase
    text = text.lower()
    # Basic tokenization with hyphen/underscore handling
    tokens = TOKEN_RE.findall(text)
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Stem
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


class VectorStore:
    """FAISS-based vector store for fragment embeddings."""

    def __init__(self, dimension: int = 1024):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.fragments: List[Fragment] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_texts: Optional[List[List[str]]] = None

    def add_fragments(self, fragments: List[Fragment], embeddings: np.ndarray):
        """
        Add fragments and their embeddings to the store.

        Args:
            fragments: List of Fragment objects
            embeddings: NumPy array of embeddings (normalized)
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.fragments.extend(fragments)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        # Build BM25 index
        self._build_bm25_index()

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar fragments.

        Args:
            query_embedding: Query embedding vector (normalized)
            k: Number of results to return

        Returns:
            Tuple of (scores, indices)
        """
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)
        return scores[0], indices[0]

    def get_fragments(self, indices: np.ndarray) -> List[Fragment]:
        """
        Get fragments by indices.

        Args:
            indices: Array of fragment indices

        Returns:
            List of Fragment objects
        """
        return [self.fragments[i] for i in indices if i < len(self.fragments)]

    def _build_bm25_index(self):
        """Build BM25 index from fragments."""
        if not self.fragments:
            return

        texts = [f.text for f in self.fragments]
        self.tokenized_texts = [tokenize(text) for text in texts]
        self.bm25_index = BM25Okapi(self.tokenized_texts)

    def save(self, path: str):
        """
        Save vector store to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save fragments and embeddings
        with open(path / "fragments.pkl", "wb") as f:
            pickle.dump(self.fragments, f)

        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)

        # Save BM25 index and tokenized texts
        if self.bm25_index is not None:
            with open(path / "bm25_index.pkl", "wb") as f:
                pickle.dump(self.bm25_index, f)

        if self.tokenized_texts is not None:
            with open(path / "tokenized_texts.pkl", "wb") as f:
                pickle.dump(self.tokenized_texts, f)

    def load(self, path: str):
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Load fragments
        with open(path / "fragments.pkl", "rb") as f:
            self.fragments = pickle.load(f)

        # Load embeddings if they exist
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # Load BM25 index if it exists
        bm25_path = path / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)

        # Load tokenized texts if they exist
        tokenized_path = path / "tokenized_texts.pkl"
        if tokenized_path.exists():
            with open(tokenized_path, "rb") as f:
                self.tokenized_texts = pickle.load(f)

    def __len__(self) -> int:
        """Return number of fragments in store."""
        return len(self.fragments)
