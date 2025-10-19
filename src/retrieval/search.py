"""Hybrid search combining dense and sparse retrieval."""

import re
import numpy as np
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from ..models.types import Fragment, RetrievalResult
from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .reranker import CohereReranker


# Initialize NLTK components
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

stemmer = PorterStemmer()
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)?")  # keep words, numbers, simple hyphens


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


class HybridSearcher:
    """Hybrid search combining dense (FAISS) and sparse (BM25) retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        cohere_api_key: Optional[str] = None,
        rerank_model: str = "rerank-v3.5"
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_store: FAISS vector store
            embedding_generator: Embedding generator
            cohere_api_key: Optional Cohere API key for reranking
            rerank_model: Reranking model name
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        # Use BM25 index from vector store if available, otherwise build it
        if hasattr(vector_store, 'bm25_index') and vector_store.bm25_index is not None:
            self.bm25_index = vector_store.bm25_index
        else:
            self._build_bm25_index()

        # Initialize reranker if API key provided
        self.reranker = None
        if cohere_api_key:
            try:
                self.reranker = CohereReranker(api_key=cohere_api_key, model=rerank_model)
            except ImportError:
                print("⚠️  Cohere not installed - reranking disabled. Install with: pip install cohere")

    def _build_bm25_index(self):
        """Build BM25 index from fragments (fallback if not persisted)."""
        if not self.vector_store.fragments:
            return

        texts = [f.text for f in self.vector_store.fragments]
        tokenized_texts = [tokenize(text) for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)

    def dense_search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform dense vector search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Tuple of (scores, indices)
        """
        query_embedding = self.embedding_generator.embed_query(query)
        return self.vector_store.search(query_embedding, k)

    def sparse_search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sparse BM25 search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Tuple of (scores, indices)
        """
        if self.bm25_index is None:
            return np.array([]), np.array([])

        query_tokens = tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        indices = np.argsort(-scores)[:k]
        return scores[indices], indices

    def rrf_fuse(self, id_lists: List[np.ndarray], k_final: int = 8, k_const: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse multiple ranking lists using Reciprocal Rank Fusion.

        Args:
            id_lists: List of ranking arrays
            k_final: Final number of results
            k_const: RRF constant

        Returns:
            Tuple of (fused_scores, fused_indices)
        """
        rrf_scores = {}

        for id_list in id_lists:
            for rank, doc_id in enumerate(id_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (k_const + rank + 1)

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_ids = np.array([doc_id for doc_id, _ in sorted_docs[:k_final]])
        final_scores = np.array([score for _, score in sorted_docs[:k_final]])

        return final_scores, final_ids

    def retrieve_dense(self, query: str, k_dense: int = 12, k_final: int = 8) -> List[RetrievalResult]:
        """
        Perform dense-only retrieval (matching notebook).

        Args:
            query: Search query
            k_dense: Number of candidates to retrieve
            k_final: Number of final results

        Returns:
            List of RetrievalResult objects
        """
        scores, indices = self.dense_search(query, k_dense)
        indices = indices[:k_final]
        scores = scores[:k_final]

        fragments = self.vector_store.get_fragments(indices)
        results = []

        for idx, score, fragment in zip(indices, scores, fragments):
            result = RetrievalResult(
                idx=int(idx),
                score=float(score),
                fragment=fragment,
                rank=0
            )
            results.append(result)

        return results

    def retrieve_sparse(self, query: str, k_bm25: int = 12, k_final: int = 8) -> List[RetrievalResult]:
        """
        Perform sparse-only retrieval (matching notebook).

        Args:
            query: Search query
            k_bm25: Number of candidates to retrieve
            k_final: Number of final results

        Returns:
            List of RetrievalResult objects
        """
        scores, indices = self.sparse_search(query, k_bm25)
        indices = indices[:k_final]
        scores = scores[:k_final]

        fragments = self.vector_store.get_fragments(indices)
        results = []

        for idx, score, fragment in zip(indices, scores, fragments):
            result = RetrievalResult(
                idx=int(idx),
                score=float(score),
                fragment=fragment,
                rank=0
            )
            results.append(result)

        return results

    def retrieve_hybrid(self, query: str, k_dense: int = 12, k_bm25: int = 12, k_final: int = 8) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with RRF fusion (matching notebook).

        Args:
            query: Search query
            k_dense: Number of dense candidates
            k_bm25: Number of sparse candidates
            k_final: Number of final results

        Returns:
            List of RetrievalResult objects
        """
        # Get results from both methods
        _, dense_indices = self.dense_search(query, k_dense)
        _, sparse_indices = self.sparse_search(query, k_bm25)

        # Fuse using RRF
        rrf_scores, fused_indices = self.rrf_fuse([dense_indices, sparse_indices], k_final=k_final, k_const=60)

        # Get fragments and create results
        fragments = self.vector_store.get_fragments(fused_indices)
        results = []

        for idx, score, fragment in zip(fused_indices, rrf_scores, fragments):
            result = RetrievalResult(
                idx=int(idx),
                score=float(score),
                fragment=fragment,
                rank=0
            )
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        retrieval_type: str = "hybrid",
        rerank: bool = False,
        k_dense: int = 12,
        k_bm25: int = 12,
        k_final: int = 8
    ) -> List[RetrievalResult]:
        """
        Complete retrieval function matching notebook interface.

        Args:
            query: Search query
            retrieval_type: "dense" | "sparse" | "hybrid"
            rerank: Whether to apply reranking
            k_dense: Number of dense candidates
            k_bm25: Number of sparse candidates
            k_final: Number of final results

        Returns:
            List of RetrievalResult objects
        """
        retrieval_type = (retrieval_type or "hybrid").lower()

        # Initial retrieval
        if retrieval_type == "dense":
            initial = self.retrieve_dense(query, k_dense, k_final)
        elif retrieval_type == "sparse":
            initial = self.retrieve_sparse(query, k_bm25, k_final)
        else:  # hybrid
            initial = self.retrieve_hybrid(query, k_dense, k_bm25, k_final)

        # Reranking
        if rerank and self.reranker:
            reranked = self.reranker.rerank(query, initial, top_k=k_final)
            return reranked
        elif rerank and not self.reranker:
            print("⚠️  Reranking requested but reranker not initialized (missing Cohere API key)")

        # Update ranks
        for rank, result in enumerate(initial):
            result.rank = rank

        return initial

    def hybrid_search(self, query: str, k: int = 8, rerank: bool = False) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        Backward compatible method that calls retrieve().

        Args:
            query: Search query
            k: Number of final results
            rerank: Whether to apply reranking

        Returns:
            List of RetrievalResult objects
        """
        return self.retrieve(
            query=query,
            retrieval_type="hybrid",
            rerank=rerank,
            k_dense=k * 2,
            k_bm25=k * 2,
            k_final=k
        )