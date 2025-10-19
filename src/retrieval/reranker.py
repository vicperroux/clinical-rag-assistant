"""Reranking functionality using Cohere API."""

from typing import List, Optional
from ..models.types import RetrievalResult

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


def _doc_text_enriched(r: RetrievalResult, max_chars: int) -> str:
    """
    Create enriched document text for reranker with metadata prefix.

    Args:
        r: RetrievalResult to enrich
        max_chars: Maximum characters from text body

    Returns:
        Enriched text with drug/ingredient/section metadata
    """
    f = r.fragment
    drug = f.drug_name or ""
    ingredient = f.ingredient_name or ""
    section = f"{f.section_code} {f.section_title}".strip()
    body = (f.text or "")[:max_chars]
    # Adding prefix the reranker can learn from
    return f"[DRUG: {drug}] [INGREDIENT: {ingredient}] [SECTION: {section}] {body}"


class CohereReranker:
    """Reranks search results using Cohere's rerank API."""

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Reranking model name
        """
        if not HAS_COHERE:
            raise ImportError("cohere package not installed. Install with: pip install cohere")

        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 8,
        max_chars: int = 1200
    ) -> List[RetrievalResult]:
        """
        Rerank search results using Cohere cross-encoder model.

        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of top results to return
            max_chars: Maximum characters from each document

        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return []

        # Prepare documents with enriched text
        docs = [{"text": _doc_text_enriched(r, max_chars)} for r in results]

        # Call Cohere rerank API
        resp = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs))
        )

        # Extract reranked results with new scores
        reranked = []
        for item in resp.results:
            idx_in_batch = item.index
            rr = results[idx_in_batch]
            # Update score with relevance score from reranker
            rr.score = float(item.relevance_score)
            reranked.append(rr)

        # Sort by score (should already be sorted but ensure it)
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Update rank field
        for rank, result in enumerate(reranked[:top_k]):
            result.rank = rank

        return reranked[:top_k]


def llm_reranker(
    query: str,
    results: List[RetrievalResult],
    cohere_api_key: Optional[str] = None,
    top_k: int = 8,
    model: str = "rerank-v3.5",
    max_chars: int = 1200
) -> List[RetrievalResult]:
    """
    Standalone function for reranking (matches notebook interface).

    Args:
        query: Search query
        results: Initial retrieval results
        cohere_api_key: Cohere API key
        top_k: Number of top results to return
        model: Reranking model name
        max_chars: Maximum characters from each document

    Returns:
        Reranked list of RetrievalResult objects
    """
    if not cohere_api_key:
        raise ValueError("Cohere API key required for reranking")

    reranker = CohereReranker(api_key=cohere_api_key, model=model)
    return reranker.rerank(query, results, top_k, max_chars)
