"""Data models and types for the clinical RAG assistant."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class Fragment:
    """
    Represents a chunk of a drug label.
    """
    set_id: str
    version: str
    effective_date: str
    rxcui: Optional[str]            # unique id in RxNorm standard
    ingredient_name: Optional[str]   # ingredient
    drug_name: Optional[str]        # manufactured name
    product_type: Optional[str]     # human/animal & OTC/Prescription
    section_code: str               # for instance "34067-9"
    section_title: str              # normalized title
    section_rank: int               # order within the label
    fragment_id: str                # deterministic local id within section
    path: str                       # canonical path for audit/filters
    text: str                       # actual text content


@dataclass
class RetrievalResult:
    """
    Represents a search result from the vector database.
    """
    idx: int            # index in the fragment list
    score: float
    fragment: Fragment
    rank: int = 0       # rank in final results (optional, set after sorting)


@dataclass
class QueryResult:
    """
    Represents the complete result of a query including retrieved fragments and generated response.
    """
    query: str
    retrieved_fragments: List[RetrievalResult]
    response: str
    retrieval_time: float
    generation_time: float