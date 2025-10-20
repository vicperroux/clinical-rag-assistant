# Clinical RAG Assistant

A Retrieval-Augmented Generation (RAG) system for clinical drug information using DailyMed data.

## The Challenge

- Clinicians and pharmacists handle **20-50 drug label questions each day**
- FDA labels are **voluminous (150k+ SPL documents)**, messy, and frequently revised, making it difficult to research dosage, contraindications, or interactions within a **<5-minute clinical window**
- In practice, providers either **skip the lookup** (risking errors) or **lose valuable time** searching through dense documentation

## The Goal
Provide a **fast, reliable, and traceable** answer to any drug-related query, grounded in official FDA drug labels. This system enables clinicians to get accurate information in seconds instead of minutes, improving both patient safety and workflow efficiency.

## Project Ambition

This project provides a framework for building a comprehensive clinical drug information retrieval system. While the examples demonstrate collecting data for individual drugs (aspirin, ibuprofen, etc.), **the system is designed to scale to the entire DailyMed database**. DailyMed offers [bulk XML downloads](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) of all currently approved drug labels, enabling you to download and process tens of thousands of drug labels at once. This allows you to create a knowledge base covering virtually all FDA-approved medications with their complete prescribing information, warnings, interactions, and clinical guidance.

## Overview

This system provides accurate, source-backed answers to clinical questions about medications by:
- Fetching drug data from RxNorm and DailyMed APIs
- Processing drug labels into searchable fragments
- Using hybrid search (dense + sparse) for retrieval
- Generating responses with proper source attribution

## Repository Structure

```
clinical-rag-assistant/
├── src/
│   ├── data/              # Data processing and APIs
│   │   ├── rxnorm.py      # RxNorm API integration
│   │   ├── dailymed.py    # DailyMed API and XML parsing
│   │   └── processing.py  # Text processing and fragmentation
│   ├── retrieval/         # Search and retrieval
│   │   ├── embeddings.py  # Embedding generation (Mistral)
│   │   ├── vector_store.py # FAISS vector store
│   │   └── search.py      # Hybrid search implementation
│   └── models/
│       └── types.py       # Data models (Fragment, RetrievalResult)
├── notebooks/             # Jupyter notebooks
├── data/                  # Local data storage
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd clinical-rag-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```
- Mistral API key for embeddings and LLM generation (required)
- Cohere API key for reranking (optional)

## Quick Start

### Command Line Interface

Run the complete pipeline:
```bash
python main.py full-pipeline --drugs "aspirin"
```

Or run individual steps:
```bash
python main.py collect --drugs "aspirin" # Can be replaced with a bulk download of drugs from DailyMed
python main.py process
python main.py index
python main.py search --query "What are the side effects of aspirin?"
```

### Web Interface

Launch the medical web interface:
```bash
python app.py
```
Then navigate to `http://localhost:7860` for an interactive interface with:
- LLM-powered clinical answers with source citations
- AI reranking for improved precision
- DailyMed source linking
- Medical-focused UI

### Programmatic Usage

### 1. Data Collection

```python
from src.data.rxnorm import get_rxcui_mapping
from src.data.dailymed import dailymed_setids_for_rxcui, fetch_spl_xml

# Get RxNorm IDs for drug names
drug_names = ["aspirin", "ibuprofen", "acetaminophen"]
rxcui_map = get_rxcui_mapping(drug_names)

# Fetch DailyMed data
for drug, rxcui in rxcui_map.items():
    if rxcui:
        setids = dailymed_setids_for_rxcui(rxcui)
        # Download and save XML files...
```

### 2. Processing and Indexing

```python
from src.data.processing import make_fragments_from_json_label
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore

# Process XML files into fragments
fragments = []
for json_data in parsed_xml_files:
    frags = make_fragments_from_json_label(json_data)
    fragments.extend(frags)

# Create embeddings and vector store
embedder = EmbeddingGenerator(api_key="your-mistral-key")
embeddings = embedder.get_text_embeddings([f.text for f in fragments])

vector_store = VectorStore()
vector_store.add_fragments(fragments, embeddings)
```

### 3. Search and Retrieval

```python
from src.retrieval.search import HybridSearcher

# Initialize hybrid searcher with optional reranking
searcher = HybridSearcher(
    vector_store,
    embedder,
    cohere_api_key="your-cohere-key"  # Optional for reranking
)

# Search with different retrieval modes
results = searcher.retrieve(
    query="What are the side effects of aspirin?",
    retrieval_type="hybrid",  # "dense", "sparse", or "hybrid"
    rerank=True,              # Enable AI reranking for better precision
    k_final=8
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.fragment.drug_name}")
    print(f"Text: {result.fragment.text[:200]}...")
```


## Features

- **Multi-source data**: Integrates RxNorm and DailyMed APIs
- **Hybrid search**: Combines dense (FAISS) and sparse (BM25) retrieval with RRF fusion
- **AI Reranking**: Optional Cohere reranking for improved precision
- **LLM-powered answers**: Mistral-based answer generation with source citations
- **Paragraph-aware chunking**: Smart text splitting preserving semantic coherence
- **Web interface**: Medical-focused UI with Gradio
- **Flexible processing**: Configurable retrieval modes and parameters
- **Data export**: Parquet export, source attribution, DailyMed linking

## API Requirements

- **RxNorm API**: Free, no key required
- **DailyMed API**: Free, no key required
- **Mistral API**: Required for embeddings and LLM generation ([Get API key](https://console.mistral.ai/))
- **Cohere API**: Optional for reranking ([Get API key](https://dashboard.cohere.com/))

## Scaling and Optimization

### Performance Tips

- Enable reranking with Cohere for improved precision
- Use hybrid retrieval mode for best recall/precision balance
- Adjust `k_final` based on your quality/latency requirements

### Scaling to Full DailyMed Database

To process the complete DailyMed database:

1. Download bulk XML files from [DailyMed Resources](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm)
2. Extract to `data/xml_files/`
3. Run processing and indexing as normal
4. The system handles tens of thousands of drug labels

### Security Notes

- Store API keys in environment variables
- Never commit `.env` file to version control
- `.env` is already in `.gitignore`

### Medical Disclaimer

**This system is for informational purposes only.** It provides information extracted from FDA drug labels but should not replace professional medical judgment. Always consult current prescribing information and clinical guidelines for patient care decisions.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
