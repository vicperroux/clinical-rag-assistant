#!/usr/bin/env python3
"""
Clinical RAG Assistant - Command Line Interface

This script provides a command-line interface for the complete RAG pipeline:
1. Data collection from RxNorm and DailyMed APIs
2. Processing and fragment creation
3. Embedding generation and indexing
4. Interactive search and querying
"""

import os
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from dotenv import load_dotenv

from src.data.rxnorm import get_rxcui_mapping
from src.data.dailymed import dailymed_setids_for_rxcui, fetch_spl_xml, parse_spl_xml
from src.data.processing import make_fragments_from_json_label
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.retrieval.search import HybridSearcher
from src.models.types import Fragment, QueryResult

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def to_md5(s: str) -> str:
    """Generate MD5 hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class ClinicalRAGSystem:
    """Main RAG system orchestrator."""

    def __init__(self, data_dir: str = "data", mistral_api_key: str = None, cohere_api_key: str = None):
        """
        Initialize the RAG system.

        Args:
            data_dir: Directory for storing data files
            mistral_api_key: Mistral API key for embeddings
            cohere_api_key: Optional Cohere API key for reranking
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize components
        self.embedding_generator = None
        self.vector_store = None
        self.searcher = None
        self.cohere_api_key = cohere_api_key

        if mistral_api_key:
            self.embedding_generator = EmbeddingGenerator(mistral_api_key)

        # Data storage paths
        self.xml_dir = self.data_dir / "xml_files"
        self.json_dir = self.data_dir / "json_files"
        self.index_dir = self.data_dir / "faiss_index"

        # Ensure directories exist
        self.xml_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

    def collect_drug_data(self, drug_names: List[str]) -> Dict[str, Any]:
        """
        Collect drug data from RxNorm and DailyMed APIs.

        Args:
            drug_names: List of drug names to collect data for

        Returns:
            Dictionary with collection statistics
        """
        print(f"🔍 Getting RxNorm IDs for {len(drug_names)} drugs...")
        rxcui_map = get_rxcui_mapping(drug_names)

        # Save RXCUI mapping
        rxcui_path = self.data_dir / "rxcui_map.json"
        with open(rxcui_path, "w") as f:
            json.dump(rxcui_map, f, indent=2)

        print(f"💾 Saved RXCUI mapping to {rxcui_path}")

        # Collect DailyMed data
        drug_spl_mapping = {}
        collected = 0

        for drug_name, rxcui in rxcui_map.items():
            if not rxcui:
                print(f"⚠️  Skipping '{drug_name}' - no RXCUI found")
                continue

            print(f"📥 Fetching DailyMed data for '{drug_name}' (RXCUI: {rxcui})")

            try:
                setids = dailymed_setids_for_rxcui(rxcui, pagesize=1)
                if not setids:
                    print(f"⚠️  No setids found for {drug_name}")
                    continue

                # Get latest SPL
                latest_setid, title, product_type = setids[0]
                xml_content = fetch_spl_xml(latest_setid)

                # Save XML file
                xml_path = self.xml_dir / f"{latest_setid}.xml"
                with open(xml_path, "w") as f:
                    f.write(xml_content)

                drug_spl_mapping[latest_setid] = {
                    "ingredient_name": drug_name,
                    "drug_name": title,
                    "product_type": product_type,
                    "rxcui": rxcui
                }

                collected += 1
                print(f"✅ Collected data for {drug_name}")

            except Exception as e:
                print(f"❌ Error collecting data for {drug_name}: {e}")

        # Save SPL mapping
        spl_path = self.data_dir / "drug_spl_mapping.json"
        with open(spl_path, "w") as f:
            json.dump(drug_spl_mapping, f, indent=2)

        stats = {
            "total_drugs": len(drug_names),
            "found_rxcuis": len([r for r in rxcui_map.values() if r]),
            "collected_spls": collected
        }

        print(f"📊 Collection complete: {stats}")
        return stats

    def process_xml_files(self) -> List[Fragment]:
        """
        Process XML files into fragments.

        Returns:
            List of Fragment objects
        """
        print("🔄 Processing XML files into fragments...")

        # Load drug metadata
        spl_mapping_path = self.data_dir / "drug_spl_mapping.json"
        drug_metadata = {}
        if spl_mapping_path.exists():
            with open(spl_mapping_path, "r") as f:
                drug_metadata = json.load(f)

        fragments = []
        processed = 0

        for xml_file in self.xml_dir.glob("*.xml"):
            try:
                # Parse XML
                parsed_data = parse_spl_xml(str(xml_file))
                if not parsed_data["sections"]:
                    continue

                # Get metadata for this setid
                setid = xml_file.stem
                metadata = drug_metadata.get(setid, {})

                # Create fragments
                file_fragments = make_fragments_from_json_label(parsed_data, metadata)
                fragments.extend(file_fragments)

                # Save parsed JSON
                json_path = self.json_dir / f"{setid}.json"
                with open(json_path, "w") as f:
                    json.dump(parsed_data, f, indent=2)

                processed += 1
                print(f"✅ Processed {xml_file.name} -> {len(file_fragments)} fragments")

            except Exception as e:
                print(f"❌ Error processing {xml_file.name}: {e}")

        print(f"🎯 Created {len(fragments)} fragments from {processed} XML files")

        # Save fragments to Parquet file (matching RAG_extract_dailymed notebook)
        if fragments and HAS_PANDAS:
            print("💾 Saving fragments to Parquet...")
            rows = []
            for f in fragments:
                d = asdict(f)  # each fragment is a dict
                d["char_count"] = len(f.text)
                d["content_md5"] = to_md5(f.text)  # hashing of the text for storage efficiency
                # Stable unique key for joins/lookups
                d["uid"] = f"{f.set_id}::{f.fragment_id}"
                rows.append(d)

            df = pd.DataFrame(rows)

            # save it as a parquet
            out_path = self.data_dir / "fragments.parquet"
            df.to_parquet(out_path, index=False)
            print(f"✅ Saved {len(df):,} fragments → {out_path}")
        elif fragments and not HAS_PANDAS:
            print("⚠️  Pandas not installed - skipping Parquet export. Install with: pip install pandas pyarrow")

        return fragments

    def build_index(self, fragments: List[Fragment]):
        """
        Build embeddings and search index.

        Args:
            fragments: List of Fragment objects to index
        """
        if not self.embedding_generator:
            raise ValueError("Mistral API key required for embedding generation")

        print("🧠 Generating embeddings...")
        texts = [f.text for f in fragments]
        embeddings = self.embedding_generator.get_text_embeddings(texts)

        print("🗂️  Building vector store...")
        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.add_fragments(fragments, embeddings)

        # Save index
        self.vector_store.save(str(self.index_dir))
        print(f"💾 Saved index to {self.index_dir}")

        # Initialize searcher
        self.searcher = HybridSearcher(
            self.vector_store,
            self.embedding_generator,
            cohere_api_key=self.cohere_api_key
        )
        if self.cohere_api_key:
            print("🔍 Search system ready (with reranking)!")
        else:
            print("🔍 Search system ready!")

    def load_index(self):
        """Load existing index from disk."""
        if not self.embedding_generator:
            raise ValueError("Mistral API key required for search")

        print("📂 Loading existing index...")
        self.vector_store = VectorStore()
        self.vector_store.load(str(self.index_dir))
        self.searcher = HybridSearcher(
            self.vector_store,
            self.embedding_generator,
            cohere_api_key=self.cohere_api_key
        )
        print(f"✅ Loaded {len(self.vector_store)} fragments")

    def search(self, query: str, k: int = 5) -> QueryResult:
        """
        Search for information.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            QueryResult object
        """
        if not self.searcher:
            raise ValueError("Search system not initialized. Run build_index() or load_index() first.")

        start_time = time.time()
        results = self.searcher.hybrid_search(query, k=k)
        retrieval_time = time.time() - start_time

        # For now, just return the retrieved fragments as the "response"
        # In a full implementation, this would call an LLM to generate a response
        response_parts = []
        for i, result in enumerate(results[:3], 1):
            response_parts.append(f"{i}. {result.fragment.text[:200]}...")

        response = "\n\n".join(response_parts)

        query_result = QueryResult(
            query=query,
            retrieved_fragments=results,
            response=response,
            retrieval_time=retrieval_time,
            generation_time=0.0  # No LLM generation in this basic version
        )

        return query_result

    def interactive_search(self):
        """Interactive search loop."""
        print("\n🤖 Interactive Search Mode")
        print("Type your questions about medications. Type 'quit' to exit.\n")

        while True:
            try:
                query = input("❓ Your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query:
                    continue

                print("🔍 Searching...")
                result = self.search(query)

                print(f"\n📊 Found {len(result.retrieved_fragments)} relevant fragments")
                print(f"⏱️  Retrieval time: {result.retrieval_time:.3f}s\n")

                print("📋 Top Results:")
                for i, res in enumerate(result.retrieved_fragments[:3], 1):
                    print(f"\n{i}. Score: {res.score:.3f}")
                    print(f"   Drug: {res.fragment.drug_name}")
                    print(f"   Section: {res.fragment.section_title}")
                    print(f"   Text: {res.fragment.text[:300]}...")

                print("\n" + "="*80)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Clinical RAG Assistant")
    parser.add_argument("command", choices=["collect", "process", "index", "search", "full-pipeline"],
                       help="Command to run")
    parser.add_argument("--drugs", nargs="+", default=["aspirin", "ibuprofen", "acetaminophen"],
                       help="Drug names to collect data for")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--mistral-key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--cohere-key", help="Cohere API key for reranking (or set COHERE_API_KEY env var)")
    parser.add_argument("--query", help="Search query for search command")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking with Cohere")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Get API keys
    mistral_key = args.mistral_key or os.getenv("MISTRAL_API_KEY")
    cohere_key = args.cohere_key or os.getenv("COHERE_API_KEY")

    # Initialize system
    rag = ClinicalRAGSystem(data_dir=args.data_dir, mistral_api_key=mistral_key, cohere_api_key=cohere_key)

    try:
        if args.command == "collect":
            rag.collect_drug_data(args.drugs)

        elif args.command == "process":
            fragments = rag.process_xml_files()
            print(f"✅ Processing complete: {len(fragments)} fragments created")

        elif args.command == "index":
            if not mistral_key:
                print("❌ Mistral API key required for indexing")
                return
            fragments = rag.process_xml_files()
            rag.build_index(fragments)

        elif args.command == "search":
            if not mistral_key:
                print("❌ Mistral API key required for search")
                return
            rag.load_index()
            if args.query:
                result = rag.search(args.query)
                print(f"Query: {result.query}")
                print(f"Results: {len(result.retrieved_fragments)}")
                for res in result.retrieved_fragments[:3]:
                    print(f"  - {res.fragment.drug_name}: {res.fragment.text[:100]}...")
            else:
                rag.interactive_search()

        elif args.command == "full-pipeline":
            if not mistral_key:
                print("❌ Mistral API key required for full pipeline")
                return
            print("🚀 Running full pipeline...")
            rag.collect_drug_data(args.drugs)
            fragments = rag.process_xml_files()
            rag.build_index(fragments)
            rag.interactive_search()

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())