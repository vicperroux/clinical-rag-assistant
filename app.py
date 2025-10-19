#!/usr/bin/env python3
"""
Clinical RAG Assistant - Medical Web Interface

A retrieval-augmented generation system for clinical drug information.
Features LLM-powered answers, AI reranking, and source attribution with DailyMed links.
"""

import os
import re
import textwrap
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv

from main import ClinicalRAGSystem
from mistralai import Mistral


def run_mistral(
    client: Mistral,
    prompt: str,
    model: str = "mistral-small-latest",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> str:
    """
    Run Mistral model using the Mistral client.

    Args:
        client: Mistral client instance
        prompt: Input prompt
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Mistral API error: {str(e)}")


def clean_markdown_formatting(text: str) -> str:
    """Minimal formatting - let Mistral's markdown come through naturally."""
    # Make citations bold [1], [2], etc.
    text = re.sub(r"\[(\d+)\]", r"**[\1]**", text)

    # Clean up excessive whitespace only
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_drug_info(fragment):
    """Format drug information consistently."""
    drug = fragment.drug_name or fragment.ingredient_name or "Unknown Drug"
    section = (
        f"{fragment.section_code} - {fragment.section_title}"
        if fragment.section_code
        else fragment.section_title
    )
    return drug, section


def enhanced_rag_qa(
    rag_system: ClinicalRAGSystem,
    mistral_client: Mistral,
    query: str,
    retrieval_type: str = "hybrid",
    rerank: bool = True,
    k_final: int = 6,
):
    """
    Generate clinical answers with source attribution using RAG.

    Args:
        rag_system: Initialized RAG system
        mistral_client: Mistral client for LLM generation
        query: Clinical question
        retrieval_type: Retrieval mode - "hybrid", "dense", or "sparse"
        rerank: Enable AI reranking for improved precision
        k_final: Number of source documents to retrieve

    Returns:
        Tuple of (answer_text, sources_html)
    """
    if not query.strip():
        return "Please enter a question about drug information.", ""

    # Enhanced clinical context for structured responses
    clinical_context = """
    You are a medical information assistant helping clinicians and pharmacists with drug label information.

    Provide clear, clinically relevant answers based only on the provided excerpts.

    RESPONSE GUIDELINES:
    - Start with a direct answer to the question (no label needed)
    - Organize supporting information naturally using markdown formatting
    - Use **bold** for important medical terms and doses
    - Always cite sources as [1], [2] after key statements
    - When citing multiple sources, separate them with commas like [1], [2], [3] instead of [1][2][3]
    - Use section headers (like **Dosing Information** or **Important Warnings**) only when they help organize content
    - Write in natural clinical language, not rigid templates
    - Note any important information gaps

    Answer ONLY from the provided excerpts. If information is missing, state this clearly.
    """

    try:
        # 1) Retrieve relevant fragments using the notebook's retrieve interface
        if not hasattr(rag_system, "searcher") or not rag_system.searcher:
            return "❌ System not ready. Please load the index first.", ""

        hits = rag_system.searcher.retrieve(
            query=query, retrieval_type=retrieval_type, rerank=rerank, k_final=k_final
        )

        if not hits:
            return "No relevant drug information found for your query.", ""

        # Filter low-confidence results for medical safety
        if rerank:
            hits = [h for h in hits if h.score >= 0.3] or hits[:2]  # at least 2

        # 2) Build context for LLM
        ctx_blocks = []
        for i, h in enumerate(hits, 1):
            f = h.fragment
            drug, section = format_drug_info(f)
            header = f"[{i}] {drug} — {section}"
            body = f.text or ""
            ctx_blocks.append(f"{header}\n{body}")
        context_text = "\n\n---------------------\n\n".join(ctx_blocks)

        # 3) Natural prompt for markdown-formatted responses
        prompt = f"""
        {clinical_context}

        Context excerpts from drug labels:
        ---------------------
        {context_text}
        ---------------------

        Clinical Question: {query}

        Provide a natural clinical response using markdown formatting. Start directly with the answer, then organize any supporting information clearly. Use **bold** for important terms and **[1]**, **[2]** for citations.

        Response:
        """.strip()

        # 4) Get response and clean formatting
        raw_answer = run_mistral(mistral_client, prompt, model="mistral-small-latest")
        clean_answer = clean_markdown_formatting(raw_answer)

        # 5) Create compact source boxes using HTML with CSS classes
        sources_html = ""
        if hits:
            for i, hit in enumerate(hits):
                f = hit.fragment
                drug, section = format_drug_info(f)

                # Build DailyMed link
                link_html = ""
                if hasattr(f, "set_id") and f.set_id:
                    dailymed_url = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={f.set_id}"
                    link_html = f'<div class="source-field"><strong>🔗 Full Label:</strong> <a href="{dailymed_url}" target="_blank">View on DailyMed</a></div>'

                # Create compact HTML box
                sources_html += f"""
                <div class="source-box">
                    <div class="source-header">📋 Source [{i + 1}]</div>

                    <div class="source-field"><strong>🏥 Drug:</strong> {drug}</div>
                    <div class="source-field"><strong>📖 Section:</strong> {section}</div>
                    <div class="source-field"><strong>📅 Date:</strong> {f.effective_date or "Not specified"}</div>
                    <div class="source-field"><strong>📍 Path:</strong> {f.path}</div>
                    {link_html}

                    <div class="source-content">
                        {textwrap.shorten(f.text or "", width=300)}
                    </div>
                </div>
                """

        return clean_answer, sources_html

    except Exception as e:
        error_msg = f"System Error: {str(e)}\nPlease try again or contact support."
        return error_msg, ""


def create_example_queries():
    """Predefined clinical queries for quick testing."""
    return [
        "What is the recommended dosage of atorvastatin?",
        "How should warfarin be monitored?",
        "What are the contraindications for ibuprofen?",
        "What are the common side effects of lisinopril?",
        "Can metformin be used during pregnancy?",
    ]


def create_medical_interface(rag_system: ClinicalRAGSystem, mistral_client: Mistral):
    """Create the medical web interface with LLM-powered answers."""

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Medical Drug Information Assistant",
        css="""
        .medical-header {
            background: #1e3a8a;
            color: white !important;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .medical-header h1, .medical-header p, .medical-header small {
            color: white !important;
        }
        .disclaimer-top {
            background-color: #fef3cd;
            border: 1px solid #faebcc;
            padding: 6px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            background: #f0f7ff;
            border-radius: 10px;
            border-left: 4px solid #3b82f6;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .source-box {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: #f9fafb;
            font-size: 13px;
            line-height: 1.4;
        }
        .source-header {
            font-weight: bold;
            color: #1e3a8a;
            margin-bottom: 6px;
            font-size: 14px;
        }
        .source-field {
            margin: 3px 0;
        }
        .source-content {
            background: #e5e7eb;
            padding: 6px;
            border-radius: 4px;
            margin: 4px 0;
            font-style: italic;
            font-size: 12px;
        }
        """,
    ) as demo:

        # Header
        gr.HTML(
            f"""
        <div class="medical-header">
            <h1>🏥 Medical Drug Information Assistant</h1>
            <p>Drug label information retrieval for clinicians and pharmacists</p>
            <p><small>Based on FDA-approved drug labels from DailyMed</small></p>
        </div>
        """
        )

        # Disclaimer moved to top
        gr.HTML(
            f"""
        <div class="disclaimer-top">
            <h4>⚠️ Medical Disclaimer</h4>
            <p><strong>For informational purposes only.</strong> This tool provides information from FDA drug labels but should not replace professional medical judgment. Always consult current prescribing information and clinical guidelines for patient care decisions.</p>
            <p><small>Last updated: {datetime.now().strftime('%Y-%m-%d')} | Data source: FDA DailyMed</small></p>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Query input with examples
                query_input = gr.Textbox(
                    lines=3,
                    label="🔍 Enter your clinical question",
                    placeholder="Example: What is the recommended dosing for pediatric patients?",
                    info="Ask about dosing, contraindications, side effects, interactions, etc.",
                )

                # Example queries
                gr.Examples(
                    examples=create_example_queries(),
                    inputs=query_input,
                    label="💡 Common Clinical Questions",
                )

            with gr.Column(scale=1):
                # Settings panel - hidden by default
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    retrieval_type = gr.Dropdown(
                        choices=["hybrid", "dense", "sparse"],
                        value="hybrid",
                        label="Retrieval Method",
                        info="Hybrid combines semantic and keyword search",
                    )

                    rerank = gr.Checkbox(
                        value=True,
                        label="Use AI Reranking",
                        info="Improves result quality (recommended)",
                    )

                    k_final = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=6,
                        step=1,
                        label="Number of Sources",
                        info="More sources = more context but longer processing",
                    )

        # Submit button
        submit_btn = gr.Button(
            "🔍 Search Drug Information", variant="primary", size="lg"
        )

        # Knowledge Enriched Answer section
        gr.Markdown("## Knowledge Enriched Answer")
        answer_output = gr.Markdown(value="", elem_id="clinical_answer")

        # Supporting Sources section with more spacing
        gr.HTML("<div style='margin-top: 40px;'></div>")  # Extra spacing
        gr.Markdown("## Supporting Sources")
        sources_output = gr.HTML(value="", elem_id="supporting_sources")

        # Connect the interface with loading states
        def handle_submit(query, retrieval_type, rerank, k_final):
            if not query.strip():
                return "Please enter a question about drug information.", ""

            # Perform the actual search
            answer, sources = enhanced_rag_qa(
                rag_system, mistral_client, query, retrieval_type, rerank, k_final
            )
            return answer, sources

        submit_btn.click(
            fn=handle_submit,
            inputs=[query_input, retrieval_type, rerank, k_final],
            outputs=[answer_output, sources_output],
            show_progress=True,
        )

        # Allow Enter key to submit
        query_input.submit(
            fn=handle_submit,
            inputs=[query_input, retrieval_type, rerank, k_final],
            outputs=[answer_output, sources_output],
            show_progress=True,
        )

    return demo


def main():
    """Launch the enhanced interface."""
    # Load environment variables
    load_dotenv()

    # Get API keys
    mistral_key = os.getenv("MISTRAL_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")

    if not mistral_key:
        print("❌ MISTRAL_API_KEY not found in environment")
        return

    print("🔑 API keys loaded")

    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag_system = ClinicalRAGSystem(
        data_dir="data", mistral_api_key=mistral_key, cohere_api_key=cohere_key
    )

    # Load existing index
    print("📂 Loading index...")
    rag_system.load_index()
    print(f"✅ Loaded {len(rag_system.vector_store)} fragments")

    # Initialize Mistral client for LLM generation
    mistral_client = Mistral(api_key=mistral_key)

    # Create and launch interface
    print("🌐 Launching interface...")
    demo = create_medical_interface(rag_system, mistral_client)
    demo.launch(debug=True, share=False, server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
