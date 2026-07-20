"""
rag_pipeline.py
---------------
A small RAG pipeline over the markdown docs in ./docs.

Design:
  Load (.md) -> chunk -> embed (local sentence-transformers, free) -> FAISS
  -> similarity search WITH SCORES -> threshold guardrail -> LLM answer with
     [source] citations.

Key requirements from the test, and where each is handled:
  * Chunk + embed into a vector store      -> build_vectorstore()
  * Retrieve + answer WITH CITATIONS       -> answer()  (citations = chunk source)
  * "No relevant context" handled explicitly -> RELEVANCE_THRESHOLD gate in answer()
  * 3 example Q&As                          -> see README.md (run `python rag_pipeline.py --demo`)

Embeddings are local & free (all-MiniLM-L6-v2). The LLM is any OpenAI-compatible
endpoint (real OpenAI, or a local Ollama/vLLM via OPENAI_BASE_URL). If no LLM is
configured, --demo still shows retrieval + the guardrail using an extractive
fallback so you can see the pipeline work end-to-end.

Usage:
  python rag_pipeline.py --build                 # build the FAISS index
  python rag_pipeline.py --ask "your question"
  python rag_pipeline.py --demo                  # run the 3 example questions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# FAISS `similarity_search_with_score` returns L2 distance for these embeddings:
# smaller = closer. Anything above this distance is treated as "no relevant
# context" so the model can't hallucinate from garbage retrieval.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.1"))
TOP_K = int(os.getenv("TOP_K", "3"))


def _embeddings():
    # Imported lazily so the module loads even if this backend isn't installed
    # (and so tests can inject a fake embedding).
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore() -> FAISS:
    """Load docs -> chunk -> embed -> FAISS, and persist to disk."""
    docs = []
    for md in sorted(DOCS_DIR.glob("*.md")):
        loaded = TextLoader(str(md), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = md.name  # citation handle
        docs.extend(loaded)

    if not docs:
        sys.exit(f"No .md files found in {DOCS_DIR}")

    # Small docs -> small chunks with overlap so a fact isn't split across a
    # boundary. See README write-up for how this changes on longer docs.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, _embeddings())
    vs.save_local(str(INDEX_DIR))
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs -> {INDEX_DIR}")
    return vs


def load_vectorstore() -> FAISS:
    if not INDEX_DIR.exists():
        return build_vectorstore()
    return FAISS.load_local(
        str(INDEX_DIR), _embeddings(), allow_dangerous_deserialization=True
    )


def _format_context(hits) -> str:
    """Number each retrieved chunk so the LLM can cite [1], [2], ... which we
    map back to file names."""
    lines = []
    for i, (doc, score) in enumerate(hits, 1):
        lines.append(f"[{i}] (source: {doc.metadata['source']})\n{doc.page_content.strip()}")
    return "\n\n".join(lines)


PROMPT_TMPL = """You are a support assistant. Answer the question using ONLY the
context below. Cite the sources you used with their bracket numbers like [1].
If the context does not contain the answer, reply exactly:
"I don't have information about that in the provided documents."

Context:
{context}

Question: {question}

Answer (with citations):"""


def _llm_answer(question: str, context: str) -> str | None:
    """Call an OpenAI-compatible LLM. Returns None if none is configured."""
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")):
        return None
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user",
                   "content": PROMPT_TMPL.format(context=context, question=question)}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, vs: FAISS | None = None) -> dict:
    """Retrieve, apply the no-context guardrail, then answer with citations."""
    vs = vs or load_vectorstore()
    hits = vs.similarity_search_with_score(question, k=TOP_K)

    # --- Guardrail: if nothing is close enough, refuse instead of guessing ---
    relevant = [(d, s) for (d, s) in hits if s <= RELEVANCE_THRESHOLD]
    if not relevant:
        return {
            "question": question,
            "answer": "I don't have information about that in the provided documents.",
            "citations": [],
            "grounded": False,
        }

    context = _format_context(relevant)
    citations = sorted({d.metadata["source"] for d, _ in relevant})

    llm_text = _llm_answer(question, context)
    if llm_text is None:
        # Extractive fallback so the pipeline is demonstrable without an LLM key.
        top_doc = relevant[0][0]
        llm_text = (
            "[LLM not configured — extractive fallback]\n"
            + top_doc.page_content.strip()
            + f"\n(from {top_doc.metadata['source']})"
        )

    return {
        "question": question,
        "answer": llm_text,
        "citations": citations,
        "grounded": True,
        "scores": [round(float(s), 3) for _, s in relevant],
    }


EXAMPLE_QUESTIONS = [
    "Can I cancel my order after it's out for delivery?",
    "How many PiPoints do I need for a discount, and what's it worth?",
    "What is the capital of France?",  # deliberately out-of-scope -> guardrail
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the index")
    ap.add_argument("--ask", type=str, help="ask a single question")
    ap.add_argument("--demo", action="store_true", help="run the 3 example questions")
    args = ap.parse_args()

    if args.build:
        build_vectorstore()
        if not (args.ask or args.demo):
            return

    vs = load_vectorstore()

    if args.ask:
        _print(answer(args.ask, vs))
    if args.demo:
        for q in EXAMPLE_QUESTIONS:
            _print(answer(q, vs))
    if not (args.ask or args.demo or args.build):
        ap.print_help()


def _print(res: dict) -> None:
    print("\n" + "=" * 70)
    print("Q:", res["question"])
    print("A:", res["answer"])
    print("Citations:", ", ".join(res["citations"]) or "(none)")
    print("Grounded:", res["grounded"], "| scores:", res.get("scores"))


if __name__ == "__main__": 
    main() 