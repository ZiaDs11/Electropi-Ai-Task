"""The 5 fixed prompts used for BOTH precisions, so quality is comparable.
Chosen to probe different capabilities: factual, reasoning, code, summarisation,
and instruction-following/format."""

PROMPTS = [
    # factual / knowledge
    "In two sentences, explain what a vector database is and why RAG systems use one.",
    # arithmetic / step reasoning
    "A courier delivers 3 orders per hour and works 8 hours. If 15% of orders are "
    "cancelled, how many are delivered? Show your reasoning.",
    # code generation
    "Write a Python function `is_palindrome(s)` that ignores case and spaces. "
    "Return only the code.",
    # summarisation
    "Summarise in one sentence: 'The team migrated the telemetry pipeline from "
    "DBSCAN to HDBSCAN, added UMAP dimensionality reduction, and shipped two "
    "Streamlit dashboards for cluster and state monitoring.'",
    # instruction-following / structured output
    "List exactly three benefits of 4-bit quantization as a JSON array of strings.",
]
