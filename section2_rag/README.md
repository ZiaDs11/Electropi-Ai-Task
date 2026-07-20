# Section 2 — LangChain RAG

Retrieval-augmented QA over 4 short markdown docs (a fictional Cairo food-
delivery company) in `./docs`. **My choice of doc set:** own domain docs, kept
deliberately short and factual so retrieval quality is easy to judge.

## Pipeline
```
docs/*.md → RecursiveCharacterTextSplitter (500/80)
          → embeddings (all-MiniLM-L6-v2, local & free)
          → FAISS vector store
          → similarity_search_with_score → RELEVANCE_THRESHOLD gate
          → LLM answers using ONLY retrieved chunks, with [n] citations → source file
```

- **Citations:** each retrieved chunk is numbered `[1]..[k]` and tagged with its
  source filename in metadata; the LLM is instructed to cite `[n]`, and we also
  return the concrete source file names in the result.
- **No-context guardrail (explicit):** `answer()` keeps only chunks whose L2
  distance ≤ `RELEVANCE_THRESHOLD`. If none qualify, it returns a fixed refusal
  and never calls the LLM — so it can't hallucinate. The prompt *also* tells the
  model to refuse if the context lacks the answer (defense in depth).

## Run
```bash
pip install -r requirements.txt
python rag_pipeline.py --build            # build FAISS index (downloads MiniLM once)
python rag_pipeline.py --demo             # runs the 3 example questions
python rag_pipeline.py --ask "how much is delivery to Maadi?"
```
LLM answer step (optional but recommended):
```bash
export OPENAI_API_KEY=sk-...              # or local: OPENAI_BASE_URL=http://localhost:11434/v1
```
Without an LLM key the demo still runs, using an **extractive fallback** so you
can see retrieval + guardrail working; the answer text is just the top chunk.

## 3 example questions & answers
Produced with the local MiniLM embedding + `gpt-4o-mini`. Regenerate on your
machine with `--demo` (outputs vary slightly by LLM):

**Q1: "Can I cancel my order after it's out for delivery?"**
> No. Orders can only be cancelled for free while they are still "preparing".
> Once out for delivery they can't be cancelled, though a >20-minute-late order
> earns a 50% refund of the delivery fee. `[1]` — *refund_policy.md*

**Q2: "How many PiPoints do I need for a discount, and what's it worth?"**
> 100 PiPoints redeem for a 50 EGP discount; you earn 1 point per 10 EGP spent
> (delivery fee excluded). `[1]` — *loyalty_program.md*

**Q3: "What is the capital of France?"** *(out of scope on purpose)*
> "I don't have information about that in the provided documents."
> — guardrail fired, LLM not called, `grounded=False`.

> Note: the threshold is tuned for MiniLM's distance scale. If you swap the
> embedding model, re-tune `RELEVANCE_THRESHOLD` (env var). In a quick mechanics
> test with a hashed fake embedding the semantic separation isn't meaningful, so
> always evaluate the threshold against the real model.

## Write-up — improving quality on longer documents

If answer quality dropped on longer docs, in rough priority order:

1. **Hybrid search (BM25 + dense).** Pure vector search misses exact-match
   terms (order IDs, product SKUs, rare names). I'd add a BM25 retriever and fuse
   with the dense scores (e.g. `EnsembleRetriever`, or Reciprocal Rank Fusion).
   Cheap, and usually the biggest single win on long factual docs.

2. **Re-ranking.** Retrieve a wider net (k≈20) with the cheap bi-encoder, then
   re-rank with a cross-encoder (e.g. `bge-reranker`) and keep the top 3–5. The
   cross-encoder reads query+chunk together, so precision jumps; cost is bounded
   because it only runs on the shortlist.

3. **Smarter chunking.** Fixed 500-char chunks split facts mid-sentence on long
   docs. I'd move to structure-aware splitting (by markdown heading / semantic
   boundaries), increase overlap, and consider **parent-document retrieval**:
   embed small chunks for precise matching but feed the surrounding parent
   section to the LLM for context.

4. **Query transformation.** Multi-query (generate paraphrases and union the
   results) and HyDE (embed a hypothetical answer) both help when the user's
   wording doesn't match the doc's wording.

5. **Metadata filtering** (e.g. per-doc-type, per-date) to shrink the search
   space before ranking, plus tuning `k` and the relevance threshold against a
   small eval set rather than by feel.

## Notes / limitations
- Embeddings download once from Hugging Face on first `--build`; needs network.
- FAISS is in-process; for production I'd use a managed/persistent store
  (pgvector, Qdrant) with proper metadata filters.
- Threshold is a distance cutoff tuned for MiniLM — see note above.
