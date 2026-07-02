# TODOS

## Benchmark `cuda-qwen4b-contextual-v1` on rag-literature-rag
- **What:** Build + eval the one untested retrieval config (Qwen3-4B embedder + contextual-retrieval recipe) via the profile-bench flow.
- **Why:** Last open question from the 2026-06-27 quality campaign; closes the embedder ladder for good.
- **Pros:** Runs unattended overnight; definitive answer.
- **Cons:** Expected NULL-ish — 4B tied 0.6B on bpref at higher cost on the plain recipe.
- **Context:** `rag-literature-rag/docs/quality-campaign-2026-06-27.md` names it as the untested edge. Requires an overnight contextual ingest for the 4B profile (indexes live on the desktop tree).
- **Depends on:** Desktop/laptop tree reconciliation (E0 of the research-tool plan).
