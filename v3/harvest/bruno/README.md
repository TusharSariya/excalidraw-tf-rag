# harvest — Bruno collection

A [Bruno](https://github.com/usebruno/bruno) collection that reproduces the upstream HTTP
requests the harvest workers make. Harvest has no API of its own — it's a client of arXiv,
Semantic Scholar, OpenAlex, and Supabase — so these requests let you inspect the raw
responses by hand while building out the pipeline.

## Layout

```
bruno.json                 collection manifest
environments/staging.bru   base URLs, default arxiv_id, secret-var declaration
arxiv/                     fetch-paper            (fetch_arxiv_paper)
semantic-scholar/          paper / authors / references / citations
supabase/                  paper-exists           (paper_exists dedupe lookup)
```

## Environment

Select the **staging** environment before running. It defines:

| var           | value                                    | notes                        |
| ------------- | ---------------------------------------- | ---------------------------- |
| `arxiv_base`  | `https://export.arxiv.org/api`           |                              |
| `ss_base`     | `https://api.semanticscholar.org/graph/v1` |                            |
| `supabase_url`| `https://vumvzcugxbnkfnujpjjc.supabase.co` |                            |
| `arxiv_id`    | `1706.03762`                             | "Attention Is All You Need"  |
| `supabase_key`| *(secret)*                               | set locally, never committed |

arXiv and Semantic Scholar need no auth (public tier). Only the Supabase request needs
`supabase_key` (the `SUPABASE_KEY` from the package's `.env`). It's declared as a Bruno
**secret var** — its value lives in Bruno's local store / your shell, not in these files.

## Run it

**Bruno app:** open this folder as a collection → pick the `staging` env → set `supabase_key` →
run any request.

**CLI (headless):**

```bash
npm i -g @usebruno/cli
cd v3/harvest/bruno

bru run arxiv semantic-scholar --env staging          # no secrets needed
bru run supabase --env staging --env-var supabase_key=$SUPABASE_KEY
```
