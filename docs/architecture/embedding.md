# Embedding

Mapping each chunk to a **vector** so retrieval becomes nearest-neighbour search, then
loading those vectors (plus metadata) into a store. Two decisions dominate: the
**embedding model** and the **vector store**.

Part of the [harvester pipeline](harvest.md): **[chunk](chunking.md) → _embed_ → load**.

## The consistency invariant (the footgun)

**Documents and queries must be embedded by the same model and version.** Vectors from
different models are not comparable, so swapping the model means **re-embedding the entire
corpus**. Pin the model, and record its id + version in the catalog next to every vector
so a future migration knows what needs re-embedding.

## Choosing a model

Axes that decide it:

- **Hosted vs self-hosted** — the cluster has GPU nodes, so local models are viable and
  remove per-call cost and data-egress.
- **General vs domain** — science-tuned models beat general ones on papers.
- **Dimensions / max input tokens / cost / [MTEB](https://huggingface.co/spaces/mteb/leaderboard) score / multilingual** needs.

| Model | Host | Notes |
|---|---|---|
| OpenAI `text-embedding-3-small` / `-large` | hosted | strong general baseline, tunable dims |
| Cohere `embed` / **Voyage** | hosted | competitive; Voyage has domain variants |
| **SPECTER2** | local | trained on **scientific documents** — good for papers |
| `bge` / `e5` / `nomic-embed` / `gte` | local | open, run on the GPU nodes |

## Distance metric & normalization

Pick **cosine** or **dot product** to match how the model was trained, and configure the
vector store's index with the **same metric**. Normalize vectors if the model expects it.
A mismatch here silently degrades every search.

## Vector stores

| Store | Notes |
|---|---|
| **pgvector** | Postgres extension; simplest if you already run Postgres |
| **Qdrant / Weaviate / Milvus** | purpose-built, rich metadata filtering, HNSW indexes |
| **OpenSearch Serverless** | AWS-managed; pairs with the rest of the AWS footprint |
| **Pinecone** | fully managed SaaS |
| **Bedrock Knowledge Bases** | AWS-managed *end to end* — does chunk + embed + store for you if you want less to build |

All of the above support **metadata filtering** — essential so retrieval can constrain by
DOI/year/venue. Choose the **index type** (HNSW vs IVF) for your recall/latency tradeoff.

## Throughput & idempotency

- **Batch** embedding calls — hosted APIs have rate limits; local models want large GPU
  batches for throughput.
- **Idempotency** — key each vector by `chunk_id` and skip chunks whose content hash is
  unchanged, so re-runs are incremental, not full recomputes.

## Reranking (adjacent)

Embeddings give fast approximate recall; a **cross-encoder reranker** (e.g. Cohere Rerank,
`bge-reranker`) re-scores the top-K for precision at query time. It lives on the retrieval
side, not in the harvester, but it's the usual next quality lever after embeddings.

## Notes

- Record model id, dimensions, and metric in the catalog — a corpus you can't identify the
  embedding model for is a corpus you can't safely migrate.
- Managed (Bedrock KB) trades the GPU work of local embedding for per-call cost and less
  control — a per-stage cost/control call, not a whole-pipeline one.
- Previous stage: [chunking](chunking.md). Overview: [harvester pipeline](harvest.md).
