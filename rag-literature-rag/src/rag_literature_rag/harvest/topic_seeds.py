"""Curated RAG research seeds — foundations, pipeline techniques, GraphRAG, agentic RAG, evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from rag_literature_rag.harvest.doi_resolver import resolve_doi_with_fallbacks
from rag_literature_rag.harvest.download import download_to_file
from rag_literature_rag.harvest.parallel import parallel_map
from rag_literature_rag.manifest import ManifestItem, relative_local_path
from rag_literature_rag.paths import PDF_DIR


def _arxiv_pdf(
    arxiv_id: str,
    title: str,
    year: int,
    tags: list[str],
    *,
    authors: list[str] | None = None,
    discovery_sources: list[str] | None = None,
) -> dict:
    slug = arxiv_id.replace(".", "-").lower()
    return {
        "id": f"arxiv-{slug}",
        "title": title,
        "authors": authors or [],
        "year": year,
        "url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "doi": f"10.48550/arXiv.{arxiv_id}",
        "source": "topic-seed",
        "tags": [*tags, "topic-seed"],
        "discoverySources": discovery_sources or [],
    }


# Tier 0 — foundational
TOPIC_PDF_SEEDS: list[dict] = [
    _arxiv_pdf(
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        2020,
        ["foundations", "generation"],
        authors=["Lewis", "Perez", "Piktus"],
    ),
    _arxiv_pdf(
        "2004.04906",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        2020,
        ["foundations", "dense-retrieval"],
        authors=["Karpukhin", "Oguz", "Min"],
    ),
    _arxiv_pdf(
        "2004.12832",
        "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        2020,
        ["foundations", "late-interaction"],
        authors=["Khattab", "Zaharia"],
    ),
    _arxiv_pdf(
        "2002.08909",
        "REALM: Retrieval-Augmented Language Model Pre-Training",
        2020,
        ["foundations", "pretraining"],
        authors=["Guu", "Lee", "Tung"],
    ),
    _arxiv_pdf(
        "2112.04426",
        "Improving language models by retrieving from trillions of tokens",
        2021,
        ["foundations", "parametric-memory"],
        authors=["Borgeaud", "Mensch", "Hoffmann"],
    ),
    _arxiv_pdf(
        "2007.01282",
        "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        2020,
        ["foundations", "reader"],
        authors=["Izacard", "Grave"],
    ),
    _arxiv_pdf(
        "2312.10997",
        "Retrieval-Augmented Generation for Large Language Models: A Survey",
        2023,
        ["survey", "taxonomy"],
        authors=["Gao", "Yunfan", "Xinyu"],
    ),
    # Tier 5 — gap-analysis 2026-07: embeddings/reranking foundations, agentic
    # search + RL, reasoning-intensive retrieval, LLM-as-judge evaluation.
    # arXiv-direct (id verified against export.arxiv.org title match) — added
    # because OpenAlex's metered endpoints were quota-exhausted on 2026-07-02.
    _arxiv_pdf(
        "2208.03299",
        "Atlas: Few-shot Learning with Retrieval Augmented Language Models",
        2022,
        ["foundations", "training"],
        authors=["Izacard", "Lewis", "Lomeli"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2212.03533",
        "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
        2022,
        ["dense-retrieval", "embeddings"],
        authors=["Wang", "Yang", "Huang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2309.07597",
        "C-Pack: Packaged Resources To Advance General Chinese Embedding",
        2023,
        ["dense-retrieval", "embeddings"],
        authors=["Xiao", "Liu", "Zhang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2506.05176",
        "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models",
        2025,
        ["dense-retrieval", "embeddings", "reranking"],
        authors=["Zhang", "Li", "Yang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2104.08663",
        "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
        2021,
        ["evaluation"],
        authors=["Thakur", "Reimers", "Ruckle"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2304.09542",
        "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents",
        2023,
        ["reranking"],
        authors=["Sun", "Yan", "Ma"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2003.06713",
        "Document Ranking with a Pretrained Sequence-to-Sequence Model",
        2020,
        ["reranking"],
        authors=["Nogueira", "Jiang", "Pradeep"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "1908.10084",
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        2019,
        ["dense-retrieval", "embeddings", "foundations"],
        authors=["Reimers", "Gurevych"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "1911.00172",
        "Generalization through Memorization: Nearest Neighbor Language Models",
        2019,
        ["foundations", "memory"],
        authors=["Khandelwal", "Levy", "Jurafsky"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "1904.08375",
        "Document Expansion by Query Prediction",
        2019,
        ["sparse-retrieval", "query-expansion"],
        authors=["Nogueira", "Yang", "Lin"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2307.03172",
        "Lost in the Middle: How Language Models Use Long Contexts",
        2023,
        ["long-context", "evaluation"],
        authors=["Liu", "Lin", "Hewitt"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2210.03350",
        "Measuring and Narrowing the Compositionality Gap in Language Models",
        2022,
        ["agentic", "query-expansion"],
        authors=["Press", "Zhang", "Min"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2210.03629",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
        2022,
        ["agentic"],
        authors=["Yao", "Zhao", "Yu"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2302.04761",
        "Toolformer: Language Models Can Teach Themselves to Use Tools",
        2023,
        ["agentic", "training"],
        authors=["Schick", "Dwivedi-Yu", "Dessi"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2112.09332",
        "WebGPT: Browser-assisted question-answering with human feedback",
        2021,
        ["agentic", "training"],
        authors=["Nakano", "Hilton", "Balaji"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2407.12883",
        "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval",
        2024,
        ["evaluation", "reasoning"],
        authors=["Su", "Yen", "Xia"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2508.07995",
        "DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval",
        2025,
        ["dense-retrieval", "reasoning"],
        authors=["Long", "Xu", "Wang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2502.18418",
        "Rank1: Test-Time Compute for Reranking in Information Retrieval",
        2025,
        ["reranking", "reasoning"],
        authors=["Weller", "Van Durme", "Lawrie"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2508.07050",
        "ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability",
        2025,
        ["reranking", "reasoning"],
        authors=["Liu", "Zhuang", "Ma"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2409.04701",
        "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models",
        2024,
        ["chunking", "embeddings"],
        authors=["Gunther", "Mohr", "Williams"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2407.16833",
        "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach",
        2024,
        ["long-context", "engineering"],
        authors=["Li", "Li", "Bhardwaj"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2409.01666",
        "In Defense of RAG in the Era of Long-Context Language Models",
        2024,
        ["long-context", "chunking"],
        authors=["Yu", "Ho", "Xu"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2501.06713",
        "MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation",
        2025,
        ["engineering"],
        authors=["Fan", "Chen", "Wang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2406.06519",
        "UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor",
        2024,
        ["evaluation"],
        authors=["Upadhyay", "MacAvaney", "Lin"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2411.16594",
        "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge",
        2024,
        ["evaluation", "survey"],
        authors=["Li", "Guo", "Fan"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2504.12558",
        "Benchmarking LLM-based Relevance Judgment Methods",
        2025,
        ["evaluation"],
        authors=["Abbasiantaeb", "Meng", "Azzopardi"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2409.13740",
        "Language agents achieve superhuman synthesis of scientific knowledge",
        2024,
        ["agentic", "evaluation"],
        authors=["Skarlinski", "Cox", "Laurent"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2503.00223",
        "DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning",
        2025,
        ["agentic", "training"],
        authors=["Jiang", "Ma", "Chen"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2503.05592",
        "R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning",
        2025,
        ["agentic", "training"],
        authors=["Song", "Jiang", "Chen"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2504.21776",
        "WebThinker: Empowering Large Reasoning Models with Deep Research Capability",
        2025,
        ["agentic"],
        authors=["Li", "Chen", "Su"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2504.03160",
        "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments",
        2025,
        ["agentic", "training"],
        authors=["Zheng", "Fu", "Zhang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2503.19470",
        "ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning",
        2025,
        ["agentic", "training"],
        authors=["Chen", "Li", "Yan"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2506.10408",
        "Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges",
        2025,
        ["agentic", "survey"],
        authors=["Zeng", "Chen", "Wang"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2504.13128",
        "FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents",
        2025,
        ["evaluation"],
        authors=["Thakur", "Chen", "Lipani"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
    _arxiv_pdf(
        "2407.18940",
        "LitSearch: A Retrieval Benchmark for Scientific Literature Search",
        2024,
        ["evaluation"],
        authors=["Ajith", "Xia", "Chevalier"],
        discovery_sources=["gap-analysis-2026-07"],
    ),
]

# P1c thin-topic fill (2026-07): learned sparse retrieval (SPLADE family), late
# interaction / multi-vector retrieval (ColBERTv2, PLAID, JaColBERT, PyLate),
# embedding-benchmark evaluation (MTEB/MMTEB), and modern embedding tech
# reports (jina-v3, nomic, Matryoshka Representation Learning). All arXiv IDs
# verified against the arxiv.org abstract page title before being added (wrong
# IDs poison the harvest).
SPARSE_LATE_INTERACTION_PDF_SEEDS: list[dict] = [
    _arxiv_pdf(
        "2107.05720",
        "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking",
        2021,
        ["sparse-retrieval"],
        authors=["Formal", "Piwowarski", "Clinchant"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2109.10086",
        "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval",
        2021,
        ["sparse-retrieval"],
        authors=["Formal", "Lassance", "Piwowarski", "Clinchant"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2403.06789",
        "SPLADE-v3: New baselines for SPLADE",
        2024,
        ["sparse-retrieval"],
        authors=["Lassance", "Formal", "Clinchant"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2112.01488",
        "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
        2021,
        ["dense-retrieval", "late-interaction", "evaluation"],
        authors=["Santhanam", "Khattab", "Saad-Falcon", "Potts", "Zaharia"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2205.09707",
        "PLAID: An Efficient Engine for Late Interaction Retrieval",
        2022,
        ["dense-retrieval", "late-interaction", "engineering"],
        authors=["Santhanam", "Khattab", "Potts", "Zaharia"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2312.16144",
        "JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings "
        "for Retrieval: Early Technical Report",
        2023,
        ["dense-retrieval", "late-interaction"],
        authors=["Clavié"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2407.20750",
        "JaColBERTv2.5: Optimising Multi-Vector Retrievers to Create "
        "State-of-the-Art Japanese Retrievers with Constrained Resources",
        2024,
        ["dense-retrieval", "late-interaction"],
        authors=["Clavié"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2508.03555",
        "PyLate: Flexible Training and Retrieval for Late Interaction Models",
        2025,
        ["dense-retrieval", "late-interaction", "engineering"],
        authors=["Chaffin", "Sourty"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2510.14880",
        "Fantastic (small) Retrievers and How to Train Them: mxbai-edge-colbert-v0 Tech Report",
        2025,
        ["dense-retrieval", "late-interaction"],
        authors=["Lee", "Shakir"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2210.07316",
        "MTEB: Massive Text Embedding Benchmark",
        2022,
        ["evaluation", "embeddings"],
        authors=["Muennighoff", "Tazi", "Magne", "Reimers"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2502.13595",
        "MMTEB: Massive Multilingual Text Embedding Benchmark",
        2025,
        ["evaluation", "embeddings"],
        authors=["Enevoldsen", "Chung", "Kerboua"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2409.10173",
        "jina-embeddings-v3: Multilingual Embeddings With Task LoRA",
        2024,
        ["dense-retrieval", "embeddings"],
        authors=["Sturua", "Mohr", "Akram", "Günther", "Wang"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2402.01613",
        "Nomic Embed: Training a Reproducible Long Context Text Embedder",
        2024,
        ["dense-retrieval", "embeddings"],
        authors=["Nussbaum", "Morris", "Duderstadt", "Mulyar"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2205.13147",
        "Matryoshka Representation Learning",
        2022,
        ["embeddings", "engineering"],
        authors=["Kusupati", "Bhatt", "Rege", "Wallingford"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
    _arxiv_pdf(
        "2411.04403",
        "Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers",
        2024,
        ["sparse-retrieval", "engineering"],
        authors=["Geng", "Zhu"],
        discovery_sources=["p1c-thin-topics-2026-07"],
    ),
]

# Tier 1–4 — resolved via OpenAlex / arXiv DOI
TOPIC_DOI_SEEDS: list[dict] = [
    # Tier 1 — core pipeline
    {
        "doi": "10.48550/arXiv.2310.11511",
        "tags": ["self-correcting", "reflection"],
        "title_hint": "Self-RAG",
    },
    {
        "doi": "10.48550/arXiv.2401.15884",
        "tags": ["self-correcting", "evaluator"],
        "title_hint": "Corrective Retrieval Augmented Generation",
    },
    {
        "doi": "10.48550/arXiv.2401.18059",
        "tags": ["chunking", "hierarchical-index"],
        "title_hint": "RAPTOR",
    },
    {
        "doi": "10.48550/arXiv.2209.10063",
        "tags": ["query-expansion"],
        "title_hint": "Generate rather than Retrieve: GenRead",
    },
    {
        "doi": "10.48550/arXiv.2305.06983",
        "tags": ["agentic", "iterative-retrieval"],
        "title_hint": "FLARE Active Retrieval Augmented Generation",
    },
    {
        "doi": "10.48550/arXiv.2305.14283",
        "tags": ["query-rewriting"],
        "title_hint": "Query Rewriting for Retrieval-Augmented Large Language Models",
    },
    {
        "doi": "10.48550/arXiv.2310.01352",
        "tags": ["training", "dual-tuning"],
        "title_hint": "RA-DIT",
    },
    {
        "doi": "10.48550/arXiv.2310.04408",
        "tags": ["compression", "context"],
        "title_hint": "RECOMP",
    },
    {
        "doi": "10.48550/arXiv.2311.09210",
        "tags": ["robustness", "noise"],
        "title_hint": "Chain-of-Note",
    },
    {"doi": "10.48550/arXiv.2311.08377", "tags": ["filtering", "context"], "title_hint": "FILCO"},
    {
        "doi": "10.48550/arXiv.2406.19215",
        "tags": ["adaptive", "uncertainty"],
        "title_hint": "SeaKR",
    },
    {"doi": "10.48550/arXiv.2407.02485", "tags": ["reranking", "unified"], "title_hint": "RankRAG"},
    {
        "doi": "10.48550/arXiv.2406.15319",
        "tags": ["long-context", "chunking"],
        "title_hint": "LongRAG",
    },
    {
        "doi": "10.48550/arXiv.2409.05591",
        "tags": ["memory", "draft-retrieval"],
        "title_hint": "MemoRAG",
    },
    {
        "doi": "10.48550/arXiv.2407.01219",
        "tags": ["engineering", "ablation"],
        "title_hint": "Searching for Best Practices in RAG",
    },
    {
        "doi": "10.48550/arXiv.2501.07391",
        "tags": ["engineering"],
        "title_hint": "Enhancing Retrieval-Augmented Generation: Best Practices",
    },
    # Tier 2 — GraphRAG
    {
        "doi": "10.48550/arXiv.2404.16130",
        "tags": ["graphrag", "community"],
        "title_hint": "From Local to Global: A Graph RAG Approach",
    },
    {
        "doi": "10.48550/arXiv.2408.08921",
        "tags": ["graphrag", "survey"],
        "title_hint": "Graph Retrieval-Augmented Generation: A Survey",
    },
    {"doi": "10.48550/arXiv.2405.14831", "tags": ["graphrag", "ppr"], "title_hint": "HippoRAG"},
    {
        "doi": "10.48550/arXiv.2410.05779",
        "tags": ["graphrag", "dual-level"],
        "title_hint": "LightRAG",
    },
    {
        "doi": "10.48550/arXiv.2504.11544",
        "tags": ["graphrag", "heterogeneous"],
        "title_hint": "NodeRAG",
    },
    {
        "doi": "10.48550/arXiv.2506.05690",
        "tags": ["graphrag", "evaluation"],
        "title_hint": "When to use Graphs in RAG",
    },
    {
        "doi": "10.48550/arXiv.2602.01965",
        "tags": ["graphrag", "query-aware"],
        "title_hint": "CatRAG",
    },
    {
        "doi": "10.1145/3795880",
        "tags": ["graphrag", "survey"],
        "title_hint": "Graph-Based Approaches in RAG",
    },
    # Tier 3 — agentic
    {
        "doi": "10.48550/arXiv.2501.09136",
        "tags": ["agentic", "survey"],
        "title_hint": "Agentic Retrieval-Augmented Generation Survey",
    },
    {
        "doi": "10.48550/arXiv.2507.09477",
        "tags": ["agentic", "reasoning", "survey"],
        "title_hint": "Towards Agentic RAG with Deep Reasoning",
    },
    {
        "doi": "10.48550/arXiv.2603.07379",
        "tags": ["agentic", "sok", "evaluation"],
        "title_hint": "SoK Agentic RAG",
    },
    {
        "doi": "10.48550/arXiv.2601.05264",
        "tags": ["engineering", "trust", "survey"],
        "title_hint": "Engineering the RAG Stack",
    },
    {"doi": "10.48550/arXiv.2602.03442", "tags": ["agentic", "interfaces"], "title_hint": "A-RAG"},
    {"doi": "10.48550/arXiv.2604.00865", "tags": ["agentic", "repair"], "title_hint": "Doctor-RAG"},
    {"doi": "10.48550/arXiv.2411.19443", "tags": ["agentic", "planning"], "title_hint": "Auto-RAG"},
    {
        "doi": "10.48550/arXiv.2310.03714",
        "tags": ["framework", "optimization"],
        "title_hint": "DSPy",
    },
    # Tier 4 — evaluation & embeddings
    {"doi": "10.48550/arXiv.2309.15217", "tags": ["evaluation", "metrics"], "title_hint": "RAGAS"},
    {
        "doi": "10.48550/arXiv.2309.01431",
        "tags": ["evaluation", "robustness"],
        "title_hint": "RGB Benchmark",
    },
    {
        "doi": "10.48550/arXiv.2406.04744",
        "tags": ["evaluation", "dynamic"],
        "title_hint": "CRAG Benchmark",
    },
    {
        "doi": "10.48550/arXiv.2409.12941",
        "tags": ["evaluation", "multi-hop"],
        "title_hint": "FRAMES",
    },
    {
        "doi": "10.48550/arXiv.2602.05975",
        "tags": ["evaluation", "bm25"],
        "title_hint": "SAGE Benchmark",
    },
    {"doi": "10.48550/arXiv.2601.21937", "tags": ["evaluation", "reasoning"], "title_hint": "DeR2"},
    {
        "doi": "10.48550/arXiv.2602.02053",
        "tags": ["evaluation", "graphrag"],
        "title_hint": "WildGraphBench",
    },
    {
        "doi": "10.48550/arXiv.2508.01959",
        "tags": ["embeddings", "contextual"],
        "title_hint": "SitEmb",
    },
    {
        "doi": "10.48550/arXiv.2202.06671",
        "tags": ["embeddings", "citation"],
        "title_hint": "SciNCL",
    },
    {
        "doi": "10.48550/arXiv.2501.01880",
        "tags": ["long-context", "comparison"],
        "title_hint": "Long Context vs RAG",
    },
]

TOPIC_METADATA_SEEDS: list[dict] = [
    {
        "id": "anthropic-contextual-retrieval-cookbook",
        "title": "Contextual Retrieval (Anthropic cookbook)",
        "authors": ["Anthropic"],
        "year": 2024,
        "url": "https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide",
        "source": "anthropic-cookbook",
        "tags": ["chunking", "contextual-retrieval", "implementer-guide"],
        "abstract": "Prepends LLM-generated context to chunks before embedding and BM25 indexing.",
    },
    {
        "id": "llamaindex-advanced-rag",
        "title": "LlamaIndex Advanced RAG Techniques",
        "authors": ["LlamaIndex"],
        "year": 2024,
        "url": "https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/",
        "source": "implementer-guide",
        "tags": ["engineering", "implementer-guide"],
        "abstract": "Practitioner guide to query transforms, reranking, and hybrid retrieval in LlamaIndex.",
    },
]


def _download_pdf_seed(spec: dict, *, dry_run: bool) -> ManifestItem:
    item = ManifestItem(
        id=spec["id"],
        title=spec["title"],
        authors=spec.get("authors", []),
        year=spec.get("year"),
        source=spec.get("source", "topic-seed"),
        url=spec["url"],
        localPath=f"data/raw/pdf/{spec['id']}.pdf",
        contentType="application/pdf",
        status="failed",
        tags=list(spec.get("tags", [])),
        doi=spec.get("doi"),
        discoverySources=list(spec.get("discoverySources", [])),
    )
    if dry_run:
        return item

    dest = PDF_DIR / f"{spec['id']}.pdf"
    for url in [spec["url"], *spec.get("fallback_urls", [])]:
        try:
            dl = download_to_file(dest, url)
        except Exception:
            # download_to_file writes the file to disk *before* it logs the
            # attempt to the harvest ledger; a failure in that logging step
            # (e.g. a sqlite hiccup) must never be mistaken for a failed
            # download and destroy bytes that already landed successfully.
            dl = None
        valid_pdf = dest.exists() and dest.read_bytes()[:4] == b"%PDF"
        if valid_pdf and (dl is None or dl.get("ok")):
            item.status = "ok"
            item.url = url
            item.sha256 = (dl or {}).get("sha256") or hashlib.sha256(dest.read_bytes()).hexdigest()
            item.localPath = relative_local_path(dest)
            return item
        if not valid_pdf:
            dest.unlink(missing_ok=True)
    return item


def harvest_topic_seeds(
    *,
    dry_run: bool = False,
    workers: int | None = None,
    on_doi_batch: Callable[[list[ManifestItem]], None] | None = None,
    doi_batch_size: int = 10,
) -> list[ManifestItem]:
    results: list[ManifestItem] = []

    all_pdf_seeds = [*TOPIC_PDF_SEEDS, *SPARSE_LATE_INTERACTION_PDF_SEEDS]
    results.extend(
        parallel_map(
            lambda spec: _download_pdf_seed(spec, dry_run=dry_run),
            all_pdf_seeds,
            workers=workers,
        )
    )

    def _resolve_doi_seed(spec: dict) -> ManifestItem:
        return resolve_doi_with_fallbacks(
            spec["doi"],
            source="topic-seed",
            tags=[*spec.get("tags", []), "topic-seed"],
            pdf_urls=spec.get("pdf_urls"),
            dry_run=dry_run,
            include_archive=False,
            include_paywall_guesses=False,
        )

    for start in range(0, len(TOPIC_DOI_SEEDS), doi_batch_size):
        batch = TOPIC_DOI_SEEDS[start : start + doi_batch_size]
        batch_items = parallel_map(_resolve_doi_seed, batch, workers=workers)
        results.extend(batch_items)
        if on_doi_batch and batch_items:
            on_doi_batch(batch_items)

    for spec in TOPIC_METADATA_SEEDS:
        results.append(
            ManifestItem(
                id=spec["id"],
                title=spec["title"],
                authors=spec.get("authors", []),
                year=spec.get("year"),
                source=spec.get("source", "topic-seed"),
                url=spec["url"],
                localPath="",
                contentType="text/html",
                status="metadata_only",
                tags=list(spec.get("tags", [])),
                abstract=spec.get("abstract"),
            )
        )

    return results
