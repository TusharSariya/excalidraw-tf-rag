# /// script
# requires-python = ">=3.11"
# dependencies = ["supabase>=2.31", "pydantic>=2"]
# ///
"""Bulk-seed public.seeds from a TOML file and fan each row out to the harvest queues.

The TOML-driven successor to v3/supabase/scripts/seed.py (one seed per invocation,
via flags) and the bulk `harvest-seed` the harvest README documents but never
shipped. Like that stopgap it is NON-DESTRUCTIVE: it only inserts, reuses the row
when a paper is already seeded in the corpus, then re-publishes. Safe against staging.

Reaches the DB over PostgREST with the service key (public.seeds is RLS-locked with
no policies — backend/service-role only). For each seed it inserts (or reuses) the
row, then calls public.publish_seed(paper_id), which stamps + pg_jsonschema-validates
the v1 envelope before enqueuing to each active queue. Inserting also fires the seeds
AFTER INSERT trigger, which publishes too; the explicit call is idempotent
(harvest_dispatch is claimed per-generation) and is what re-publishes an
already-existing seed.

TOML shape:

    corpus = "graph-layout-rag"        # default corpus for every seed

    [[seed]]
    title = "ForceAtlas2, ..."
    doi   = "10.1371/journal.pone.0098679"

    [[seed]]
    arxiv_id = "2311.00533"
    corpus   = "some-other-corpus"     # optional per-seed override

Each seed needs a corpus (default, per-seed, or --corpus) and at least one of
doi / arxiv_id / openalex_id / title.

Run it:

    uv run v3/seed/seed.py                        # seeds.toml next to this script
    uv run v3/seed/seed.py path/to/other.toml
    uv run v3/seed/seed.py --dry-run              # validate + print, no writes
    uv run v3/seed/seed.py --corpus my-corpus     # override corpus for all

Env: SUPABASE_URL + SUPABASE_SERVICE_KEY, read from the environment or, if unset,
from v3/harvest/.env.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator
from supabase import create_client

logger = logging.getLogger("seed")

HERE = Path(__file__).resolve().parent
DEFAULT_SEEDS = HERE / "seeds"  # a directory of <corpus>/<theme>.toml files
HARVEST_ENV = HERE.parent / "harvest" / ".env"

# Identities with a UNIQUE (corpus, <id>) constraint — usable for dedup lookup.
# `title` has no such constraint, so a title-only seed cannot be deduped.
ID_FIELDS = (
    "openalex_id",
    "doi",
    "arxiv_id",
    "dblp_id",
    "openreview_id",
    "anthology_id",
    "ia_identifier",
)
ALL_ID_FIELDS = (*ID_FIELDS, "title")


class Seed(BaseModel):
    """Insert shape for public.seeds; mirrors harvest.persistence.models.Seed."""

    corpus: str
    openalex_id: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    title: str | None = None
    dblp_id: str | None = None
    openreview_id: str | None = None
    anthology_id: str | None = None
    ia_identifier: str | None = None

    @model_validator(mode="after")
    def at_least_one_identity(self) -> Seed:
        if not any(getattr(self, f) for f in ALL_ID_FIELDS):
            raise ValueError(
                "requires at least one of openalex_id, doi, arxiv_id, title, "
                "dblp_id, openreview_id, anthology_id, ia_identifier"
            )
        return self

    def ids(self) -> dict[str, str]:
        """The non-null identity fields, in insert order."""
        return {f: getattr(self, f) for f in ALL_ID_FIELDS if getattr(self, f)}


def load_env_file(path: Path) -> None:
    """Populate SUPABASE_* from a KEY=VALUE .env, without overriding the environment."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY") and key not in os.environ:
            os.environ[key] = val.strip().strip('"').strip("'")


def parse_seeds(toml_path: Path, corpus_override: str | None) -> list[Seed]:
    data = tomllib.loads(toml_path.read_text())
    default_corpus = data.get("corpus")
    raw = data.get("seed")
    if not raw:
        raise SystemExit(f"{toml_path}: no [[seed]] entries found")

    seeds: list[Seed] = []
    errors: list[str] = []
    for i, entry in enumerate(raw):
        corpus = corpus_override or entry.get("corpus") or default_corpus
        label = entry.get("title") or entry.get("doi") or entry.get("arxiv_id") or entry
        if not corpus:
            errors.append(
                f"seed[{i}] ({label}): no corpus (set top-level `corpus`, a per-seed `corpus`, or --corpus)"
            )
            continue
        try:
            seeds.append(Seed(corpus=corpus, **{k: entry.get(k) for k in ALL_ID_FIELDS}))
        except ValidationError as e:
            errors.append(f"seed[{i}] ({label}): {e.errors()[0]['msg']}")
    if errors:
        raise SystemExit("invalid seeds:\n  " + "\n  ".join(errors))
    return seeds


def load_seeds(path: Path, corpus_override: str | None) -> list[Seed]:
    """Parse one TOML file, or every ``*.toml`` under a directory (recursively).

    A directory holds ``<corpus>/<theme>.toml`` files, each with its own top-level
    ``corpus`` — that per-file corpus is what lets one run seed several corpora at
    once. ``--corpus`` still overrides them all.
    """
    if path.is_dir():
        files = sorted(path.glob("**/*.toml"))
        if not files:
            raise SystemExit(f"{path}: no .toml files found")
        seeds: list[Seed] = []
        for f in files:
            seeds.extend(parse_seeds(f, corpus_override))
        return seeds
    return parse_seeds(path, corpus_override)


def main() -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "toml_path",
        nargs="?",
        default=str(DEFAULT_SEEDS),
        help="a seed TOML file, or a directory of <corpus>/<theme>.toml (default: ./seeds)",
    )
    ap.add_argument("--corpus", help="override the corpus for every seed")
    ap.add_argument("--dry-run", action="store_true", help="validate and print; no DB writes")
    ap.add_argument(
        "--no-publish",
        action="store_true",
        help="insert only; skip the publish_seed RPC (the insert trigger still fans out)",
    )
    args = ap.parse_args()

    toml_path = Path(args.toml_path)
    if not toml_path.exists():
        print(f"no such seed file or directory: {toml_path}", file=sys.stderr)
        return 2

    seeds = load_seeds(toml_path, args.corpus)
    logger.info("parsed %d seed(s) from %s", len(seeds), toml_path)

    if args.dry_run:
        by_corpus: dict[str, int] = {}
        for s in seeds:
            by_corpus[s.corpus] = by_corpus.get(s.corpus, 0) + 1
        for corpus, n in sorted(by_corpus.items()):
            print(f"  {n:4}  {corpus}")
        print(
            f"--dry-run: {len(seeds)} seed(s) valid across {len(by_corpus)} corpus(es), no writes."
        )
        return 0

    load_env_file(HARVEST_ENV)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        print(f"set SUPABASE_URL and SUPABASE_SERVICE_KEY (env or {HARVEST_ENV})", file=sys.stderr)
        return 2
    client = create_client(url, key)

    inserted = reused = 0
    for s in seeds:
        ids = s.ids()
        lookup = {k: v for k, v in ids.items() if k in ID_FIELDS}
        row = None
        if lookup:
            q = client.table("seeds").select("paper_id").eq("corpus", s.corpus)
            for k, v in lookup.items():
                q = q.eq(k, v)
            rows = q.execute().data
            row = rows[0] if rows else None

        if row:
            paper_id = row["paper_id"]
            reused += 1
            label = "reused"
        else:
            new = client.table("seeds").insert({"corpus": s.corpus, **ids}).execute().data
            paper_id = new[0]["paper_id"]
            inserted += 1
            label = "inserted"

        desc = next((f"{k}={v}" for k, v in lookup.items()), f"title={s.title!r}")
        print(f"  {label:8} paper_id={paper_id}  {desc}")

        if not args.no_publish:
            client.schema("public").rpc("publish_seed", {"p_paper_id": paper_id}).execute()

    print(f"\n{inserted} inserted, {reused} reused ({len(seeds)} total).")
    if not args.no_publish:
        subs = (
            client.table("queue_subscriptions")
            .select("queue_name")
            .eq("active", True)
            .execute()
            .data
        )
        names = ", ".join(sub["queue_name"] for sub in subs)
        print(f"published to {len(subs)} active queue(s): {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
