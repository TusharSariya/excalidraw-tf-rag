-- pipeline-v2 P1 schema: work-state ledger + manifest migration table.
--
-- work_state is the single source of truth for "did stage X finish for
-- doc_id in corpus Y, and how". The v1 architecture destroyed 2,266 docs
-- because a 'degraded' outcome could be read back as "done" and indexed
-- anyway. Two structural guards against that regressing here:
--
--   1. The CHECK constraint below only allows the three real outcomes.
--      There is no 'pending' or 'in_progress' state to accidentally
--      treat as terminal-success.
--   2. indexable_work_state is the *only* sanctioned read path for "is
--      this doc/stage indexable" — it hard-filters state = 'ok'. Callers
--      must never query work_state directly for that question.

CREATE TABLE IF NOT EXISTS work_state (
    doc_id             text        NOT NULL,
    corpus             text        NOT NULL,
    content_sha256     text        NOT NULL,
    stage              text        NOT NULL,
    stage_fingerprint  text        NOT NULL,
    attempt            integer     NOT NULL DEFAULT 1,
    state              text        NOT NULL CHECK (state IN ('ok', 'degraded', 'failed')),
    reason             text,
    artifact_uri       text,
    run_id             text,
    updated_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, corpus, stage)
);

-- Dead-letter lookups (get_dead_letters / claim_retry) always filter by
-- stage (+ optionally corpus) and a non-ok state; index that access path.
CREATE INDEX IF NOT EXISTS work_state_dead_letter_idx
    ON work_state (stage, corpus)
    WHERE state IN ('degraded', 'failed');

-- The only sanctioned "is this row indexable" read path. See header note.
CREATE OR REPLACE VIEW indexable_work_state AS
SELECT *
FROM work_state
WHERE state = 'ok';

-- v1 manifest.json migrated in: full record kept as jsonb (payload), plus
-- the columns callers actually filter/join on pulled out for indexing.
-- corpus is part of the identity because graph-layout-rag and
-- rag-literature-rag each have their own manifest/id space.
CREATE TABLE IF NOT EXISTS manifest_item (
    id          text        NOT NULL,
    corpus      text        NOT NULL,
    doi         text,
    status      text,
    sha256      text,
    payload     jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (id, corpus)
);

CREATE INDEX IF NOT EXISTS manifest_item_corpus_idx ON manifest_item (corpus);
CREATE INDEX IF NOT EXISTS manifest_item_doi_idx ON manifest_item (doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS manifest_item_sha256_idx ON manifest_item (sha256) WHERE sha256 IS NOT NULL;
