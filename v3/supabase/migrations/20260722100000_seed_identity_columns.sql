-- Add four new identity columns to public.seeds and wire them into EVERY place that
-- enumerates identity fields. Missing any one of them makes the capability-gated
-- fan-out silently no-op for that identity — see 20260716223300_capability_gated_fanout.sql
-- for why a silent no-op is the exact failure class this schema exists to prevent.
--
-- New identities:
--   dblp_id        : DBLP record key.
--   openreview_id  : OpenReview forum/submission id.
--   anthology_id   : ACL Anthology paper id.
--   ia_identifier  : Internet Archive item identifier.
--
-- Each is `text null` with a unique (corpus, <col>) constraint, matching the existing
-- openalex_id / doi / arxiv_id identities.

-- 1. Columns + per-corpus uniqueness. Idempotent-friendly.
alter table public.seeds add column if not exists dblp_id       text;
alter table public.seeds add column if not exists openreview_id text;
alter table public.seeds add column if not exists anthology_id  text;
alter table public.seeds add column if not exists ia_identifier text;

alter table public.seeds
    add constraint seeds_corpus_dblp_id_key       unique (corpus, dblp_id);
alter table public.seeds
    add constraint seeds_corpus_openreview_id_key unique (corpus, openreview_id);
alter table public.seeds
    add constraint seeds_corpus_anthology_id_key  unique (corpus, anthology_id);
alter table public.seeds
    add constraint seeds_corpus_ia_identifier_key unique (corpus, ia_identifier);

-- 2. harvest_filter_is_valid: extend the allowed-field IN-list. A typo'd field name in
--    a subscription filter must still fail LOUDLY at configuration time.
create or replace function public.harvest_filter_is_valid(p_filter jsonb)
returns boolean
language sql
immutable
as $$
    select p_filter is null              -- null = unconditional (back-compat)
        or (
            jsonb_typeof(p_filter -> 'any_of') = 'array'
            and jsonb_array_length(p_filter -> 'any_of') > 0
            and (
                select bool_and(x in (
                    'arxiv_id', 'doi', 'openalex_id', 'title',
                    'dblp_id', 'openreview_id', 'anthology_id', 'ia_identifier'
                ))
                from jsonb_array_elements_text(p_filter -> 'any_of') as t(x)
            )
        )
$$;

-- 3. seed_satisfies: add a branch per new field.
create or replace function public.seed_satisfies(s public.seeds, p_filter jsonb)
returns boolean
language sql
stable
as $$
    select p_filter is null
        or exists (
            select 1
            from jsonb_array_elements_text(p_filter -> 'any_of') as t(field)
            where (field = 'arxiv_id'      and s.arxiv_id      is not null)
               or (field = 'doi'           and s.doi           is not null)
               or (field = 'openalex_id'   and s.openalex_id   is not null)
               or (field = 'title'         and s.title         is not null)
               or (field = 'dblp_id'       and s.dblp_id       is not null)
               or (field = 'openreview_id' and s.openreview_id is not null)
               or (field = 'anthology_id'  and s.anthology_id  is not null)
               or (field = 'ia_identifier' and s.ia_identifier is not null)
        )
$$;

-- 4. seeds_identity_monotonic: add a coalesce line per new field. Identity is MONOTONIC
--    (NULL -> value, never cleared or changed) — this is what bounds the cascade.
create or replace function public.seeds_identity_monotonic() returns trigger
language plpgsql as $$
begin
    new.doi           := coalesce(old.doi, new.doi);
    new.arxiv_id      := coalesce(old.arxiv_id, new.arxiv_id);
    new.openalex_id   := coalesce(old.openalex_id, new.openalex_id);
    new.title         := coalesce(old.title, new.title);
    new.dblp_id       := coalesce(old.dblp_id, new.dblp_id);
    new.openreview_id := coalesce(old.openreview_id, new.openreview_id);
    new.anthology_id  := coalesce(old.anthology_id, new.anthology_id);
    new.ia_identifier := coalesce(old.ia_identifier, new.ia_identifier);
    return new;
end $$;

-- 5. seeds_identity_changed: drop + recreate to add the 4 fields to the WHEN clause. A
--    WHEN clause cannot be ALTERed in place, so the trigger is dropped and recreated.
drop trigger if exists seeds_identity_changed on public.seeds;
create trigger seeds_identity_changed
after update on public.seeds
for each row
when (
    old.doi           is distinct from new.doi
    or old.arxiv_id      is distinct from new.arxiv_id
    or old.openalex_id   is distinct from new.openalex_id
    or old.title         is distinct from new.title
    or old.dblp_id       is distinct from new.dblp_id
    or old.openreview_id is distinct from new.openreview_id
    or old.anthology_id  is distinct from new.anthology_id
    or old.ia_identifier is distinct from new.ia_identifier
)
execute function public.publish_seed_trigger();
