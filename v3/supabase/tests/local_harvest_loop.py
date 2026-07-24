# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
# ]
# ///
"""LOCAL live-stack smoke harness: drain every harvest source against a real DB.

Drives the whole pipeline end to end against a DISPOSABLE local `supabase start`
stack by shelling out to the same commands a human would run:

  1. `supabase db reset`                      (rebuild schema from migrations,
     then run supabase/seed.sql — the LOCAL/CI prod-parity setup that installs
     the pgmq_public.* wrappers + service_role grants prod gets from the
     dashboard Queues integration; never applied to prod)
  2. `uv run v3/seed/seed.py v3/seed/seeds/harvest-smoke`   (fan seeds out to the
     per-source harvest queues)
  3. for each requested source: `uv run harvest-work --source <key>`, which
     drains that source's queue and exits.

Then, for each source, it inspects the DB directly. A source PASSES iff:
    * >= 1 row in source_<key> with status = 'succeeded', AND
    * 0 rows in source_<key> with status = 'error', AND
    * the backing pgmq live table pgmq.q_harvest_<key> is empty (queue drained).
If the source's required API env is unset (e.g. UNPAYWALL_EMAIL, CORE_API_KEY)
the source is SKIPPED, not failed. A crash in one source's worker is caught and
recorded as that source's FAILURE — it never aborts the run.

Prints a per-source pass/fail/skip table and exits non-zero if any source FAILED.

DESTRUCTIVE: `supabase db reset` drops and rebuilds the local database. As a
guard (borrowed verbatim from e2e_fanout.py) it refuses to run unless
SUPABASE_DB_URL points at a loopback host; set E2E_ALLOW_NONLOCAL=1 to override
for a deliberately-disposable remote DB.

Run it:

    SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \\
    uv run v3/supabase/tests/local_harvest_loop.py                 # all sources
    uv run v3/supabase/tests/local_harvest_loop.py --source arxiv  # just one
    uv run v3/supabase/tests/local_harvest_loop.py --skip core,osf # all but two

SUPABASE_URL / SUPABASE_SERVICE_KEY (needed by the seeder) always point at the
LOCAL stack this harness just reset: they are read from `supabase status -o env`
(the running local stack), falling back to the standard local demo values. They
are DELIBERATELY not read from v3/harvest/.env — that file is the prod deploy
config, and seeding it would (a) leave the local queues empty so every source
reports "no succeeded rows", and (b) write throwaway smoke seeds into prod. The
resolved SUPABASE_URL is guarded to a loopback host for the same reason
SUPABASE_DB_URL is (see `_guard_disposable`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import psycopg

# ---- Layout -----------------------------------------------------------------

HERE = Path(__file__).resolve().parent  # v3/supabase/tests
SUPABASE_DIR = HERE.parent  # v3/supabase
V3_DIR = SUPABASE_DIR.parent  # v3
HARVEST_DIR = V3_DIR / "harvest"
SEED_SCRIPT = V3_DIR / "seed" / "seed.py"
SEED_DIR = V3_DIR / "seed" / "seeds" / "harvest-smoke"

# All 12 sources this harness knows how to smoke, in a stable order.
ALL_SOURCES = [
    "unpaywall",
    "crossref",
    "dblp",
    "openreview",
    "acl_anthology",
    "opencitations",
    "datacite",
    "core",
    "internet_archive",
    "zenodo",
    "figshare",
    "osf",
]

# Per-source required env. A source whose required var is unset is SKIPPED (its
# worker can't authenticate, so a failure there would be noise, not signal).
# Sources absent from this map have no hard env requirement.
REQUIRED_ENV: dict[str, list[str]] = {
    "unpaywall": ["UNPAYWALL_EMAIL"],
    "core": ["CORE_API_KEY"],
}

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}

# Standard local `supabase start` demo credentials — used only as a last resort
# when neither the environment nor v3/harvest/.env supplies them.
DEMO_URL = "http://127.0.0.1:54321"
DEMO_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0."
    "EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
)


# ---- Guards / env -----------------------------------------------------------


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: ${name} is required")
    return value


def _guard_disposable(db_url: str) -> None:
    """Refuse to touch a non-loopback DB unless explicitly overridden.

    This test purges real queues and truncates real seeds; running it against
    staging/prod would wipe live data. Only a disposable ephemeral DB is safe.
    """
    host = (urlsplit(db_url).hostname or "").lower()
    if host in LOOPBACK_HOSTS:
        return
    if os.environ.get("E2E_ALLOW_NONLOCAL") == "1":
        print(f"warning: E2E_ALLOW_NONLOCAL=1 — running destructively against host {host!r}")
        return
    sys.exit(
        f"refusing to run: SUPABASE_DB_URL host {host!r} is not loopback. This test is "
        "destructive (purges real queues, truncates seeds) and is only safe on a "
        "disposable ephemeral DB. Set E2E_ALLOW_NONLOCAL=1 to override for a throwaway DB."
    )


def _local_stack_env() -> dict[str, str]:
    """Read the running LOCAL stack's API URL + service key from `supabase status`.

    `supabase status -o env` emits API_URL / SERVICE_ROLE_KEY for the stack this
    harness just reset — the authoritative local target, independent of ports or
    of whatever v3/harvest/.env (the prod config) holds. Returns {} if the CLI is
    unavailable or the fields are absent, and the caller falls back to the demo
    constants.
    """
    try:
        proc = subprocess.run(
            ["supabase", "status", "-o", "env"],
            cwd=str(SUPABASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    parsed: dict[str, str] = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip()] = v.strip().strip('"').strip("'")
    out: dict[str, str] = {}
    if parsed.get("API_URL"):
        out["SUPABASE_URL"] = parsed["API_URL"]
    if parsed.get("SERVICE_ROLE_KEY"):
        out["SUPABASE_SERVICE_KEY"] = parsed["SERVICE_ROLE_KEY"]
    return out


def _resolve_seeder_env() -> dict[str, str]:
    """SUPABASE_URL / SUPABASE_SERVICE_KEY for the seeder, pinned to the LOCAL stack.

    Resolution order: an explicit loopback SUPABASE_URL in the environment, then
    the running local stack (`supabase status -o env`), then the demo constants.
    v3/harvest/.env is intentionally never consulted — it points at prod, and this
    harness is destructive against the DB it seeds. The resolved URL is guarded to
    a loopback host (honouring E2E_ALLOW_NONLOCAL) so a stray remote SUPABASE_URL
    in the environment can never redirect the seed at a live project.
    """
    status_env = _local_stack_env()
    url = os.environ.get("SUPABASE_URL") or status_env.get("SUPABASE_URL") or DEMO_URL
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY")
        or status_env.get("SUPABASE_SERVICE_KEY")
        or DEMO_SERVICE_KEY
    )
    _guard_seeder_loopback(url)
    return {"SUPABASE_URL": url, "SUPABASE_SERVICE_KEY": key}


def _guard_seeder_loopback(url: str) -> None:
    """Refuse to seed a non-loopback SUPABASE_URL (mirrors _guard_disposable)."""
    host = (urlsplit(url).hostname or "").lower()
    if host in LOOPBACK_HOSTS:
        return
    if os.environ.get("E2E_ALLOW_NONLOCAL") == "1":
        print(f"warning: E2E_ALLOW_NONLOCAL=1 — seeding non-loopback SUPABASE_URL host {host!r}")
        return
    sys.exit(
        f"refusing to run: SUPABASE_URL host {host!r} is not loopback. This harness resets "
        "and seeds a DISPOSABLE local stack; seeding a remote project would write throwaway "
        "smoke seeds into it and leave the local queues empty. Unset SUPABASE_URL (it is "
        "resolved from `supabase status`), or set E2E_ALLOW_NONLOCAL=1 for a throwaway remote."
    )


# ---- Subprocess steps -------------------------------------------------------


class StepError(RuntimeError):
    """A shelled-out step exited non-zero (captured for reporting)."""


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a command, streaming its output; raise StepError on non-zero exit."""
    printable = " ".join(cmd)
    print(f"\n$ (cd {cwd}) {printable}")
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, cwd=str(cwd), env=full_env, check=False)
    if proc.returncode != 0:
        raise StepError(f"`{printable}` exited {proc.returncode}")


def db_reset() -> None:
    # `db reset` re-applies every migration AND runs supabase/seed.sql (config.toml
    # [db.seed]), which installs the pgmq_public wrappers + service_role grants — so
    # there is no separate shim step to apply here.
    _run(["supabase", "db", "reset"], cwd=SUPABASE_DIR)


def wait_for_postgrest(
    seeder_env: dict[str, str], *, attempts: int = 60, delay: float = 3.0
) -> None:
    """Block until PostgREST has reloaded its schema cache after `db reset`.

    `supabase db reset` restarts the stack's containers; for a few seconds after,
    PostgREST answers every request with HTTP 503 / PGRST002 ("Could not query the
    database for the schema cache"). The seeder and workers both go through
    PostgREST, so seeding immediately after the reset races that reload and the
    whole run dies on the first request. Nudge a reload, then poll a trivial query
    until it returns 200 (or give up after ~3 min).
    """
    url = seeder_env.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = seeder_env.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        raise StepError("wait_for_postgrest needs SUPABASE_URL and SUPABASE_SERVICE_KEY")

    probe = f"{url.rstrip('/')}/rest/v1/queue_subscriptions?select=source&limit=1"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    print(f"\n$ waiting for PostgREST schema cache at {url}")
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(probe, headers=headers)  # noqa: S310 (fixed local URL)
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    print(f"  PostgREST ready after {i} poll(s)")
                    return
        except urllib.error.HTTPError as e:
            if e.code not in (503, 500):
                raise
        except urllib.error.URLError:
            pass  # container/port not up yet
        time.sleep(delay)
    raise StepError(f"PostgREST not ready after {attempts} polls (~{int(attempts * delay)}s)")


def seed(seeder_env: dict[str, str]) -> None:
    _run(["uv", "run", str(SEED_SCRIPT), str(SEED_DIR)], cwd=V3_DIR, env=seeder_env)


def drain_source(key: str, supabase_env: dict[str, str]) -> None:
    """Run harvest-work for one source (drain-then-exit). Raises on non-zero.

    `supabase_env` pins SUPABASE_URL / SUPABASE_SERVICE_KEY at the LOCAL stack the
    harness just reset. It MUST be passed: harvest-work runs in HARVEST_DIR, where
    the worker's pydantic settings would otherwise read v3/harvest/.env (the prod
    project) and drain a remote queue — or fail because the remote lacks the local
    source's queue. As an env var it overrides the .env value; source API keys not
    in this dict (UNPAYWALL_EMAIL, CORE_API_KEY, …) still resolve from .env.
    """
    _run(["uv", "run", "harvest-work", "--source", key], cwd=HARVEST_DIR, env=supabase_env)


# ---- DB verification --------------------------------------------------------


def _source_status_counts(conn: psycopg.Connection, key: str) -> dict[str, int]:
    """Map status -> row count in source_<key>. {} if the table doesn't exist."""
    table = f"source_{key}"
    with conn.cursor() as cur:
        cur.execute(
            "select to_regclass(%s)",
            (f"public.{table}",),
        )
        if cur.fetchone()[0] is None:
            raise StepError(f"table public.{table} does not exist")
        cur.execute(f"select status, count(*) from public.{table} group by status")  # noqa: S608
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def _queue_depth(conn: psycopg.Connection, key: str) -> int:
    """Rows remaining in the pgmq live table pgmq.q_harvest_<key>."""
    q_table = f"pgmq.q_harvest_{key}"
    with conn.cursor() as cur:
        cur.execute("select to_regclass(%s)", (q_table,))
        if cur.fetchone()[0] is None:
            raise StepError(f"queue table {q_table} does not exist")
        cur.execute(f"select count(*) from {q_table}")  # noqa: S608 (fixed identifier)
        return int(cur.fetchone()[0])


# ---- Result model -----------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


class Result:
    def __init__(self, key: str, verdict: str, detail: str) -> None:
        self.key = key
        self.verdict = verdict
        self.detail = detail


def _missing_env(key: str) -> list[str]:
    return [v for v in REQUIRED_ENV.get(key, []) if not os.environ.get(v)]


def evaluate_source(conn: psycopg.Connection, key: str, supabase_env: dict[str, str]) -> Result:
    """Drain one source and judge it. Never raises — crashes become FAILs."""
    missing = _missing_env(key)
    if missing:
        return Result(key, SKIP, f"required env unset: {', '.join(missing)}")

    try:
        drain_source(key, supabase_env)
    except StepError as e:
        return Result(key, FAIL, f"worker error: {e}")
    except Exception as e:  # noqa: BLE001 — resilience: one source must not abort the run
        return Result(key, FAIL, f"worker crashed: {type(e).__name__}: {e}")

    try:
        counts = _source_status_counts(conn, key)
        depth = _queue_depth(conn, key)
    except StepError as e:
        return Result(key, FAIL, str(e))
    except Exception as e:  # noqa: BLE001
        return Result(key, FAIL, f"verify crashed: {type(e).__name__}: {e}")

    succeeded = counts.get("succeeded", 0)
    errors = counts.get("error", 0)
    summary = (
        f"succeeded={succeeded} error={errors} queue_depth={depth} all_status={counts or '{}'}"
    )
    if succeeded >= 1 and errors == 0 and depth == 0:
        return Result(key, PASS, summary)
    reasons = []
    if succeeded < 1:
        reasons.append("no succeeded rows")
    if errors > 0:
        reasons.append(f"{errors} error rows")
    if depth > 0:
        reasons.append(f"queue not drained ({depth} left)")
    return Result(key, FAIL, f"{'; '.join(reasons)} [{summary}]")


# ---- Reporting --------------------------------------------------------------


def print_table(results: list[Result]) -> None:
    print("\n" + "=" * 72)
    print("HARVEST SMOKE RESULTS")
    print("=" * 72)
    print(f"{'source':<18}  {'verdict':<7}  detail")
    print(f"{'-' * 18}  {'-' * 7}  {'-' * 40}")
    for r in results:
        print(f"{r.key:<18}  {r.verdict:<7}  {r.detail}")
    n_pass = sum(1 for r in results if r.verdict == PASS)
    n_fail = sum(1 for r in results if r.verdict == FAIL)
    n_skip = sum(1 for r in results if r.verdict == SKIP)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped ({len(results)} sources)")


# ---- Main -------------------------------------------------------------------


def _select_sources(source: str | None, skip: str | None) -> list[str]:
    if source:
        if source not in ALL_SOURCES:
            sys.exit(f"error: --source {source!r} is not one of {ALL_SOURCES}")
        chosen = [source]
    else:
        chosen = list(ALL_SOURCES)
    if skip:
        skipset = {s.strip() for s in skip.split(",") if s.strip()}
        unknown = skipset - set(ALL_SOURCES)
        if unknown:
            sys.exit(f"error: --skip names unknown source(s): {sorted(unknown)}")
        chosen = [s for s in chosen if s not in skipset]
    if not chosen:
        sys.exit("error: no sources left to run after --skip")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source",
        help=f"run just this one source (default: all {len(ALL_SOURCES)})",
    )
    ap.add_argument(
        "--skip",
        help="comma-separated source keys to skip, e.g. --skip core,osf",
    )
    args = ap.parse_args()

    db_url = _require_env("SUPABASE_DB_URL")
    _guard_disposable(db_url)
    sources = _select_sources(args.source, args.skip)
    seeder_env = _resolve_seeder_env()

    print(f"harvest smoke: {len(sources)} source(s): {', '.join(sources)}")

    # --- Shared setup: reset (migrations + seed.sql) -> wait for PostgREST ->
    #     seed the smoke corpus. A failure here aborts (there is nothing
    #     meaningful to test without a seeded stack).
    try:
        db_reset()
        wait_for_postgrest(seeder_env)
        seed(seeder_env)
    except StepError as e:
        print(f"\nFATAL: setup failed: {e}", file=sys.stderr)
        return 2

    # --- Per-source drain + verify over one shared connection.
    results: list[Result] = []
    with psycopg.connect(db_url, autocommit=True) as conn:
        for key in sources:
            print(f"\n--- source: {key} ---")
            results.append(evaluate_source(conn, key, seeder_env))

    print_table(results)
    return 1 if any(r.verdict == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
