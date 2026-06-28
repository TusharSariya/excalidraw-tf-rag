"""Resilient judge backend over the Antigravity CLI (`agy`).

Built for unattended GATE-3 judging when Vertex billing is dead. Hardened against
the failure modes that have bitten this project before:
  - non-TTY stdout drop (agy GH #76)  -> always run under a real PTY
  - transient rate-limit / empty / timeout  -> exponential backoff retry
  - judge cache-poisoning (transient frozen as grade-0)  -> failures stay UNCACHED
  - systemic auth/account death mid-run  -> FATAL abort (not silent grade-0 sweep)
  - crash mid-run losing all work  -> incremental ATOMIC cache writes + resume

Determinism was measured at 6/6 stable (temp-0), so retries are safe to repeat.
"""
from __future__ import annotations

import json
import os
import pty
import re
import select
import subprocess
import sys
import time

# --- output classification --------------------------------------------------
_AUTH_PAT = re.compile(
    r"unauthenticated|not authenticated|please (log|sign).?in|permission denied|"
    r"forbidden|invalid.?grant|account .*(delet|suspend|disabl|terminat)|"
    r"\b401\b|\b403\b|credentials? (expired|invalid)|re.?authenticate",
    re.I,
)
_RATE_PAT = re.compile(
    r"rate.?limit|resource exhausted|quota|too many requests|\b429\b|"
    r"try again later|overloaded|temporarily unavailable|\b503\b|backoff",
    re.I,
)
_ANSI = re.compile(r"\x1B\[[0-9;?]*[A-Za-z]")


class JudgeFatal(Exception):
    """Systemic auth/account failure — abort the whole run loudly."""


def validate_model(model: str, *, timeout: float = 40.0) -> None:
    """agy SILENTLY falls back to its default model on an unknown --model string
    (verified: a bogus name still returns grades). So a typo would judge the whole
    pool with the wrong model undetected. Refuse to start unless the exact model
    string appears in `agy models`."""
    raw, _rc, timed_out = _pty_run(["agy", "models"], timeout)
    if timed_out:
        raise JudgeFatal("`agy models` timed out — cannot verify judge model")
    # Strip braille spinner frames + "Fetching available models..." progress text
    # that \r-collapse onto the first model line.
    clean = re.sub(r"[⠀-⣿]", "", raw)
    clean = re.sub(r"Fetching available models\.*", "", clean)
    listed = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if model not in listed:
        raise JudgeFatal(
            f"model {model!r} not in `agy models` (agy would silently use its "
            f"DEFAULT). Available: {[m for m in listed if 'Gemini' in m or 'Claude' in m]}"
        )


def _pty_run(args: list[str], timeout: float) -> tuple[str, int, bool]:
    """Run a command under a pseudo-terminal so agy doesn't drop stdout."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    buf = b""
    t0 = time.time()
    timed_out = False
    while True:
        if time.time() - t0 > timeout:
            proc.kill()
            timed_out = True
            break
        r, _, _ = select.select([master], [], [], 1.0)
        if master in r:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
        if proc.poll() is not None:
            while True:  # drain
                r, _, _ = select.select([master], [], [], 0.2)
                if master in r:
                    try:
                        c = os.read(master, 4096)
                    except OSError:
                        c = b""
                    if not c:
                        break
                    buf += c
                else:
                    break
            break
    try:
        os.close(master)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    txt = _ANSI.sub("", buf.decode("utf-8", "replace")).replace("\r", "")
    return txt, (proc.returncode if proc.returncode is not None else -1), timed_out


def _classify(raw: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if _AUTH_PAT.search(raw):
        return "fatal"
    if _RATE_PAT.search(raw):
        return "ratelimit"
    if not raw.strip():
        return "empty"
    if not re.search(r'"grade"|\b[0-3]\b', raw):
        return "parsefail"
    return "ok"


def parse_grade(raw: str) -> tuple[int | None, str]:
    m = re.search(r'\{[^{}]*"grade"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            return max(0, min(3, int(o["grade"]))), str(o.get("reason", ""))[:200]
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
    d = re.search(r"\b([0-3])\b", raw)
    if d:
        return int(d.group(1)), raw[:200]
    return None, raw[:200]


def judge_one(prompt: str, model: str, *, max_retries: int = 5,
              timeout: float = 120.0, log=print) -> tuple[int | None, str, str]:
    """Return (grade|None, reason_or_status, final_status).

    grade is None ONLY when every retry was exhausted on a transient error —
    the caller must leave that hole UNCACHED (never write grade-0). A systemic
    auth/account failure raises JudgeFatal instead.
    """
    delay = 2.0
    status = "?"
    for attempt in range(1, max_retries + 1):
        raw, _rc, timed_out = _pty_run(["agy", "-p", prompt, "--model", model], timeout)
        status = _classify(raw, timed_out)
        if status == "ok":
            g, reason = parse_grade(raw)
            if g is not None:
                return g, reason, "ok"
            status = "parsefail"  # looked ok but unparseable -> treat as transient
        if status == "fatal":
            raise JudgeFatal(raw.strip()[:300])
        # transient: empty / ratelimit / timeout / parsefail -> backoff + retry
        wait = min(delay * (4.0 if status == "ratelimit" else 1.0), 90.0)
        if attempt < max_retries:
            log(f"    retry {attempt}/{max_retries} ({status}); sleep {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 90.0)
    return None, status, status  # exhausted -> UNCACHED


# --- atomic cache -----------------------------------------------------------
def load_cache(path: str) -> dict:
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache_atomic(cache: dict, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_judge(holes, *, model: str, cache_path: str, key_prefix: str,
              breaker: int = 25, max_retries: int = 5, flush_every: int = 1,
              log=print) -> dict:
    """holes: iterable of (case_id, doc_id, prompt). Resumable + crash-safe."""
    validate_model(model)  # refuse to start on a silent-fallback model name
    cache = load_cache(cache_path)
    judged = skipped = uncached = 0
    fail_streak = 0
    holes = list(holes)
    log(f"run_judge: {len(holes)} holes, model={model!r}, breaker={breaker}, "
        f"cache={cache_path} (existing {len(cache)})")
    for i, (case_id, doc_id, prompt) in enumerate(holes, 1):
        key = f"{key_prefix}:{case_id}:{doc_id}"
        if key in cache:
            skipped += 1
            continue
        try:
            grade, reason, status = judge_one(prompt, model, max_retries=max_retries, log=log)
        except JudgeFatal as e:
            save_cache_atomic(cache, cache_path)
            raise SystemExit(
                f"\nFATAL (auth/account) after {judged} judged, {len(cache)} cached: {e}\n"
                f"Cache saved; rerun resumes. Likely Antigravity flagged the account."
            ) from e
        if grade is None:  # transient exhausted -> UNCACHED (no poison)
            uncached += 1
            fail_streak += 1
            log(f"  [{i}/{len(holes)}] UNCACHED ({status}) streak={fail_streak} {case_id}:{doc_id}")
            if fail_streak >= breaker:
                save_cache_atomic(cache, cache_path)
                raise SystemExit(
                    f"\nCIRCUIT BREAKER: {fail_streak} consecutive failures ({status}) "
                    f"after {judged} judged — systemic (rate-limit wall / agy down). "
                    f"Cache saved; rerun resumes."
                )
            continue
        fail_streak = 0
        cache[key] = {"grade": grade, "reason": reason}
        judged += 1
        if judged % flush_every == 0:
            save_cache_atomic(cache, cache_path)
        if i % 10 == 0 or i == len(holes):
            log(f"  [{i}/{len(holes)}] judged={judged} skipped={skipped} uncached={uncached}")
    save_cache_atomic(cache, cache_path)
    log(f"DONE judged={judged} skipped={skipped} uncached={uncached} cache={len(cache)}")
    return {"judged": judged, "skipped": skipped, "uncached": uncached, "cache_size": len(cache)}


if __name__ == "__main__":
    # smoke: judge one ad-hoc (query, title) via agy
    q = sys.argv[1] if len(sys.argv) > 1 else "network simplex rank assignment"
    model = sys.argv[2] if len(sys.argv) > 2 else "Gemini 3.1 Pro (Low)"
    prompt = (
        'Grade 0-3 relevance. Respond ONLY JSON {"grade": <0-3>, "reason": "<=20 words"}\n'
        f"QUERY: {q}\nTITLE: A Technique for Drawing Directed Graphs"
    )
    print(judge_one(prompt, model))
