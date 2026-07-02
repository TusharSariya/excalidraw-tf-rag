"""Judge calibration harness — run BEFORE any judge-gated metric is trusted (decisions-log
item 19 / D7): the user hand-labels a sample of cases against a written rubric, the
candidate judge grades the same cases blind, and agreement is measured with Cohen's kappa.

Three pieces, used in sequence:

1. `build_labeling_sheet()` — sample N cases, write a markdown sheet with an empty label
   column for the user to hand-fill.
2. `parse_labeling_sheet()` / `score_agreement()` — read the filled sheet back, run the
   judge callable over the same cases (cache-keyed like `eval/judge.py`'s `judge_pool` —
   this module does not own a cache itself; callers pass a cache-keyed callable, e.g. a
   thin wrapper around a tool's `judge_model()`/`judge_pool` plumbing), and compute kappa +
   a per-case disagreement list.
3. Fail-path logic (`decide()`): kappa >= 0.7 -> PASS, use the judge for this metric.
   kappa < 0.7 -> the caller's contract is "revise the judge prompt, re-test once; if it's
   still < 0.7, that gate stays human-judged" (decisions-log item 19) — this module can't
   revise a prompt itself, so `decide()` just classifies the outcome and names the required
   next action so the CLI/human doesn't have to re-derive the threshold logic.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MIN_KAPPA = 0.70

_HEADER_RE = re.compile(r"^##\s*Case:\s*(?P<case_id>\S+)\s*$")
_LABEL_RE = re.compile(r"^Label:\s*(?P<label>\S*)\s*$")


def sample_cases(
    cases: Sequence[dict[str, Any]],
    *,
    n: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministic sample of `n` cases (or all of them if fewer). Each case dict must
    have an `id` key; other keys are caller-defined content to render in the sheet."""
    if len(cases) <= n:
        return list(cases)
    rng = random.Random(seed)
    return rng.sample(list(cases), n)


def build_labeling_sheet(
    cases: Sequence[dict[str, Any]],
    out_path: Path,
    *,
    rubric: str,
    content_key: str = "content",
    n: int = 30,
    seed: int = 42,
) -> Path:
    """Write a markdown labeling sheet for the user to hand-fill.

    Each case gets a `## Case: <id>` block with its content and an empty `Label:` line —
    `parse_labeling_sheet` reads exactly this shape back. The rubric header is written
    verbatim so the user grades against the same criteria the judge prompt will use.
    """
    sample = sample_cases(cases, n=n, seed=seed)
    lines = [
        "# Judge calibration labeling sheet",
        "",
        "Fill in a label after `Label:` for every case below, then run "
        "`eval judge-calibrate --score <this file>`.",
        "",
        "## Rubric",
        "",
        rubric.strip(),
        "",
        "---",
        "",
    ]
    for case in sample:
        case_id = case["id"]
        content = str(case.get(content_key, ""))
        lines.append(f"## Case: {case_id}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("Label: ")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def parse_labeling_sheet(path: Path) -> dict[str, str]:
    """Read a filled labeling sheet back into `{case_id: label}` (label is the raw
    trimmed string after `Label:` — caller casts to int/whatever the rubric's scale is).
    Cases left blank are omitted (the user hasn't labeled them yet), not zero-filled."""
    text = Path(path).read_text(encoding="utf-8")
    labels: dict[str, str] = {}
    current_id: str | None = None
    for line in text.splitlines():
        header = _HEADER_RE.match(line.strip())
        if header:
            current_id = header.group("case_id")
            continue
        label_match = _LABEL_RE.match(line.strip())
        if label_match and current_id is not None:
            value = label_match.group("label").strip()
            if value:
                labels[current_id] = value
            current_id = None  # one label per case block
    return labels


def cohen_kappa(labels_a: dict[str, Any], labels_b: dict[str, Any]) -> float:
    """Cohen's kappa over the cases present in both label sets (arbitrary discrete
    categories — not restricted to binary). Returns `1.0` for empty/perfect-agreement
    trivial input (nothing to disagree on) and `0.0` if inputs share no cases."""
    shared = sorted(set(labels_a) & set(labels_b))
    n = len(shared)
    if n == 0:
        return 0.0
    pairs = [(str(labels_a[c]), str(labels_b[c])) for c in shared]
    po = sum(1 for a, b in pairs if a == b) / n

    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    from collections import Counter

    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    pe = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in categories)

    if pe >= 1.0:
        # Every rater always picked the same single category — no variance to explain,
        # kappa is undefined by the classic formula (0/0); agreement is total, so 1.0.
        return 1.0
    return (po - pe) / (1.0 - pe)


@dataclass
class AgreementResult:
    kappa: float
    n_cases: int
    disagreements: list[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kappa": round(self.kappa, 4),
            "n_cases": self.n_cases,
            "min_kappa": MIN_KAPPA,
            "passed": self.passed,
            "disagreements": self.disagreements,
        }


def score_agreement(
    human_labels: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    judge_fn: Callable[[dict[str, Any]], Any],
) -> AgreementResult:
    """Run `judge_fn` over every case that has a human label, compute kappa, and list
    per-case disagreements. `judge_fn` should be cache-keyed by the caller (mirrors
    `eval/judge.py:judge_pool`'s `{model}:{case_id}` cache key) — this module doesn't
    cache itself since the cache key/model choice is tool-specific plumbing."""
    labeled_cases = [c for c in cases if c["id"] in human_labels]
    judge_labels: dict[str, Any] = {c["id"]: judge_fn(c) for c in labeled_cases}

    kappa = cohen_kappa(human_labels, judge_labels)
    disagreements = [
        {"case_id": cid, "human": human_labels[cid], "judge": judge_labels[cid]}
        for cid in sorted(set(human_labels) & set(judge_labels))
        if str(human_labels[cid]) != str(judge_labels[cid])
    ]
    return AgreementResult(
        kappa=kappa,
        n_cases=len(labeled_cases),
        disagreements=disagreements,
        passed=kappa >= MIN_KAPPA,
    )


def decide(result: AgreementResult, *, already_retried: bool = False) -> dict[str, Any]:
    """Fail-path classification per decisions-log item 19: kappa>=0.7 -> PASS. Below
    that on the first attempt -> REVISE_AND_RETRY (revise the judge prompt, re-run once).
    Still below after a documented retry (`already_retried=True`) -> HUMAN_JUDGED (the
    metric stays human-labeled; this gate does not get an automated judge)."""
    if result.passed:
        return {"outcome": "PASS", "action": "judge is calibrated; use it for this metric."}
    if not already_retried:
        return {
            "outcome": "REVISE_AND_RETRY",
            "action": (
                f"kappa={result.kappa:.3f} < {MIN_KAPPA}: revise the judge prompt and "
                "re-run score_agreement() once more (pass already_retried=True on that call)."
            ),
        }
    return {
        "outcome": "HUMAN_JUDGED",
        "action": (
            f"kappa={result.kappa:.3f} < {MIN_KAPPA} after a prompt revision: this metric "
            "stays human-judged, do not gate promotion decisions on this judge for it."
        ),
    }
