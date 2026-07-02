#!/usr/bin/env python3
"""Multi-model panel labeling for the judge-calibration sheets (D7 amendment).

Instead of a single human hand-labeling the cluster-label sanity sheets, a panel of
diverse-family models produces the reference labels (LLM-jury pattern, PoLL /
Verga 2024), with the human arbitrating only the cases the panel cannot settle:

  round 1: every voter labels every case independently (blind, no cross-talk)
  round 2: on non-unanimous cases only, each voter sees the strongest rationale
           from the opposing side (unattributed) and may defend or flip
  tally:   >= 4/5 post-debate majority -> consensus; 3/2 splits -> human arbitration
  fill:    consensus/arbitrated labels written into the sheet's `Label:` lines,
           ready for `eval judge-calibrate --score`

deepseek-* is deliberately NOT on the panel: it is the judge being calibrated.
Claude's votes are read from claude_votes.json next to each sheet (written blind,
in-session, before any CLI voter runs); its round-2 revisits happen in-session too.

Usage (from repo root; stdlib only):
  python3 scripts/panel_calibrate.py vote   --tool rag-literature-rag
  python3 scripts/panel_calibrate.py debate --tool rag-literature-rag
  python3 scripts/panel_calibrate.py tally  --tool rag-literature-rag
  python3 scripts/panel_calibrate.py fill   --tool rag-literature-rag \
      --arbitration arbitration_answers.json   # only if tally escalated splits

State lives in data/eval/judge_calibration/panel_votes.json per tool (resume-safe:
already-recorded votes are never re-queried).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = ("rag-literature-rag", "graph-layout-rag")

OPENCODE_VOTERS = {
    "glm-5.2": "opencode-go/glm-5.2",
    "qwen3.7-max": "opencode-go/qwen3.7-max",
    "kimi-k2.7": "opencode-go/kimi-k2.7-code",
}
CODEX_VOTER = "codex"
CLAUDE_VOTER = "claude-fable-5"
ALL_VOTERS = (*OPENCODE_VOTERS, CODEX_VOTER, CLAUDE_VOTER)

ANSWER_INSTRUCTIONS = (
    "Respond with a single JSON object and nothing else: "
    '{"answer": "yes" or "no", "rationale": "<one sentence>"}'
)


def calib_dir(tool: str) -> Path:
    return REPO_ROOT / tool / "data" / "eval" / "judge_calibration"


def parse_sheet(sheet_path: Path) -> tuple[str, list[dict[str, str]]]:
    """Return (rubric, cases). A case is {'id', 'content'} where content is the
    cluster-label + member-titles block shown to every voter."""
    text = sheet_path.read_text(encoding="utf-8")
    rubric_match = re.search(r"## Rubric\n\n(.*?)\n\n---", text, re.DOTALL)
    if not rubric_match:
        raise SystemExit(f"No rubric found in {sheet_path}")
    rubric = rubric_match.group(1).strip()
    cases = []
    for match in re.finditer(r"## Case: (\S+)\n\n(.*?)\n\nLabel:", text, re.DOTALL):
        cases.append({"id": match.group(1), "content": match.group(2).strip()})
    if not cases:
        raise SystemExit(f"No cases found in {sheet_path}")
    return rubric, cases


def parse_vote_text(text: str) -> tuple[str, str] | None:
    """Extract (label, rationale) from a voter's reply; JSON first, regex fallback."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            answer = str(obj.get("answer", "")).strip().lower()
            if answer in ("yes", "no"):
                return answer, str(obj.get("rationale", "")).strip()
        except json.JSONDecodeError:
            pass
    tokens = re.findall(r"\b(yes|no)\b", text.lower())
    if tokens:
        return tokens[-1], text.strip()[:300]
    return None


def ask_opencode(model: str, prompt: str, timeout: float = 240.0) -> str:
    proc = subprocess.run(
        ["opencode", "run", "-m", model, "--format", "json", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencode exit {proc.returncode}: {(proc.stderr or '')[:300]}")
    parts = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            text = (event.get("part") or {}).get("text")
            if text:
                parts.append(text)
    out = "".join(parts).strip()
    if not out:
        raise RuntimeError(f"empty opencode response from {model}")
    return out


def ask_codex(prompt: str, timeout: float = 300.0) -> str:
    proc = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exit {proc.returncode}: {(proc.stderr or '')[:300]}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("empty codex response")
    return out


def ask_voter(voter: str, prompt: str) -> str:
    if voter in OPENCODE_VOTERS:
        return ask_opencode(OPENCODE_VOTERS[voter], prompt)
    if voter == CODEX_VOTER:
        return ask_codex(prompt)
    raise ValueError(f"{voter} is not a CLI voter")


def load_state(tool: str) -> dict:
    path = calib_dir(tool) / "panel_votes.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"round1": {}, "round2": {}}


def save_state(tool: str, state: dict) -> None:
    path = calib_dir(tool) / "panel_votes.json"
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def merge_claude_votes(tool: str, state: dict, cases: list[dict[str, str]]) -> None:
    claude_path = calib_dir(tool) / "claude_votes.json"
    if not claude_path.exists():
        raise SystemExit(f"Missing {claude_path} — Claude's blind votes must exist first.")
    claude = json.loads(claude_path.read_text(encoding="utf-8"))["votes"]
    for case in cases:
        cid = case["id"]
        if cid not in claude:
            raise SystemExit(f"Claude vote missing for {cid} in {claude_path}")
        state["round1"].setdefault(cid, {})[CLAUDE_VOTER] = {
            "label": claude[cid]["label"],
            "rationale": claude[cid].get("rationale", ""),
        }


def cmd_vote(tool: str, workers: int) -> None:
    rubric, cases = parse_sheet(calib_dir(tool) / "cluster_label_sheet.md")
    state = load_state(tool)
    merge_claude_votes(tool, state, cases)

    pending = [
        (case, voter)
        for case in cases
        for voter in (*OPENCODE_VOTERS, CODEX_VOTER)
        if voter not in state["round1"].get(case["id"], {})
    ]
    print(f"[{tool}] round 1: {len(pending)} votes to collect ({len(cases)} cases)")

    def collect(job: tuple[dict[str, str], str]) -> tuple[str, str, dict | None, str]:
        case, voter = job
        prompt = f"{rubric}\n\n{case['content']}\n\n{ANSWER_INSTRUCTIONS}"
        try:
            reply = ask_voter(voter, prompt)
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return case["id"], voter, None, str(exc)
        parsed = parse_vote_text(reply)
        if parsed is None:
            return case["id"], voter, None, f"unparseable reply: {reply[:200]}"
        return case["id"], voter, {"label": parsed[0], "rationale": parsed[1]}, ""

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, voter, vote, err in pool.map(collect, pending):
            if vote is None:
                failures += 1
                print(f"  FAIL {cid} × {voter}: {err}", file=sys.stderr)
                continue
            state["round1"].setdefault(cid, {})[voter] = vote
            save_state(tool, state)
            print(f"  {cid:12s} {voter:12s} -> {vote['label']}")

    done = sum(len(v) for v in state["round1"].values())
    print(
        f"[{tool}] round 1 votes recorded: {done}/{len(cases) * len(ALL_VOTERS)}"
        f" ({failures} failures — re-run `vote` to retry)"
    )


def final_votes(state: dict, cid: str) -> dict[str, dict]:
    votes = dict(state["round1"].get(cid, {}))
    votes.update(state["round2"].get(cid, {}))
    return votes


def split_cases(state: dict) -> list[str]:
    out = []
    for cid in state["round1"]:
        labels = {v["label"] for v in final_votes(state, cid).values()}
        if len(labels) > 1:
            out.append(cid)
    return sorted(out)


def strongest_rationale(votes: dict[str, dict], side: str) -> str:
    candidates = [v["rationale"] for v in votes.values() if v["label"] == side and v["rationale"]]
    return max(candidates, key=len) if candidates else f"(no rationale given for '{side}')"


def cmd_debate(tool: str, workers: int) -> None:
    rubric, cases = parse_sheet(calib_dir(tool) / "cluster_label_sheet.md")
    case_by_id = {c["id"]: c for c in cases}
    state = load_state(tool)
    contested = split_cases(state)
    print(f"[{tool}] round 2: {len(contested)} contested case(s): {', '.join(contested) or '-'}")

    jobs = []
    for cid in contested:
        votes = state["round1"][cid]
        for voter in (*OPENCODE_VOTERS, CODEX_VOTER):
            if voter in votes and voter not in state["round2"].get(cid, {}):
                jobs.append((cid, voter))

    def revisit(job: tuple[str, str]) -> tuple[str, str, dict | None, str]:
        cid, voter = job
        votes = state["round1"][cid]
        own = votes[voter]
        other_side = "no" if own["label"] == "yes" else "yes"
        counter = strongest_rationale(votes, other_side)
        prompt = (
            f"{rubric}\n\n{case_by_id[cid]['content']}\n\n"
            f"You previously answered '{own['label']}' with this rationale: "
            f"{own['rationale'] or '(none)'}\n"
            f"Another reviewer answered '{other_side}' and argued: {counter}\n\n"
            "Reconsider carefully. Keep your answer only if your rationale survives the "
            f"counter-argument; flip if it does not.\n\n{ANSWER_INSTRUCTIONS}"
        )
        try:
            reply = ask_voter(voter, prompt)
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return cid, voter, None, str(exc)
        parsed = parse_vote_text(reply)
        if parsed is None:
            return cid, voter, None, f"unparseable reply: {reply[:200]}"
        return cid, voter, {"label": parsed[0], "rationale": parsed[1]}, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, voter, vote, err in pool.map(revisit, jobs):
            if vote is None:
                print(f"  FAIL {cid} × {voter}: {err}", file=sys.stderr)
                continue
            flipped = vote["label"] != state["round1"][cid][voter]["label"]
            state["round2"].setdefault(cid, {})[voter] = vote
            save_state(tool, state)
            print(f"  {cid:12s} {voter:12s} -> {vote['label']}{' (flipped)' if flipped else ''}")

    print(
        f"[{tool}] round 2 done. Claude's revisits (if any) must be merged into "
        f"panel_votes.json round2 by the orchestrator, then run `tally`."
    )


def consensus(votes: dict[str, dict]) -> tuple[str | None, int, int]:
    yes = sum(1 for v in votes.values() if v["label"] == "yes")
    no = len(votes) - yes
    if yes >= 4:
        return "yes", yes, no
    if no >= 4:
        return "no", yes, no
    return None, yes, no


def cmd_tally(tool: str) -> None:
    state = load_state(tool)
    rows, escalations = [], []
    for cid in sorted(state["round1"], key=lambda c: int(c.rsplit("-", 1)[1])):
        votes = final_votes(state, cid)
        label, yes, no = consensus(votes)
        rows.append((cid, label, yes, no))
        if label is None:
            escalations.append(cid)

    unanimous = sum(1 for _, label, yes, no in rows if label and (yes == 0 or no == 0))
    print(
        f"[{tool}] tally: {len(rows)} cases, {unanimous} unanimous, "
        f"{len(escalations)} escalated to human"
    )
    for cid, label, yes, no in rows:
        print(f"  {cid:12s} yes={yes} no={no} -> {label or 'ESCALATE'}")

    per_model = {v: 0 for v in ALL_VOTERS}
    decided = [(cid, label) for cid, label, _, _ in rows if label]
    for cid, label in decided:
        for voter, vote in final_votes(state, cid).items():
            if vote["label"] != label:
                per_model[voter] += 1
    print("  deviations from consensus:", ", ".join(f"{v}={n}" for v, n in per_model.items()))

    if escalations:
        sheet = calib_dir(tool) / "arbitration_sheet.md"
        _, cases = parse_sheet(calib_dir(tool) / "cluster_label_sheet.md")
        case_by_id = {c["id"]: c for c in cases}
        lines = ["# Human arbitration — panel 3/2 splits\n"]
        for cid in escalations:
            votes = final_votes(state, cid)
            lines += [
                f"## {cid}\n",
                case_by_id[cid]["content"],
                "",
                f"YES side: {strongest_rationale(votes, 'yes')}",
                f"NO side: {strongest_rationale(votes, 'no')}",
                "",
                "Decision: ",
                "",
            ]
        sheet.write_text("\n".join(lines), encoding="utf-8")
        print(f"  wrote {sheet} — decide these, then `fill --arbitration <json>`")


def cmd_fill(tool: str, arbitration_path: Path | None) -> None:
    state = load_state(tool)
    arbitration = {}
    if arbitration_path:
        arbitration = json.loads(arbitration_path.read_text(encoding="utf-8"))

    labels = {}
    for cid in state["round1"]:
        label, _, _ = consensus(final_votes(state, cid))
        if label is None:
            label = arbitration.get(cid)
        if label not in ("yes", "no"):
            raise SystemExit(f"{cid} unresolved — arbitrate it first (tally wrote the sheet).")
        labels[cid] = label

    sheet_path = calib_dir(tool) / "cluster_label_sheet.md"
    text = sheet_path.read_text(encoding="utf-8")

    def fill_case(match: re.Match) -> str:
        return f"## Case: {match.group(1)}{match.group(2)}Label: {labels[match.group(1)]}"

    filled, n = re.subn(r"## Case: (\S+)(.*?)Label:[^\n]*", fill_case, text, flags=re.DOTALL)
    if n != len(labels):
        raise SystemExit(f"filled {n} cases but have {len(labels)} labels — sheet drifted?")
    sheet_path.write_text(filled, encoding="utf-8")
    print(f"[{tool}] wrote {n} consensus labels into {sheet_path}")
    print(
        f"  next: cd {tool} && uv run {tool} eval judge-calibrate "
        f"--score data/eval/judge_calibration/cluster_label_sheet.md "
        f"--profile cuda-qwen0.6b-1024"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("vote", "debate", "tally", "fill"))
    parser.add_argument("--tool", choices=TOOLS, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--arbitration",
        type=Path,
        default=None,
        help="JSON {case_id: 'yes'|'no'} for escalated splits (fill only)",
    )
    args = parser.parse_args()

    if args.command == "vote":
        cmd_vote(args.tool, args.workers)
    elif args.command == "debate":
        cmd_debate(args.tool, args.workers)
    elif args.command == "tally":
        cmd_tally(args.tool)
    else:
        cmd_fill(args.tool, args.arbitration)


if __name__ == "__main__":
    main()
