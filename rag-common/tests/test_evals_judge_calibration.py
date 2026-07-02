from __future__ import annotations

from rag_common.evals import judge_calibration as jc


def test_cohen_kappa_perfect_agreement():
    a = {"c1": 1, "c2": 2, "c3": 0}
    b = {"c1": 1, "c2": 2, "c3": 0}
    assert jc.cohen_kappa(a, b) == 1.0


def test_cohen_kappa_hand_computed():
    # Classic textbook example (2 raters, 2 categories, 10 items):
    # rater A: 6 yes / 4 no ; rater B: 5 yes / 5 no ; agree on 3 yes + 2 no = 5/10 = 0.5 po
    # pe = (6/10 * 5/10) + (4/10 * 5/10) = 0.3 + 0.2 = 0.5 -> kappa = (0.5-0.5)/(1-0.5) = 0.0
    a = {f"c{i}": v for i, v in enumerate(["y", "y", "y", "y", "y", "y", "n", "n", "n", "n"])}
    b = {f"c{i}": v for i, v in enumerate(["y", "y", "y", "n", "n", "n", "n", "y", "y", "n"])}
    kappa = jc.cohen_kappa(a, b)
    assert abs(kappa - 0.0) < 1e-9


def test_cohen_kappa_no_shared_cases_is_zero():
    assert jc.cohen_kappa({"c1": 1}, {"c2": 1}) == 0.0


def test_cohen_kappa_ignores_unlabeled_cases_not_in_both():
    a = {"c1": 1, "c2": 2, "c3": 1}
    b = {"c1": 1, "c2": 2}
    assert jc.cohen_kappa(a, b) == 1.0


def test_labeling_sheet_round_trip(tmp_path):
    cases = [
        {"id": "case1", "content": "Query: foo\nDoc: bar"},
        {"id": "case2", "content": "Query: baz\nDoc: qux"},
    ]
    out = tmp_path / "sheet.md"
    jc.build_labeling_sheet(cases, out, rubric="Grade 0-3.", n=10)
    text = out.read_text()
    assert "## Case: case1" in text
    assert "## Case: case2" in text
    assert "Query: foo" in text
    assert "Grade 0-3." in text

    # Simulate the user hand-filling labels.
    filled = text.replace(
        "## Case: case1\n\nQuery: foo\nDoc: bar\n\nLabel: ",
        "## Case: case1\n\nQuery: foo\nDoc: bar\n\nLabel: 3",
    )
    filled = filled.replace(
        "## Case: case2\n\nQuery: baz\nDoc: qux\n\nLabel: ",
        "## Case: case2\n\nQuery: baz\nDoc: qux\n\nLabel: 1",
    )
    out.write_text(filled)

    labels = jc.parse_labeling_sheet(out)
    assert labels == {"case1": "3", "case2": "1"}


def test_labeling_sheet_blank_labels_omitted(tmp_path):
    cases = [{"id": "case1", "content": "x"}]
    out = tmp_path / "sheet.md"
    jc.build_labeling_sheet(cases, out, rubric="r", n=10)
    labels = jc.parse_labeling_sheet(out)
    assert labels == {}


def test_labeling_sheet_sample_respects_n(tmp_path):
    cases = [{"id": f"c{i}", "content": str(i)} for i in range(50)]
    out = tmp_path / "sheet.md"
    jc.build_labeling_sheet(cases, out, rubric="r", n=5, seed=1)
    labels_text = out.read_text()
    case_headers = [line for line in labels_text.splitlines() if line.startswith("## Case:")]
    assert len(case_headers) == 5


def test_score_agreement_and_decide_pass():
    human = {f"c{i}": (i % 4) for i in range(20)}
    cases = [{"id": f"c{i}"} for i in range(20)]

    def judge_fn(case):
        return human[case["id"]]  # perfect judge

    result = jc.score_agreement(human, cases, judge_fn)
    assert result.kappa == 1.0
    assert result.passed is True
    assert result.disagreements == []
    decision = jc.decide(result)
    assert decision["outcome"] == "PASS"


def test_score_agreement_and_decide_fail_then_retry():
    human = {f"c{i}": (i % 4) for i in range(20)}
    cases = [{"id": f"c{i}"} for i in range(20)]

    def bad_judge_fn(case):
        return 0  # always says 0 -> disagrees often

    result = jc.score_agreement(human, cases, bad_judge_fn)
    assert result.passed is False
    assert len(result.disagreements) > 0

    first = jc.decide(result, already_retried=False)
    assert first["outcome"] == "REVISE_AND_RETRY"

    second = jc.decide(result, already_retried=True)
    assert second["outcome"] == "HUMAN_JUDGED"


def test_score_agreement_only_uses_labeled_cases():
    human = {"c1": "yes"}
    cases = [{"id": "c1"}, {"id": "c2"}]
    calls = []

    def judge_fn(case):
        calls.append(case["id"])
        return "yes"

    result = jc.score_agreement(human, cases, judge_fn)
    assert calls == ["c1"]
    assert result.n_cases == 1
