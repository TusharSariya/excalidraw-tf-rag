from __future__ import annotations

from rag_common.litmap import mainpath as mp


def test_drop_temporal_violations_drops_future_citation():
    # B (2020) "cites" A (2025) -- impossible, drop it. C (2021) cites D (2019) is fine.
    edges = [("B", "A"), ("C", "D")]
    years = {"A": 2025, "B": 2020, "C": 2021, "D": 2019}
    kept, warnings = mp.drop_temporal_violations(edges, years)
    assert kept == [("C", "D")]
    assert warnings


def test_drop_temporal_violations_unknown_year_passes_through():
    edges = [("B", "A")]
    kept, warnings = mp.drop_temporal_violations(edges, years={"A": 2020})  # B's year unknown
    assert kept == edges
    assert warnings == []


def test_find_cycle_detects_simple_cycle():
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    cycle = mp._find_cycle(edges)
    assert cycle is not None
    assert len(cycle) == 3


def test_find_cycle_none_on_dag():
    edges = [("a", "b"), ("b", "c")]
    assert mp._find_cycle(edges) is None


def test_break_cycles_removes_lowest_weight_edge_and_warns():
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    weight = {("a", "b"): 5.0, ("b", "c"): 5.0, ("c", "a"): 0.1}
    remaining, warnings = mp.break_cycles(edges, weight)
    assert ("c", "a") not in remaining
    assert mp._find_cycle(remaining) is None
    assert any("broke cycle" in w for w in warnings)


def test_break_cycles_no_hang_on_pathological_graph():
    # a big cycle plus chords -- must terminate well under the iteration cap.
    n = 30
    edges = [(f"n{i}", f"n{(i + 1) % n}") for i in range(n)]
    edges += [(f"n{i}", f"n{(i + 5) % n}") for i in range(n)]
    remaining, warnings = mp.break_cycles(edges, max_iterations=1000)
    assert mp._find_cycle(remaining) is None


def test_break_cycles_already_acyclic_is_a_noop():
    edges = [("a", "b"), ("b", "c")]
    remaining, warnings = mp.break_cycles(edges)
    assert remaining == edges
    assert warnings == []


def test_spc_diamond_hand_computed():
    # Diamond flow graph A->B, A->C, B->D, C->D.
    # from_source = {A:1, B:1, C:1, D:2}; to_sink = {A:2, B:1, C:1, D:1}.
    # SPC(A->B)=1, SPC(A->C)=1, SPC(B->D)=1, SPC(C->D)=1 (each edge carries exactly one
    # of the two A-to-D paths).
    flow_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    spc = mp.spc_weights(flow_edges)
    assert spc[("A", "B")] == 1
    assert spc[("A", "C")] == 1
    assert spc[("B", "D")] == 1
    assert spc[("C", "D")] == 1


def test_spc_linear_chain():
    flow_edges = [("A", "B"), ("B", "C"), ("C", "D")]
    spc = mp.spc_weights(flow_edges)
    assert spc == {("A", "B"): 1, ("B", "C"): 1, ("C", "D"): 1}


def test_spc_empty_is_empty():
    assert mp.spc_weights([]) == {}


def test_extract_main_path_diamond_picks_a_full_path():
    flow_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    spc = mp.spc_weights(flow_edges)
    path = mp.extract_main_path(flow_edges, spc)
    assert path[0] == "A"
    assert path[-1] == "D"
    assert len(path) == 3  # A -> (B or C) -> D


def test_build_main_path_empty_input():
    result = mp.build_main_path([])
    assert result["schema_version"] == 1
    assert result["main_path"] == []
    assert result["edges"] == []


def test_build_main_path_single_edge_no_cycle():
    result = mp.build_main_path([("newer", "older")])
    assert result["main_path"] == ["older", "newer"]
    assert result["edges"] == [{"src": "older", "dst": "newer", "spc": 1}]
    assert result["warnings"] == []


def test_build_main_path_with_cycle_warns_and_terminates():
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    result = mp.build_main_path(edges)
    assert result["main_path"]  # produced something
    assert any("cycle" in w for w in result["warnings"])


def test_build_main_path_schema_keys():
    result = mp.build_main_path([("a", "b"), ("b", "c")])
    for key in ("schema_version", "main_path", "edges", "node_count", "edge_count", "warnings"):
        assert key in result
