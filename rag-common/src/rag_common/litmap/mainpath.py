"""Main-path analysis over a citation DAG (Hummon & Doreian, 1989) — the standard method
for tracing a topic's intellectual lineage: the single walk through the citation network
that accumulates the most "traffic".

**SPC (Search Path Count) choice, documented per D10:** SPC weights an edge by the number
of distinct source-to-sink paths that pass through it. It's the classic main-path traversal
weight (vs. SPLC/SPNP variants) because it needs only the DAG's topology — no edge-level
citation-context weighting the corpus doesn't have — and is cheap to compute exactly with
two linear DP passes over a topological order:

  `from_source(u)` = number of paths from any source to `u`
      = 1 if `u` is a source, else `sum(from_source(p) for p in predecessors(u))`
  `to_sink(v)` = number of paths from `v` to any sink
      = 1 if `v` is a sink, else `sum(to_sink(s) for s in successors(v))`
  `SPC(u -> v)` = `from_source(u) * to_sink(v)`

Path direction is knowledge *flow*, not citation direction: a citation edge `(src, dst)`
means `src` cites `dst`, i.e. `src` is the newer paper building on the older `dst`. The flow
of ideas runs the other way — from the foundational `dst` into the citing `src` — so the DAG
handed to `spc_weights`/`extract_main_path` is built on **flow edges** `(dst, src)`.

**Cycle handling runs before SPC, always** — real citation graphs have cycles (preprint vs.
published-version double-references, S2/OpenAlex extraction noise), and SPC is undefined on
a cyclic graph. Two passes, in order:
  1. `drop_temporal_violations` — when both endpoints' years are known, a citer can't be
     older than what it cites; drop those edges outright (data-quality signal, not a real
     cycle).
  2. `break_cycles` — anything left is broken by iteratively finding one cycle (DFS, white/
     gray/black coloring) and removing its lowest-weight edge, bounded by `max_iterations`
     so a pathological graph can never hang the CLI. Every break is recorded as a warning
     string surfaced in the JSON output, never silently.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

SCHEMA_VERSION = 1


def drop_temporal_violations(
    edges: list[tuple[str, str]],
    years: dict[str, int],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Drop citation edges `(src, dst)` where `year[src] < year[dst]` (citing something
    published later) whenever both years are known. Edges with an unknown year on either
    side pass through untouched — cycle-breaking is the backstop for those."""
    kept = []
    dropped = 0
    for src, dst in edges:
        y_src, y_dst = years.get(src), years.get(dst)
        if y_src is not None and y_dst is not None and y_src < y_dst:
            dropped += 1
            continue
        kept.append((src, dst))
    warnings = (
        [f"mainpath: dropped {dropped} temporally-invalid edge(s) (citer older than cited)"]
        if dropped
        else []
    )
    return kept, warnings


def _find_cycle(edges: list[tuple[str, str]]) -> list[tuple[str, str]] | None:
    """First cycle found by iterative DFS (white/gray/black coloring), as a list of edges,
    or `None` if the graph is already acyclic. Iterative (explicit stack) so it can't blow
    the recursion limit on a large citation subgraph."""
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    for u, v in edges:
        adj[u].append(v)
        for n in (u, v):
            if n not in seen_nodes:
                seen_nodes.add(n)
                nodes.append(n)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = defaultdict(int)

    for start in nodes:
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        path = [start]
        stack = [iter(adj[start])]
        while stack:
            try:
                v = next(stack[-1])
            except StopIteration:
                stack.pop()
                finished = path.pop()
                color[finished] = BLACK
                continue
            if color[v] == WHITE:
                color[v] = GRAY
                path.append(v)
                stack.append(iter(adj[v]))
            elif color[v] == GRAY:
                idx = path.index(v)
                cycle_nodes = path[idx:] + [v]
                return [(cycle_nodes[i], cycle_nodes[i + 1]) for i in range(len(cycle_nodes) - 1)]
            # BLACK: cross/forward edge in a DAG-of-visited-region, not part of a cycle.
    return None


def break_cycles(
    edges: list[tuple[str, str]],
    weight: dict[tuple[str, str], float] | None = None,
    *,
    max_iterations: int = 10_000,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Repeatedly find one cycle and drop its lowest-weight edge until the graph is
    acyclic or `max_iterations` is hit (never hangs). `weight` defaults every edge to 1.0,
    so with no weights supplied ties break arbitrarily-but-deterministically on edge order."""
    weight = weight or {}
    remaining = list(edges)
    warnings: list[str] = []
    for _ in range(max_iterations):
        cycle = _find_cycle(remaining)
        if cycle is None:
            return remaining, warnings
        worst = min(cycle, key=lambda e: weight.get(e, 1.0))
        remaining = [e for e in remaining if e != worst]
        warnings.append(
            f"mainpath: broke cycle by removing lowest-weight edge {worst[0]} -> {worst[1]}"
        )
    warnings.append(
        f"mainpath: cycle-breaking hit max_iterations={max_iterations}; graph may still contain cycles"
    )
    return remaining, warnings


def topo_sort(nodes: Iterable[str], fwd_adj: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm. Raises if a cycle remains — callers must run `break_cycles` first."""
    nodes = list(nodes)
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for _u, vs in fwd_adj.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    queue = deque(n for n in nodes if indeg.get(n, 0) == 0)
    order: list[str] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in fwd_adj.get(u, ()):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != len(nodes):
        raise RuntimeError(
            "mainpath: topo_sort found a remaining cycle — break_cycles must run first"
        )
    return order


def spc_weights(flow_edges: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Search Path Count per edge on an already-acyclic flow-direction edge list.

    Hand-computed toy example (see `test_litmap_mainpath.py`): a diamond
    `A->B, A->C, B->D, C->D` has `from_source = {A:1, B:1, C:1, D:2}`,
    `to_sink = {A:2, B:1, C:1, D:1}`, so `SPC(A->B)=1*1=1`, `SPC(A->C)=1*1=1`,
    `SPC(B->D)=1*1=1`, `SPC(C->D)=1*1=1` — every edge carries exactly one of the two
    A-to-D paths, which matches counting by hand.
    """
    if not flow_edges:
        return {}
    nodes = sorted({n for u, v in flow_edges for n in (u, v)})
    fwd_adj: dict[str, list[str]] = defaultdict(list)
    rev_adj: dict[str, list[str]] = defaultdict(list)
    for u, v in flow_edges:
        fwd_adj[u].append(v)
        rev_adj[v].append(u)
    order = topo_sort(nodes, fwd_adj)

    from_source: dict[str, int] = {}
    for u in order:
        preds = rev_adj.get(u, ())
        from_source[u] = sum(from_source[p] for p in preds) if preds else 1

    to_sink: dict[str, int] = {}
    for u in reversed(order):
        succs = fwd_adj.get(u, ())
        to_sink[u] = sum(to_sink[s] for s in succs) if succs else 1

    return {(u, v): from_source[u] * to_sink[v] for u, v in flow_edges}


def extract_main_path(
    flow_edges: list[tuple[str, str]],
    spc: dict[tuple[str, str], int],
) -> list[str]:
    """Greedy global main path: start at the source whose best outgoing SPC edge is
    largest, then repeatedly follow the highest-SPC unvisited successor until reaching a
    sink. `visited` guards against revisiting a node even though the input is a DAG."""
    if not flow_edges:
        return []
    fwd_adj: dict[str, list[str]] = defaultdict(list)
    has_incoming: set[str] = set()
    for u, v in flow_edges:
        fwd_adj[u].append(v)
        has_incoming.add(v)

    def _best_outgoing(n: str) -> int:
        return max((spc[(n, v)] for v in fwd_adj.get(n, ())), default=-1)

    # True sources only (no predecessor in the flow DAG) — every source's best-outgoing
    # SPC ties in a symmetric DAG (e.g. a diamond), so restricting to sources plus a
    # deterministic (sorted) tie-break keeps the walk reproducible instead of depending on
    # set iteration order.
    nodes_with_out = {u for u, _v in flow_edges}
    start_candidates = sorted(nodes_with_out - has_incoming) or sorted(nodes_with_out)
    start = max(start_candidates, key=lambda n: (_best_outgoing(n), n))

    path = [start]
    visited = {start}
    current = start
    while fwd_adj.get(current):
        candidates = [v for v in fwd_adj[current] if v not in visited]
        if not candidates:
            break
        nxt = max(candidates, key=lambda v: (spc[(current, v)], v))
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def build_main_path(
    citation_edges: list[tuple[str, str]],
    *,
    years: dict[str, int] | None = None,
) -> dict:
    """Full pipeline: temporal validation -> cycle-breaking -> SPC -> greedy main path.

    `citation_edges` are `(src, dst)` pairs in citation direction (`src` cites `dst`). The
    returned `main_path` is a list of node ids in flow order (oldest/foundational first).
    Empty input, or a topic subgraph with no citation edges, yields an empty path rather
    than raising — the caller (`map lineage`) falls back to listing the seed docs alone.
    """
    warnings: list[str] = []
    edges = list(citation_edges)

    if years:
        edges, w1 = drop_temporal_violations(edges, years)
        warnings += w1
    edges, w2 = break_cycles(edges)
    warnings += w2

    flow_edges = [(dst, src) for src, dst in edges]
    if not flow_edges:
        return {
            "schema_version": SCHEMA_VERSION,
            "main_path": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "warnings": warnings,
        }

    spc = spc_weights(flow_edges)
    path = extract_main_path(flow_edges, spc)
    return {
        "schema_version": SCHEMA_VERSION,
        "main_path": path,
        "edges": [{"src": u, "dst": v, "spc": spc[(u, v)]} for u, v in zip(path, path[1:])],
        "node_count": len({n for e in flow_edges for n in e}),
        "edge_count": len(flow_edges),
        "warnings": warnings,
    }
