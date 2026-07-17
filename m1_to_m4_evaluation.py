"""
M1-M4 Evaluation Harness
=========================

Implements the four metrics exactly as formally defined in Ramamurthy's
"IPIN - M1 to M4 draft" note, on top of the shared synthetic_layout_generator.py
(Method A / Method B construction, Node/Edge dataclasses, layout generation).

    M1  Entrance Reachability Ratio          |R_G(e)| / |Z|
    M2  Reachability Agreement (A vs B)       fraction of zones where
                                               r_A(z) == r_B(z)
    M3  Path-Length Consistency               1 - mean normalized path-length
                                               disagreement over jointly
                                               reachable zones (Z_AB)
    M4  Edge Soundness Validity Rate          fraction of edges not crossing
                                               a wall; reported as
                                               "undefined" (non-verifiable)
                                               when no wall geometry exists

Runs the parameter sweep specified in the note (layout geometry x
wall-condition x method hyperparameters), writes a tidy results table
(CSV), prints summary statistics, and renders the four suggested plots.

Author: Harita Kanuri (Nav-vera) + Ramamurthy Rallabandi — IPIN 2026 WCAL
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv
import heapq
import itertools

from synthetic_layout_generator import (
    Node, Edge, SyntheticLayout,
    generate_synthetic_layout,
    build_corridor_template_graph,
    build_centroid_proximity_graph,
    edge_crosses_any_wall,
)

EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Graph utilities: reachable set + weighted shortest paths (for M3)
# ─────────────────────────────────────────────────────────────────────────

def _adjacency(nodes: List[Node], edges: List[Edge]) -> Dict[str, List[Tuple[str, float]]]:
    adj: Dict[str, List[Tuple[str, float]]] = {n.id: [] for n in nodes}
    for e in edges:
        adj.setdefault(e.a, []).append((e.b, e.dist))
        adj.setdefault(e.b, []).append((e.a, e.dist))
    return adj


def reachable_set(nodes: List[Node], edges: List[Edge], source_id: str = "entrance") -> set:
    """BFS from source; returns the set of reached node ids."""
    adj = _adjacency(nodes, edges)
    seen = {source_id}
    stack = [source_id]
    while stack:
        cur = stack.pop()
        for nxt, _ in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def shortest_path_distances(nodes: List[Node], edges: List[Edge], source_id: str = "entrance") -> Dict[str, float]:
    """Dijkstra from source over weighted edges (Edge.dist). Returns dist map."""
    adj = _adjacency(nodes, edges)
    dist: Dict[str, float] = {n.id: float("inf") for n in nodes}
    dist[source_id] = 0.0
    pq = [(0.0, source_id)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ─────────────────────────────────────────────────────────────────────────
# M1 — Entrance Reachability Ratio
# ─────────────────────────────────────────────────────────────────────────

def compute_M1(nodes: List[Node], edges: List[Edge]) -> float:
    zone_ids = [n.id for n in nodes if n.kind == "zone"]
    if not zone_ids:
        return 0.0
    reached = reachable_set(nodes, edges)
    r = sum(1 for z in zone_ids if z in reached)
    return r / len(zone_ids)


# ─────────────────────────────────────────────────────────────────────────
# M2 — Reachability Agreement Between Construction Methods
# ─────────────────────────────────────────────────────────────────────────

def compute_M2(nodes_a: List[Node], edges_a: List[Edge],
               nodes_b: List[Node], edges_b: List[Edge],
               zone_ids: List[str]) -> float:
    reached_a = reachable_set(nodes_a, edges_a)
    reached_b = reachable_set(nodes_b, edges_b)
    agree = 0
    for z in zone_ids:
        r_a = 1 if z in reached_a else 0
        r_b = 1 if z in reached_b else 0
        if r_a == r_b:
            agree += 1
    return agree / len(zone_ids) if zone_ids else 0.0


# ─────────────────────────────────────────────────────────────────────────
# M3 — Path-Length Consistency for Commonly Reachable Zones
# ─────────────────────────────────────────────────────────────────────────

def compute_M3(nodes_a: List[Node], edges_a: List[Edge],
               nodes_b: List[Node], edges_b: List[Edge],
               zone_ids: List[str]) -> Optional[float]:
    reached_a = reachable_set(nodes_a, edges_a)
    reached_b = reachable_set(nodes_b, edges_b)
    z_ab = [z for z in zone_ids if z in reached_a and z in reached_b]
    if not z_ab:
        return None  # "undefined" per the spec

    dist_a = shortest_path_distances(nodes_a, edges_a)
    dist_b = shortest_path_distances(nodes_b, edges_b)

    disagreements = []
    for z in z_ab:
        da, db = dist_a.get(z, float("inf")), dist_b.get(z, float("inf"))
        denom = max(da, db, EPS)
        disagreements.append(abs(da - db) / denom)

    return 1 - (sum(disagreements) / len(disagreements))


# ─────────────────────────────────────────────────────────────────────────
# M4 — Edge Soundness Validity Rate
# ─────────────────────────────────────────────────────────────────────────

def compute_M4(nodes: List[Node], edges: List[Edge], layout: SyntheticLayout) -> Optional[float]:
    if not layout.walls:
        return None  # non-verifiable, per the spec — NOT the same as "sound"
    node_map = {n.id: n for n in nodes}
    if not edges:
        return None
    valid = 0
    for e in edges:
        n1, n2 = node_map[e.a], node_map[e.b]
        if not edge_crosses_any_wall(n1, n2, layout.walls):
            valid += 1
    return valid / len(edges)


# ─────────────────────────────────────────────────────────────────────────
# Parameter sweep
# ─────────────────────────────────────────────────────────────────────────

def run_sweep(out_csv: str = "m1_to_m4_results.csv"):
    rows_range = [2, 3, 4, 6]
    cols_range = [2, 4, 6]
    wall_conditions = [False, True]
    seeds = [1, 2, 3]                     # repeatability across random variation
    corridor_rows_options = [3]           # Method A hyperparameter
    k_neighbors_options = [2, 3, 4]        # Method B hyperparameter

    fieldnames = [
        "n_rows", "n_cols", "with_walls", "seed",
        "corridor_rows", "k_neighbors",
        "n_zones", "nodes_A", "edges_A", "nodes_B", "edges_B",
        "M1_A", "M1_B", "M2", "M3", "M4_A", "M4_B",
    ]
    results = []

    for n_rows, n_cols, with_walls, seed, corridor_rows, k in itertools.product(
        rows_range, cols_range, wall_conditions, seeds, corridor_rows_options, k_neighbors_options
    ):
        layout = generate_synthetic_layout(
            n_rows=n_rows, n_cols=n_cols, with_walls=with_walls, seed=seed
        )
        zone_ids = [z.id for z in layout.zones]

        nodes_a, edges_a = build_corridor_template_graph(
            layout, n_corridor_rows=corridor_rows, enforce_walls=with_walls
        )
        nodes_b, edges_b = build_centroid_proximity_graph(
            layout, k_neighbors=k, enforce_walls=with_walls
        )

        m1_a = compute_M1(nodes_a, edges_a)
        m1_b = compute_M1(nodes_b, edges_b)
        m2 = compute_M2(nodes_a, edges_a, nodes_b, edges_b, zone_ids)
        m3 = compute_M3(nodes_a, edges_a, nodes_b, edges_b, zone_ids)
        m4_a = compute_M4(nodes_a, edges_a, layout)
        m4_b = compute_M4(nodes_b, edges_b, layout)

        results.append({
            "n_rows": n_rows, "n_cols": n_cols, "with_walls": with_walls, "seed": seed,
            "corridor_rows": corridor_rows, "k_neighbors": k,
            "n_zones": len(zone_ids),
            "nodes_A": len(nodes_a), "edges_A": len(edges_a),
            "nodes_B": len(nodes_b), "edges_B": len(edges_b),
            "M1_A": round(m1_a, 4), "M1_B": round(m1_b, 4),
            "M2": round(m2, 4),
            "M3": "undefined" if m3 is None else round(m3, 4),
            "M4_A": "undefined" if m4_a is None else round(m4_a, 4),
            "M4_B": "undefined" if m4_b is None else round(m4_b, 4),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return results


def summarize(results: List[Dict]) -> None:
    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"Total configurations run: {len(results)}\n")

    for wall_cond in (False, True):
        subset = [r for r in results if r["with_walls"] == wall_cond]
        print(f"--- with_walls = {wall_cond} (n={len(subset)}) ---")
        print(f"  Mean M1 (Method A, corridor-template):   {_mean([r['M1_A'] for r in subset]):.3f}")
        print(f"  Mean M1 (Method B, centroid-proximity):  {_mean([r['M1_B'] for r in subset]):.3f}")
        print(f"  Mean M2 (reachability agreement):        {_mean([r['M2'] for r in subset]):.3f}")
        m3_vals = [r["M3"] for r in subset if r["M3"] != "undefined"]
        print(f"  Mean M3 (path-length consistency):       {_mean(m3_vals):.3f}  (n_defined={len(m3_vals)}/{len(subset)})")
        if wall_cond:
            m4a_vals = [r["M4_A"] for r in subset if r["M4_A"] != "undefined"]
            m4b_vals = [r["M4_B"] for r in subset if r["M4_B"] != "undefined"]
            print(f"  Mean M4 (Method A edge validity):        {_mean(m4a_vals):.3f}")
            print(f"  Mean M4 (Method B edge validity):        {_mean(m4b_vals):.3f}")
        else:
            print(f"  M4: non-verifiable (no wall geometry present)")
        print()


if __name__ == "__main__":
    results = run_sweep()
    summarize(results)
    print(f"Full results table written to m1_to_m4_results.csv ({len(results)} rows)")
