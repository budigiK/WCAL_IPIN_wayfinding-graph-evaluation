"""
Synthetic Indoor Layout Generator + Two Graph-Construction Method Classes
==========================================================================

Purpose
-------
Generates parameterized synthetic indoor floor-plan layouts (a grid of
rectangular "zones" arranged in aisles, with an entrance point), then builds
a navigation graph over each layout using TWO different, independently
plausible construction strategies:

    Method A — Corridor-Template Construction
        Assumes fixed front/back corridors and evenly-spaced aisle rows at
        FRACTIONAL positions within the layout bounds (not hardcoded absolute
        coordinates, so it generalizes across layout sizes). Each zone is
        wired to its nearest corridor node.

    Method B — Centroid-Proximity Construction
        Places a node at each zone's centroid plus a row of centerline nodes
        along the zone, then connects zones to their k-nearest neighboring
        zone-centroids (no assumption of a global corridor structure).

Both methods are real, independently-motivated strategies for turning a set
of zone rectangles into a walkable navigation graph — this generator lets us
run them over a controlled parameter sweep (row/column count, aisle width,
entrance position, missing-wall-geometry condition) and measure how often
each produces a routable graph.

This is the synthetic counterpart to the production system's construction
logic, generalized and anonymized for external sharing. It contains no
proprietary code or company-specific data.

Author: Harita Kanuri (Nav-vera) — for IPIN 2026 WCAL collaboration
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import random
import heapq


# ─────────────────────────────────────────────────────────────────────────
# 1. Synthetic layout generation
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class SyntheticLayout:
    width: float
    height: float
    zones: List[Zone]
    entrance: Tuple[float, float]
    walls: List[Wall] = field(default_factory=list)   # empty by default —
                                                        # models the documented
                                                        # "no wall geometry"
                                                        # condition on
                                                        # generated maps


def generate_synthetic_layout(
    n_rows: int,
    n_cols: int,
    zone_w: float = 10.0,
    zone_h: float = 6.0,
    aisle_w: float = 4.0,
    row_gap: float = 8.0,
    margin: float = 6.0,
    with_walls: bool = False,
    seed: Optional[int] = None,
) -> SyntheticLayout:
    """
    Build a synthetic layout: n_rows x n_cols grid of rectangular zones,
    separated by aisle_w horizontally and row_gap vertically, with an
    entrance placed at the bottom-center of the layout.

    with_walls=False reproduces the documented "generator emits no wall
    geometry" condition. with_walls=True adds simple perimeter walls around
    each zone, so Method A/B's wall-crossing checks have something to test
    against (used for the soundness half of the evaluation).
    """
    if seed is not None:
        random.seed(seed)

    zones: List[Zone] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x1 = margin + c * (zone_w + aisle_w)
            y1 = margin + r * (zone_h + row_gap)
            x2 = x1 + zone_w
            y2 = y1 + zone_h
            zones.append(Zone(id=f"z_{r}_{c}", x1=x1, y1=y1, x2=x2, y2=y2))

    width = margin * 2 + n_cols * zone_w + (n_cols - 1) * aisle_w
    height = margin * 2 + n_rows * zone_h + (n_rows - 1) * row_gap
    entrance = (width / 2, margin / 2)

    walls: List[Wall] = []
    if with_walls:
        for z in zones:
            walls.extend([
                Wall(z.x1, z.y1, z.x2, z.y1),
                Wall(z.x2, z.y1, z.x2, z.y2),
                Wall(z.x2, z.y2, z.x1, z.y2),
                Wall(z.x1, z.y2, z.x1, z.y1),
            ])

    return SyntheticLayout(width=width, height=height, zones=zones,
                            entrance=entrance, walls=walls)


# ─────────────────────────────────────────────────────────────────────────
# 2. Shared graph primitives
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    x: float
    y: float
    kind: str = "corridor"   # "corridor" | "zone" | "entrance"


@dataclass
class Edge:
    a: str
    b: str
    dist: float


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    p1, p2, p3, p4 = (ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def edge_crosses_any_wall(n1: Node, n2: Node, walls: List[Wall]) -> bool:
    return any(
        segments_intersect(n1.x, n1.y, n2.x, n2.y, w.x1, w.y1, w.x2, w.y2)
        for w in walls
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. Method A — Corridor-Template Construction
# ─────────────────────────────────────────────────────────────────────────

def build_corridor_template_graph(
    layout: SyntheticLayout,
    n_corridor_rows: int = 3,
    front_frac: float = 0.88,
    back_frac: float = 0.12,
    enforce_walls: bool = False,
) -> Tuple[List[Node], List[Edge]]:
    """
    Fixed-template construction: assumes a front corridor and a back
    corridor at FRACTIONAL y-positions within the layout, plus evenly
    spaced intermediate aisle-corridor rows. Each zone centroid is wired
    to its nearest corridor node on the same "row band".

    This generalizes the production system's fixed-coordinate corridor
    logic (which hardcodes absolute y-values for a specific store size) to
    an arbitrary layout by expressing corridor positions as fractions of
    layout height — the underlying construction strategy is unchanged.
    """
    nodes: List[Node] = []
    edges: List[Edge] = []

    # Entrance node
    ex, ey = layout.entrance
    nodes.append(Node("entrance", ex, ey, kind="entrance"))

    # Corridor rows at fixed fractional y-positions (front, back, + evenly
    # spaced intermediate rows)
    fracs = [back_frac] + \
            [back_frac + (front_frac - back_frac) * i / (n_corridor_rows - 1)
             for i in range(1, n_corridor_rows - 1)] + \
            [front_frac]
    corridor_y = [f * layout.height for f in fracs]

    # One corridor node per zone-column, per corridor row
    col_xs = sorted(set(round((z.x1 + z.x2) / 2, 2) for z in layout.zones))
    corridor_nodes: Dict[Tuple[int, int], Node] = {}
    for ri, y in enumerate(corridor_y):
        prev = None
        for ci, x in enumerate(col_xs):
            n = Node(f"corridor_{ri}_{ci}", x, y, kind="corridor")
            nodes.append(n)
            corridor_nodes[(ri, ci)] = n
            if prev is not None:
                edges.append(Edge(prev.id, n.id, _dist((prev.x, prev.y), (n.x, n.y))))
            prev = n
        # connect corridor rows vertically at each column
        if ri > 0:
            for ci in range(len(col_xs)):
                a, b = corridor_nodes[(ri - 1, ci)], corridor_nodes[(ri, ci)]
                edges.append(Edge(a.id, b.id, _dist((a.x, a.y), (b.x, b.y))))

    # connect entrance to nearest back-corridor node
    back_row = [corridor_nodes[(0, ci)] for ci in range(len(col_xs))]
    nearest_back = min(back_row, key=lambda n: _dist((n.x, n.y), (ex, ey)))
    edges.append(Edge("entrance", nearest_back.id, _dist((ex, ey), (nearest_back.x, nearest_back.y))))

    # wire each zone to its nearest corridor node (by row-band proximity)
    for z in layout.zones:
        zx, zy = z.centroid
        zone_node = Node(z.id, zx, zy, kind="zone")
        nodes.append(zone_node)
        nearest = min(nodes[1:len(nodes)-1], key=lambda n: _dist((n.x, n.y), (zx, zy)) if n.kind == "corridor" else float("inf"))
        d = _dist((zx, zy), (nearest.x, nearest.y))
        if enforce_walls and edge_crosses_any_wall(zone_node, nearest, layout.walls):
            continue  # HARD EXCLUSION — mirrors production wall-blocking behavior
        edges.append(Edge(zone_node.id, nearest.id, d))

    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────
# 4. Method B — Centroid-Proximity Construction
# ─────────────────────────────────────────────────────────────────────────

def build_centroid_proximity_graph(
    layout: SyntheticLayout,
    k_neighbors: int = 3,
    centerline_points: int = 2,
    enforce_walls: bool = False,
) -> Tuple[List[Node], List[Edge]]:
    """
    No global corridor assumption. Each zone gets a centroid node plus a
    small row of centerline nodes; zones are connected to their
    k-nearest-neighbor zone centroids directly. The entrance connects to
    the single nearest zone centroid.

    This mirrors the production system's alternative aisle-center /
    nearest-neighbor construction path — a genuinely different strategy
    from Method A, which is why the two methods can disagree on the same
    input layout (the core phenomenon this paper studies).
    """
    nodes: List[Node] = []
    edges: List[Edge] = []

    ex, ey = layout.entrance
    nodes.append(Node("entrance", ex, ey, kind="entrance"))

    zone_nodes: List[Node] = []
    for z in layout.zones:
        zx, zy = z.centroid
        zn = Node(z.id, zx, zy, kind="zone")
        nodes.append(zn)
        zone_nodes.append(zn)
        # centerline sub-nodes along the zone's long axis
        for i in range(1, centerline_points + 1):
            t = i / (centerline_points + 1)
            cx = z.x1 + t * (z.x2 - z.x1)
            cn = Node(f"{z.id}_c{i}", cx, zy, kind="corridor")
            nodes.append(cn)
            edges.append(Edge(zn.id, cn.id, _dist((zn.x, zn.y), (cn.x, cn.y))))

    # k-nearest-neighbor connections between zone centroids
    for zn in zone_nodes:
        dists = sorted(
            ((_dist((zn.x, zn.y), (other.x, other.y)), other) for other in zone_nodes if other.id != zn.id),
            key=lambda t: t[0],
        )
        for d, other in dists[:k_neighbors]:
            if enforce_walls and edge_crosses_any_wall(zn, other, layout.walls):
                continue  # HARD EXCLUSION
            edges.append(Edge(zn.id, other.id, d))

    # entrance connects to single nearest zone centroid
    nearest_zone = min(zone_nodes, key=lambda n: _dist((n.x, n.y), (ex, ey)))
    edges.append(Edge("entrance", nearest_zone.id, _dist((ex, ey), (nearest_zone.x, nearest_zone.y))))

    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────
# 5. Basic connectivity / routability check (seed for the M1-M4 harness)
# ─────────────────────────────────────────────────────────────────────────

def reachable_fraction(nodes: List[Node], edges: List[Edge], source_id: str = "entrance") -> float:
    """
    BFS from the entrance; returns the fraction of ZONE nodes reachable.
    This is a minimal stand-in for the M1 (connectivity) metric — intended
    as a starting point for the fuller M1-M4 harness, not the final
    implementation.
    """
    adj: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for e in edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)

    seen = {source_id}
    queue = [source_id]
    while queue:
        cur = queue.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    zone_ids = [n.id for n in nodes if n.kind == "zone"]
    if not zone_ids:
        return 0.0
    reached = sum(1 for zid in zone_ids if zid in seen)
    return reached / len(zone_ids)


# ─────────────────────────────────────────────────────────────────────────
# 6. Example parameter sweep (illustrative — expand for the real harness)
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'rows':>4} {'cols':>4} {'walls':>6} {'A: reach%':>10} {'B: reach%':>10}")
    for n_rows in (2, 3, 4):
        for n_cols in (2, 4, 6):
            for with_walls in (False, True):
                layout = generate_synthetic_layout(
                    n_rows=n_rows, n_cols=n_cols, with_walls=with_walls, seed=42
                )
                nodes_a, edges_a = build_corridor_template_graph(layout, enforce_walls=with_walls)
                nodes_b, edges_b = build_centroid_proximity_graph(layout, enforce_walls=with_walls)
                ra = reachable_fraction(nodes_a, edges_a) * 100
                rb = reachable_fraction(nodes_b, edges_b) * 100
                print(f"{n_rows:>4} {n_cols:>4} {str(with_walls):>6} {ra:>9.1f}% {rb:>9.1f}%")
