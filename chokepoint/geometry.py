"""Pure (Isaac-free) geometry for the chokepoint scene.

Everything here is derived from Tier 1's `generate_chokepoint_map()` grid and
plain numpy, so it can be unit-tested without launching Kit. `scene.py`
re-exports these names and adds the Isaac prim configs on top.

The geodesic distance field replaces Euclidean distance in the reward
shaping. Tier 1's gridworld shaping was implicitly geodesic (grid-graph
distance routes around walls); a Euclidean port pins the robot into the
second baffle's corner, where "toward the goal" points INTO the wall
(observed in the first M7 run: every rollout stalled there, 0.00 success).
"""

import heapq
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import AGENT_NAMES

# Tier 1 is the source of truth for the map
sys.path.insert(0, str(Path.home() / "latent-telepathy"))
from envs.constants import HAZARD, WALL  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402

WALL_H = 0.5     # wall height (m)
HAZARD_H = 0.05  # hazard slab height (m) — low slab, like Tier 1's
CAM_H = 0.2      # camera height (m) — roughly RoboMaster S1 turret height
ROBOT_SIZE = (0.3, 0.24, 0.15)  # RoboMaster S1 footprint-ish (m)

BAFFLE_COLS = (8, 10)
BAFFLE_L = 0.6
BAFFLE_T = 0.2


def grid_to_world(row: float, col: float, size: int, cell: float) -> tuple[float, float]:
    """Grid (row, col) -> world (x, y), relative to the env origin.

    Row 0 is 'north' (+y); col grows east (+x). USD is Z-up right-handed,
    units in meters (stated once, per Tier 1 practice).
    """
    x = (col - (size - 1) / 2) * cell
    y = ((size - 1) / 2 - row) * cell
    return x, y


def wall_runs(grid: np.ndarray):
    """Merge contiguous horizontal WALL cells into (row, col_start, col_end) runs."""
    runs = []
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1]:
            if grid[r, c] == WALL:
                c0 = c
                while c < grid.shape[1] and grid[r, c] == WALL:
                    c += 1
                runs.append((r, c0, c - 1))
            else:
                c += 1
    return runs


def obstacle_rects(grid: np.ndarray, cell: float) -> list[tuple[float, float, float, float]]:
    """All static obstacles as world-frame AABBs (x0, x1, y0, y1): walls + baffles.

    Hazard slabs are NOT obstacles — they are passable (penalized by region
    check), so the shaping field must not route around them.
    """
    size = grid.shape[0]
    mid = size // 2
    rects = []
    h = cell / 2
    for r, c0, c1 in wall_runs(grid):
        x0, y = grid_to_world(r, c0, size, cell)
        x1, _ = grid_to_world(r, c1, size, cell)
        rects.append((x0 - h, x1 + h, y - h, y + h))
    for r0, r1 in [(mid - 3, mid - 2), (mid + 2, mid + 3)]:
        _, y_n = grid_to_world(r0, 0, size, cell)
        _, y_s = grid_to_world(r1, 0, size, cell)
        y_north_edge = y_n + h
        y_south_edge = y_s - h
        for col, attach in ((BAFFLE_COLS[0], "north"), (BAFFLE_COLS[1], "south")):
            x, _ = grid_to_world(0, col, size, cell)
            yc = (y_north_edge - BAFFLE_L / 2) if attach == "north" else (y_south_edge + BAFFLE_L / 2)
            rects.append((x - BAFFLE_T / 2, x + BAFFLE_T / 2, yc - BAFFLE_L / 2, yc + BAFFLE_L / 2))
    return rects


def compute_distance_field(
    grid: np.ndarray,
    cell: float,
    goal_xy: tuple[float, float],
    res: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """Geodesic (shortest-path-around-obstacles) distance to goal, in meters.

    Returns (field, origin, res): field[ix, iy] is the distance from world
    point (origin + ix*res, origin + iy*res) to the goal, routed around walls
    and baffles via 8-connected Dijkstra. Obstacle/unreachable cells get the
    max finite value + 1 m, so a bilinear sample near a wall reads slightly
    uphill — the gradient gently pushes away from walls, which is fine for
    shaping (potential-based shaping is policy-invariant regardless).
    """
    size = grid.shape[0]
    extent = size * cell
    n = int(round(extent / res))
    origin = -extent / 2 + res / 2  # center of cell (0, 0)

    free = np.ones((n, n), dtype=bool)
    xs = origin + res * np.arange(n)
    for x0, x1, y0, y1 in obstacle_rects(grid, cell):
        ix = (xs >= x0) & (xs <= x1)
        iy = (xs >= y0) & (xs <= y1)
        free[np.ix_(ix, iy)] = False

    dist = np.full((n, n), np.inf, dtype=np.float64)
    gi = int(round((goal_xy[0] - origin) / res))
    gj = int(round((goal_xy[1] - origin) / res))
    assert free[gi, gj], "goal is inside an obstacle"

    dist[gi, gj] = 0.0
    pq = [(0.0, gi, gj)]
    diag = res * np.sqrt(2)
    steps = [(-1, -1, diag), (-1, 0, res), (-1, 1, diag),
             (0, -1, res), (0, 1, res),
             (1, -1, diag), (1, 0, res), (1, 1, diag)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, w in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and free[ni, nj] and d + w < dist[ni, nj]:
                dist[ni, nj] = d + w
                heapq.heappush(pq, (d + w, ni, nj))

    finite_max = dist[np.isfinite(dist)].max()
    dist[~np.isfinite(dist)] = finite_max + 1.0
    return dist.astype(np.float32), origin, res


def sample_free_positions(
    grid: np.ndarray,
    cell: float,
    n: int,
    rng: np.random.Generator,
    margin: float = 0.22,
) -> np.ndarray:
    """Rejection-sample (n, 2) world xy poses clear of walls and baffles.

    `margin` inflates obstacles by roughly the robot's half-diagonal so a
    spawned robot never intersects geometry. Hazard slabs are NOT excluded —
    they are passable, and views of them are exactly what the encoder needs.
    """
    size = grid.shape[0]
    half = size * cell / 2 - margin
    rects = [
        (x0 - margin, x1 + margin, y0 - margin, y1 + margin)
        for x0, x1, y0, y1 in obstacle_rects(grid, cell)
    ]
    out = np.empty((n, 2), dtype=np.float64)
    got = 0
    while got < n:
        cand = rng.uniform(-half, half, size=(2 * (n - got) + 8, 2))
        ok = np.ones(len(cand), dtype=bool)
        for x0, x1, y0, y1 in rects:
            ok &= ~(
                (cand[:, 0] >= x0) & (cand[:, 0] <= x1)
                & (cand[:, 1] >= y0) & (cand[:, 1] <= y1)
            )
        take = cand[ok][: n - got]
        out[got : got + len(take)] = take
        got += len(take)
    return out


@dataclass
class ChokepointGeometry:
    """Everything the RL env needs to know about the scene, in env-local meters."""

    size: int
    cell: float
    # per-slab-cell: (x, y_top, y_bottom) — reset moves slabs between the two y's
    slab_cells: list[tuple[float, float, float]] = field(default_factory=list)
    # hazard AABBs per side: (x_min, x_max, y_min, y_max)
    hazard_aabb_top: tuple[float, float, float, float] = (0, 0, 0, 0)
    hazard_aabb_bottom: tuple[float, float, float, float] = (0, 0, 0, 0)
    # (x, y, yaw) per agent, Tier 1 start poses; navigator faces east, scout west
    starts: dict = field(default_factory=dict)
    goals: dict = field(default_factory=dict)


def compute_geometry(seed: int = 2, cell: float = 0.5) -> ChokepointGeometry:
    """Derive scene metadata from the Tier 1 map (canonical seed builds slab TOP)."""
    map_spec = generate_chokepoint_map(np.random.default_rng(seed))
    grid = map_spec.grid
    size = grid.shape[0]
    mid = size // 2
    top_rows = (mid - 3, mid - 2)
    bot_rows = (mid + 2, mid + 3)

    geo = ChokepointGeometry(size=size, cell=cell)

    hazard_cells = sorted(map(tuple, np.argwhere(grid == HAZARD)))
    hazard_cols = sorted({c for _, c in hazard_cells})
    # slab cells enumerated as (row_pair_index, col); y for both candidate sides
    for i, (r, c) in enumerate(hazard_cells):
        row_idx = 0 if r in (top_rows[0], bot_rows[0]) else 1
        x, _ = grid_to_world(r, c, size, cell)
        _, y_top = grid_to_world(top_rows[row_idx], c, size, cell)
        _, y_bot = grid_to_world(bot_rows[row_idx], c, size, cell)
        geo.slab_cells.append((x, y_top, y_bot))

    def aabb(rows):
        x0, _ = grid_to_world(rows[0], hazard_cols[0], size, cell)
        x1, _ = grid_to_world(rows[0], hazard_cols[-1], size, cell)
        _, y0 = grid_to_world(rows[0], 0, size, cell)
        _, y1 = grid_to_world(rows[1], 0, size, cell)
        h = cell / 2
        return (x0 - h, x1 + h, min(y0, y1) - h, max(y0, y1) + h)

    geo.hazard_aabb_top = aabb(top_rows)
    geo.hazard_aabb_bottom = aabb(bot_rows)

    yaws = {"navigator": 0.0, "scout": np.pi}  # east / west
    for name, (r, c) in zip(AGENT_NAMES, map_spec.agent_starts):
        x, y = grid_to_world(r, c, size, cell)
        geo.starts[name] = (x, y, yaws[name])
    for name, (r, c) in zip(AGENT_NAMES, map_spec.goals):
        x, y = grid_to_world(r, c, size, cell)
        geo.goals[name] = (x, y)
    return geo
