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
# Camera constants aligned to the MEASURED RoboMaster retrofit (15 Aug 2026
# handoff: pitch 2.1° below horizontal, effective HFOV of the square-cropped
# 64×64 input 32.0° from a 54° lens; both validated against known-size pads).
# Height: the handoff's tape-fit said 18 ± 1 cm but a direct ruler measurement
# of the lens centre reads 20 cm — the ruler wins. Isaac's default 20.955 mm
# horizontal aperture with focal 36.5 mm gives 2·atan(20.955/73) = 32.05°.
CAM_H = 0.20     # camera height (m), ruler-verified on the robot
CAM_FOCAL_MM = 36.5
# ROS-convention orientation: forward (+X), pitched 2.1° below horizontal —
# base (0.5,-0.5,0.5,-0.5) composed with R_x(-2.1°) in the optical frame.
CAM_ROT_ROS = (0.490754, -0.509079, 0.509079, -0.490754)
ROBOT_SIZE = (0.3, 0.24, 0.15)  # RoboMaster S1 footprint-ish (m)

BAFFLE_COLS = (8, 10)
BAFFLE_L = 0.6
BAFFLE_T = 0.2


def remove_rung(grid: np.ndarray) -> np.ndarray:
    """Wall off Tier 1's inter-corridor rung (the peek-and-reroute escape).

    Tier 1 added the rung so a wrong-corridor discovery cost a SMALL detour
    (its hazard rendered as an impassable-looking wall, and without an escape
    every condition pinned there). In continuous control the same detour is
    nearly free (~68 control steps of time penalty, race v3), which collapses
    the value of the scout's message to almost nothing: the masked-message
    z_t policy solved both slab sides by peeking and rerouting through the
    rung. Sealing it makes corridor choice a real commitment — a blind
    navigator that meets the slab must either cross it (hazard penalty) or
    backtrack the full corridor (~190 steps + discounted success) — so the
    message is worth its cost again. Tier 1's pin-at-the-wall failure mode
    does not apply here: Tier 2 slabs are low, visibly passable geometry.
    """
    g = grid.copy()
    size = g.shape[0]
    mid = size // 2
    hazard_cols = sorted({c for _, c in map(tuple, np.argwhere(g == HAZARD))})
    rung_col = hazard_cols[0] - 1
    # rows strictly between the two corridors (top ends at mid-2, bottom
    # starts at mid+2), matching Tier 1's rung carve
    g[mid - 1 : mid + 2, rung_col] = WALL
    return g


def chokepoint_grid(seed: int) -> np.ndarray:
    """The Tier 2 race grid: Tier 1's chokepoint map with the rung sealed."""
    return remove_rung(generate_chokepoint_map(np.random.default_rng(seed)).grid)


def corridor_span_cols(grid: np.ndarray) -> tuple[int, int]:
    """Columns spanned by the corridor section, i.e. strictly between chambers.

    The central divider row is wall between the corridors and open in both
    chambers. Take the SPAN, not a contiguous run: Tier 1's rung punches a hole
    in the divider whenever the grid has not been through `remove_rung` yet.
    """
    size = grid.shape[0]
    mid = size // 2
    cols = [c for c in range(1, size - 1) if grid[mid, c] == WALL]  # skip border
    return min(cols), max(cols)


def corridor_rect(grid: np.ndarray, cell: float, top: bool) -> tuple[float, float, float, float]:
    """One corridor's world-frame AABB, chambers excluded."""
    size = grid.shape[0]
    mid = size // 2
    r0, r1 = (mid - 3, mid - 2) if top else (mid + 2, mid + 3)
    c0, c1 = corridor_span_cols(grid)
    x0, _ = grid_to_world(r0, c0, size, cell)
    x1, _ = grid_to_world(r0, c1, size, cell)
    _, y0 = grid_to_world(r0, 0, size, cell)
    _, y1 = grid_to_world(r1, 0, size, cell)
    h = cell / 2
    return (x0 - h, x1 + h, min(y0, y1) - h, max(y0, y1) + h)


def route_distance_fields(
    grid: np.ndarray, cell: float, goal_xy: tuple[float, float], res: float = 0.05
) -> tuple[dict, float, float]:
    """Distance-to-goal fields that route VIA a chosen corridor.

    Returns ({True: via_top, False: via_bottom}, origin, res). Following the
    downhill gradient from anywhere -- including the open chamber, where every
    previous generation stalled -- leads through the commanded corridor. This is
    the dense POSITIVE signal for the correct route that eight failed
    generations never had: penalties could only make the wrong route costly,
    which taught the policy to stop rather than to turn.

    Two properties matter and the first attempt at this got both wrong by
    marking the wrong corridor as merely expensive (cost x8):

    1. Slope magnitude must stay ~1 m/m. An x8 cost multiplier makes the field
       x8 STEEPER through the region it was meant to discourage, so crossing it
       paid 8x per meter and the policy learned to bulldoze the hazard corridor
       (success 0.73 while obedience sat at 0.00, race stage 1.5 v2).
    2. Inside the wrong corridor the gradient must point back OUT, not onward
       to the goal, and the region cannot be a flat plateau of obstacle fill
       (no slope to escape). So the wrong corridor is treated as a dead-end
       pocket: distances inside it are extended from its WEST mouth only, which
       is exactly the reroute instruction wanted. Any potential is admissible
       here -- potential-based shaping does not move the optimum -- so the
       pocket only has to point the search the right way.
    """
    free, origin = _free_mask(grid, cell, res)
    n, _, xs = _grid_axes(grid, cell, res)
    gi = int(round((goal_xy[0] - origin) / res))
    gj = int(round((goal_xy[1] - origin) / res))

    fields = {}
    for take_top in (True, False):
        px0, px1, py0, py1 = corridor_rect(grid, cell, top=not take_top)
        pocket = np.zeros((n, n), dtype=bool)
        pocket[np.ix_((xs >= px0) & (xs <= px1), (xs >= py0) & (xs <= py1))] = True
        pocket &= free

        # true geodesic to the goal with the wrong corridor removed
        dist = _dijkstra(free & ~pocket, [(0.0, gi, gj)], res)
        assert np.isfinite(dist[gi, gj]) and dist[gi, gj] == 0.0

        # extend into the pocket from the chamber cells just west of its mouth
        mouth = (xs >= px0 - 2 * res) & (xs < px0)
        band = np.zeros((n, n), dtype=bool)
        band[np.ix_(mouth, (xs >= py0) & (xs <= py1))] = True
        band &= free & np.isfinite(dist)
        seeds = [(float(dist[i, j]), int(i), int(j)) for i, j in zip(*band.nonzero())]
        inner = _dijkstra(pocket | band, seeds, res)
        dist = np.where(pocket & np.isfinite(inner), inner, dist)

        finite_max = dist[np.isfinite(dist)].max()
        dist[~np.isfinite(dist)] = finite_max + 1.0
        fields[take_top] = dist.astype(np.float32)
    return fields, origin, res


def descend_field(
    field: np.ndarray,
    origin: float,
    res: float,
    start_xy: tuple[float, float],
    stop_rect: tuple[float, float, float, float] | None = None,
    max_steps: int = 4000,
) -> np.ndarray:
    """Greedy downhill walk on a distance field -> (K, 3) array of x, y, yaw.

    Used to lay out a reverse curriculum along the route the policy was told to
    take. Placing the robot at the corridor mouth is not enough: with a fixed
    mouth spawn, obedience there saturates at ~1.00 while obedience from the
    canonical start stays at exactly 0.00 (stage 1.5 v4) -- the chamber-to-mouth
    leg has never been trained in any generation, and nothing transfers back to
    it. Walking the spawn point along this path builds that leg directly.

    Steps two cells at a time: neighbouring samples one cell apart can alias to
    the same cell under the half-cell origin offset, which stalls the walk on a
    tie.
    """
    h = 2 * res

    def sample(p):
        i = int(round((p[0] - origin) / res))
        j = int(round((p[1] - origin) / res))
        i = min(max(i, 0), field.shape[0] - 1)
        j = min(max(j, 0), field.shape[1] - 1)
        return float(field[i, j])

    def inside(p):
        x0, x1, y0, y1 = stop_rect
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    p = np.asarray(start_xy, dtype=float)
    pts = [p.copy()]
    for _ in range(max_steps):
        if stop_rect is not None and inside(p):
            break
        best, bv = None, sample(p)
        for dx in (-h, 0.0, h):
            for dy in (-h, 0.0, h):
                q = p + (dx, dy)
                v = sample(q)
                if v < bv:
                    best, bv = q, v
        if best is None:
            break
        p = best
        pts.append(p.copy())

    xy = np.asarray(pts)
    # yaw along the direction of travel; hold the last heading at the end
    d = np.diff(xy, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0]) if len(d) else np.zeros(1)
    yaw = np.concatenate([yaw, yaw[-1:]]) if len(d) else np.zeros(len(xy))
    return np.column_stack([xy, yaw]).astype(np.float32)


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


def _grid_axes(grid: np.ndarray, cell: float, res: float):
    size = grid.shape[0]
    extent = size * cell
    n = int(round(extent / res))
    origin = -extent / 2 + res / 2  # center of cell (0, 0)
    return n, origin, origin + res * np.arange(n)


def _free_mask(grid: np.ndarray, cell: float, res: float) -> tuple[np.ndarray, float]:
    n, origin, xs = _grid_axes(grid, cell, res)
    free = np.ones((n, n), dtype=bool)
    for x0, x1, y0, y1 in obstacle_rects(grid, cell):
        free[np.ix_((xs >= x0) & (xs <= x1), (xs >= y0) & (xs <= y1))] = False
    return free, origin


def _dijkstra(free: np.ndarray, seeds: list[tuple[float, int, int]], res: float) -> np.ndarray:
    """8-connected Dijkstra over `free`, from `seeds` of (cost, i, j).

    Multi-source, so it also serves to extend a field into a region seeded from
    that region's boundary values.
    """
    n = free.shape[0]
    dist = np.full((n, n), np.inf, dtype=np.float64)
    pq = []
    for c, i, j in seeds:
        if c < dist[i, j]:
            dist[i, j] = c
            heapq.heappush(pq, (c, i, j))
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
    return dist


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
    free, origin = _free_mask(grid, cell, res)
    gi = int(round((goal_xy[0] - origin) / res))
    gj = int(round((goal_xy[1] - origin) / res))
    assert free[gi, gj], "goal is inside an obstacle"

    dist = _dijkstra(free, [(0.0, gi, gj)], res)
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


def sample_free_positions_band(
    grid: np.ndarray,
    cell: float,
    n: int,
    rng: np.random.Generator,
    field: np.ndarray,
    origin: float,
    res: float,
    d_lo: float,
    d_hi: float,
    margin: float = 0.22,
    max_rounds: int = 200,
) -> np.ndarray:
    """Free poses whose GEODESIC distance to the goal is in [d_lo, d_hi].

    Reverse-curriculum spawn sampler: `field` is the Dijkstra distance field
    from compute_distance_field for the relevant goal. Rejection-samples
    free space, then filters by distance band.
    """
    out = np.empty((n, 2), dtype=np.float64)
    got = 0
    for _ in range(max_rounds):
        cand = sample_free_positions(grid, cell, 2 * (n - got) + 16, rng, margin)
        ix = np.clip(((cand[:, 0] - origin) / res).round().astype(int), 0, field.shape[0] - 1)
        iy = np.clip(((cand[:, 1] - origin) / res).round().astype(int), 0, field.shape[1] - 1)
        d = field[ix, iy]
        take = cand[(d >= d_lo) & (d <= d_hi)][: n - got]
        out[got : got + len(take)] = take
        got += len(take)
        if got == n:
            return out
    raise RuntimeError(
        f"could not sample {n} poses in geodesic band [{d_lo}, {d_hi}] m "
        f"after {max_rounds} rounds — band likely empty"
    )


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
    # Corridor bands as (x_min, x_max, y_min, y_max): the full traversable run
    # of each corridor, not just the slab's footprint. Route obedience needs to
    # know which corridor a robot is in, which is a wider question than whether
    # it is standing on the hazard.
    corridor_top: tuple[float, float, float, float] = (0, 0, 0, 0)
    corridor_bottom: tuple[float, float, float, float] = (0, 0, 0, 0)


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

    c_lo, c_hi = corridor_span_cols(grid)

    def corridor_band(rows):
        x0, _ = grid_to_world(rows[0], c_lo, size, cell)
        x1, _ = grid_to_world(rows[0], c_hi, size, cell)
        _, y0 = grid_to_world(rows[0], 0, size, cell)
        _, y1 = grid_to_world(rows[1], 0, size, cell)
        h = cell / 2
        return (x0 - h, x1 + h, min(y0, y1) - h, max(y0, y1) + h)

    geo.corridor_top = corridor_band(top_rows)
    geo.corridor_bottom = corridor_band(bot_rows)

    yaws = {"navigator": 0.0, "scout": np.pi}  # east / west
    for name, (r, c) in zip(AGENT_NAMES, map_spec.agent_starts):
        x, y = grid_to_world(r, c, size, cell)
        geo.starts[name] = (x, y, yaws[name])
    for name, (r, c) in zip(AGENT_NAMES, map_spec.goals):
        x, y = grid_to_world(r, c, size, cell)
        geo.goals[name] = (x, y)
    return geo
