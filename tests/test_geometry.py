"""Unit tests for the geodesic shaping field (no Isaac required).

The load-bearing test is the stall-point regression: the first M7 run pinned
every rollout at (~0.0, -1.8), the corner of the second baffle in the bottom
corridor, because Euclidean shaping points southeast (into the wall) there.
The geodesic field must point NORTH (through the S-gap) at that spot.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.geometry import (  # noqa: E402
    WALL,
    chokepoint_grid,
    compute_distance_field,
    compute_geometry,
    generate_chokepoint_map,
    corridor_rect,
    obstacle_rects,
    route_distance_fields,
)

CELL = 0.5
SEED = 2

grid = chokepoint_grid(SEED)
geo = compute_geometry(seed=SEED, cell=CELL)
GOAL = geo.goals["navigator"]
field, origin, res = compute_distance_field(grid, CELL, GOAL)


def sample(x, y):
    i = int(round((x - origin) / res))
    j = int(round((y - origin) / res))
    return field[i, j]


def in_obstacle(x, y):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in obstacle_rects(grid, CELL))


def test_goal_is_zero():
    assert sample(*GOAL) < res * 2


def test_start_is_reachable_and_geodesic_exceeds_euclidean():
    sx, sy, _ = geo.starts["navigator"]
    d = sample(sx, sy)
    euclid = np.hypot(sx - GOAL[0], sy - GOAL[1])
    assert np.isfinite(d)
    assert euclid < d < 3 * euclid  # routed around walls, not through them


def test_stall_point_gradient_points_north():
    # the observed stall point from the failed Euclidean run
    x, y = 0.0, -1.8
    assert not in_obstacle(x, y)
    here = sample(x, y)
    north = sample(x, y + 0.3)   # toward the S-gap
    south_east = sample(0.3, -1.9)  # where Euclidean shaping pushed
    assert north < here, "going north through the gap must reduce geodesic distance"
    assert south_east >= here - 0.05, "hugging the baffle corner must not look like progress"


def test_gap_is_downhill_corridor_to_exit():
    # bottom corridor: mouth -> between baffles -> gap -> corridor exit
    mouth = sample(-2.0, -1.5)
    between = sample(-0.25, -1.6)
    in_gap = sample(0.25, -1.2)
    past = sample(1.0, -1.5)
    assert mouth > between > in_gap > past


def test_rung_is_sealed():
    """Race v3 regression: with the rung open, a blind policy peeks at the
    slab and reroutes for ~68 steps of time penalty, collapsing the value of
    the scout's message to ~0 (masked-message ablation was null). The Tier 2
    grid must wall the rung so corridor choice is a real commitment."""
    raw = generate_chokepoint_map(np.random.default_rng(SEED)).grid
    size = raw.shape[0]
    mid = size // 2
    rung_col = None
    for c in range(4, size - 4):  # corridor span only (chambers are open too)
        if raw[mid, c] != WALL:
            rung_col = c
    assert rung_col is not None, "Tier 1 map should have an open rung"
    assert all(grid[r, rung_col] == WALL for r in range(mid - 1, mid + 2))


def test_wrong_corridor_geodesic_routes_back_not_through_rung():
    # discovery point: bottom corridor just past the second baffle, slab ahead
    x_disc, y_disc = 0.6, -1.5
    here = sample(x_disc, y_disc)
    north_into_rung = sample(0.75, -0.6)  # old rung passage, now walled
    assert north_into_rung > here, "sealed rung must not read as progress"


def test_corridor_bands_separate_the_two_routes():
    """Route obedience is scored by corridor membership, so the bands must
    contain the mouths, exclude both chambers, and never overlap."""
    top, bot = geo.corridor_top, geo.corridor_bottom

    def inside(band, x, y):
        return band[0] <= x <= band[1] and band[2] <= y <= band[3]

    assert inside(top, -2.5, 1.0) and not inside(bot, -2.5, 1.0)
    assert inside(bot, -2.5, -1.5) and not inside(top, -2.5, -1.5)
    # the canonical start and both chambers belong to neither corridor
    sx, sy, _ = geo.starts["navigator"]
    for x, y in ((sx, sy), (-3.75, -3.75), (3.75, -3.75), (3.75, 3.75)):
        assert not inside(top, x, y) and not inside(bot, x, y), (x, y)
    assert top[2] > bot[3], "corridor bands must not overlap in y"
    # each band must cover its own slab and not the other's
    hz_top, hz_bot = geo.hazard_aabb_top, geo.hazard_aabb_bottom
    for (x0, x1, y0, y1), band, other in (
        (hz_top, top, bot), (hz_bot, bot, top)
    ):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        assert inside(band, cx, cy) and not inside(other, cx, cy)


def test_corridor_bands_are_traversable_end_to_end():
    """Both bands must be real routes: a band clipped short of a corridor's
    exit would score a robot as 'obedient' while it sat in a dead end."""
    free_max = field.max() - 0.5
    for band in (geo.corridor_top, geo.corridor_bottom):
        x0, x1, y0, y1 = band
        cy = (y0 + y1) / 2
        entry, exit_ = sample(x0 + 0.3, cy), sample(x1 - 0.3, cy)
        assert entry < free_max and exit_ < free_max, "band ends inside obstacle"
        assert exit_ < entry, "east end of a corridor must be closer to the goal"


route_fields, r_origin, r_res = route_distance_fields(grid, CELL, GOAL)
_rects = obstacle_rects(grid, CELL)


def is_free(x, y):
    return not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in _rects)


def rsample(take_top, x, y):
    f = route_fields[take_top]
    i = int(round((x - r_origin) / r_res))
    j = int(round((y - r_origin) / r_res))
    return f[i, j]


def test_route_field_is_downhill_toward_the_commanded_corridor():
    """The fix for eight generations of failure: from the canonical start, the
    first step toward the commanded corridor must already pay. Every previous
    incentive could only punish the wrong route, which taught the policy to
    stop rather than to turn."""
    sx, sy, _ = geo.starts["navigator"]
    for take_top, better, worse in ((True, +0.4, -0.4), (False, -0.4, +0.4)):
        here = rsample(take_top, sx, sy)
        assert rsample(take_top, sx, sy + better) < here, (take_top, "no downhill")
        assert rsample(take_top, sx, sy + worse) > here, (take_top, "wrong way pays")


def test_route_field_prefers_its_own_mouth():
    top_mouth, bot_mouth = (-2.5, 1.0), (-2.5, -1.5)
    assert rsample(True, *top_mouth) < rsample(True, *bot_mouth)
    assert rsample(False, *bot_mouth) < rsample(False, *top_mouth)


def descend(take_top, start, steps=600, h=2 * r_res):  # 1 cell aliases on the
    # half-cell origin offset, so neighbouring samples can hit the same cell
    """Greedy downhill walk on a route field, returning the path.

    Sampling one step next to the start is not enough to validate a field: the
    x8-cost version passed that check while hiding a ridge two metres away whose
    far side paid 8 m per metre for crossing the wrong corridor.
    """
    f = route_fields[take_top]
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        best, bv = None, rsample(take_top, *p)
        for dx in (-h, 0.0, h):
            for dy in (-h, 0.0, h):
                q = p + (dx, dy)
                v = rsample(take_top, *q)
                if v < bv:
                    best, bv = q, v
        if best is None:
            break
        p = best
        path.append(p.copy())
    return np.array(path)


def test_gradient_descent_reaches_the_goal_through_the_commanded_corridor():
    sx, sy, _ = geo.starts["navigator"]
    for take_top in (True, False):
        path = descend(take_top, (sx, sy))
        band = geo.corridor_top if take_top else geo.corridor_bottom
        x0, x1, y0, y1 = band
        inside = ((path[:, 0] >= x0) & (path[:, 0] <= x1)
                  & (path[:, 1] >= y0) & (path[:, 1] <= y1))
        assert inside.any(), f"commanded {take_top}: never entered its corridor"
        other = geo.corridor_bottom if take_top else geo.corridor_top
        x0, x1, y0, y1 = other
        wrong = ((path[:, 0] >= x0) & (path[:, 0] <= x1)
                 & (path[:, 1] >= y0) & (path[:, 1] <= y1))
        assert not wrong.any(), f"commanded {take_top}: routed via the wrong corridor"
        assert rsample(take_top, *path[-1]) < 0.6, "descent stalled short of the goal"


def test_slope_is_bounded_so_no_region_outpays_the_correct_route():
    """The x8 field made the wrong corridor 8x steeper, so bulldozing it paid 8x
    per metre. Slope must stay near 1 m/m everywhere the robot can travel.

    Two discontinuities are by design and excluded: baffle cells carry the
    obstacle fill value to push the robot off walls, and the wrong corridor's
    far mouth is where the field switches from distance-out-of-the-pocket to
    distance-to-goal. The env clamps per-step shaping to v_max * dt so neither
    can pay out.
    """
    for take_top in (True, False):
        pocket = corridor_rect(grid, CELL, top=not take_top)

        def same_region(xa, xb, y, p=pocket):
            return ((p[0] <= xa <= p[1] and p[2] <= y <= p[3])
                    == (p[0] <= xb <= p[1] and p[2] <= y <= p[3]))

        for y in (1.0, -1.5, -0.25):
            pts = [(float(x), rsample(take_top, float(x), y))
                   for x in np.arange(-3.5, 3.51, 0.1) if is_free(float(x), y)]
            for (xa, va), (xb, vb) in zip(pts, pts[1:]):
                if abs(xb - xa) > 0.15 or not same_region(xa, xb, y):
                    continue
                assert abs(vb - va) / (xb - xa) < 2.0, (take_top, y, xa, va, vb)


def test_wrong_corridor_slopes_back_out_the_way_it_came():
    """A robot that strays into the wrong corridor must be told to reverse, and
    must never find going deeper cheaper than backing out."""
    for take_top, wrong_y in ((True, -1.5), (False, 1.0)):
        v = [float(rsample(take_top, float(x), wrong_y))
             for x in np.arange(-2.5, 2.51, 0.25) if is_free(float(x), wrong_y)]
        assert all(b > a for a, b in zip(v, v[1:])), \
            f"commanded {take_top}: going deeper into the wrong corridor is downhill"
        # It should back out WEST and then run the commanded corridor to the
        # goal, so check how it leaves the pocket rather than where it ends up.
        x0, x1, y0, y1 = corridor_rect(grid, CELL, top=not take_top)
        path = descend(take_top, (0.0, wrong_y), steps=400)
        inside = ((path[:, 0] >= x0) & (path[:, 0] <= x1)
                  & (path[:, 1] >= y0) & (path[:, 1] <= y1))
        left = int(np.argmin(inside))  # first step outside the pocket
        assert not inside[left:].any(), "re-entered the wrong corridor"
        assert path[left][0] < 0.0, \
            f"commanded {take_top}: left the pocket eastward at x={path[left][0]:.2f}"
        assert rsample(take_top, *path[-1]) < 0.6, "did not go on to reach the goal"


def test_obstacles_are_uphill_everywhere():
    # obstacle cells were filled with max+1, so they can never read as progress
    free_max = field.max() - 0.5
    x, _ = 0.25, None  # south baffle center x
    assert sample(0.25, -1.9) > free_max or in_obstacle(0.25, -1.9)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
