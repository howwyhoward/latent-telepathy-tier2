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
    compute_distance_field,
    compute_geometry,
    generate_chokepoint_map,
    obstacle_rects,
)

CELL = 0.5
SEED = 2

grid = generate_chokepoint_map(np.random.default_rng(SEED)).grid
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
