"""Grid routing against simple analytical shortest paths and blocked maps."""

from math import sqrt

import numpy as np
import pytest

from autonomous_vehicles.planning.search import astar
from autonomous_vehicles.simulation.geometry import Pose2, Vec2
from autonomous_vehicles.simulation.occupancy import GridSpec, OccupancyGrid, inflate_obstacles
from autonomous_vehicles.simulation.shapes import Rectangle


def test_unobstructed_diagonal_cost():
    result = astar(np.zeros((5, 5), dtype=bool), (0, 0), (4, 4))
    assert result.reached
    assert result.cells[0] == (0, 0)
    assert result.cells[-1] == (4, 4)
    assert result.cost == pytest.approx(4 * sqrt(2))


def test_start_equals_goal():
    result = astar(np.zeros((2, 2), dtype=bool), (1, 1), (1, 1))
    assert result.cells == ((1, 1),)
    assert result.cost == 0


def test_diagonal_cannot_cut_an_obstacle_corner():
    occupied = np.array([[False, True], [True, False]])
    result = astar(occupied, (0, 0), (1, 1))
    assert not result.reached
    assert result.cost is None


def test_wall_with_gap_routes_only_through_gap():
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[:, 5] = True
    occupied[7, 5] = False
    result = astar(occupied, (2, 1), (2, 8))
    assert result.reached
    assert (7, 5) in result.cells
    assert all(not occupied[cell] for cell in result.cells)


def test_search_budget_stops_without_claiming_success():
    result = astar(np.zeros((20, 20), dtype=bool), (0, 0), (19, 19), maximum_expansions=1)
    assert not result.reached
    assert result.expanded == 1


def test_blocked_start_returns_unreachable():
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[0, 0] = True
    assert not astar(occupied, (0, 0), (2, 2)).reached
    with pytest.raises(ValueError):
        astar(occupied, (-1, 0), (2, 2))


def test_grid_cell_center_roundtrip():
    spec = GridSpec(width=10, height=8, resolution=0.5, origin=Vec2(-1, -2))
    for row in range(spec.height):
        for column in range(spec.width):
            assert spec.cell(spec.center(row, column)) == (row, column)
    assert spec.cell(Vec2(-2, -2)) is None
    assert spec.cell(Vec2(4, 2)) is None
    assert spec.extent == (-1, 4, -2, 2)


def test_inflation_is_disc_shaped_and_does_not_wrap():
    occupied = np.zeros((5, 5), dtype=bool)
    occupied[0, 0] = True
    inflated = inflate_obstacles(occupied, 1)
    assert inflated.sum() == 3
    assert inflated[0, 1] and inflated[1, 0]
    assert not inflated[-1, -1]
    np.testing.assert_array_equal(inflate_obstacles(occupied, 0), occupied)
    with pytest.raises(ValueError):
        inflate_obstacles(occupied, -1)


def test_rasterization_uses_cell_centers():
    grid = OccupancyGrid(GridSpec(width=4, height=4, resolution=1, origin=Vec2(0, 0)))
    occupied = grid.rasterize((Rectangle(Pose2(1, 1), 2, 2),))
    assert occupied[:2, :2].all()
    assert occupied.sum() == 4
    assert np.all(grid.probabilities == 0.5)
    grid.log_odds[0, 0] = 2
    grid.reset()
    assert np.all(grid.log_odds == 0)
    assert not grid.observed.any()
