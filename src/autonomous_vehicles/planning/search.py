"""Deterministic eight-neighbor A* with diagonal corner-cut prevention."""

import heapq
from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    cells: tuple[tuple[int, int], ...]
    cost: float | None
    expanded: int
    reached: bool


def astar(
    occupied: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    maximum_expansions: int = 100_000,
) -> SearchResult:
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2 or occupied.size == 0 or maximum_expansions < 1:
        raise ValueError("Expected nonempty 2D occupancy and positive search budget")
    height, width = occupied.shape

    def inside(cell):
        return 0 <= cell[0] < height and 0 <= cell[1] < width

    if not inside(start) or not inside(goal):
        raise ValueError("Start and goal must be inside grid")
    if occupied[start] or occupied[goal]:
        return SearchResult((), None, 0, False)
    frontier = [(hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, start)]
    distances = {start: 0.0}
    parents = {}
    closed = set()
    while frontier and len(closed) < maximum_expansions:
        _, cost, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return SearchResult(tuple(reversed(path)), cost, len(closed), True)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            neighbor = current[0] + dy, current[1] + dx
            if not inside(neighbor) or occupied[neighbor] or neighbor in closed:
                continue
            if (
                dy
                and dx
                and (occupied[current[0] + dy, current[1]] or occupied[current[0], current[1] + dx])
            ):
                continue
            tentative = cost + (sqrt(2) if dy and dx else 1)
            if tentative < distances.get(neighbor, float("inf")):
                distances[neighbor] = tentative
                parents[neighbor] = current
                heuristic = hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heapq.heappush(frontier, (tentative + heuristic, tentative, neighbor))
    return SearchResult((), None, len(closed), False)
