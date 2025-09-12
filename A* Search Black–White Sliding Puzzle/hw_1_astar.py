#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSci 384 — Artificial Intelligence
Home Assignment 1 (150 points total)

Sliding-Tile “Black/White + Blank” Puzzle
-----------------------------------------
We have 9 positions in a 1-D row: four Black tiles ('B'), four White tiles ('W'),
and one Blank ('_'). Legal moves and costs:

1) A tile may move into an adjacent empty location — cost = 1.
2) A tile may hop over 1, 2, or 3 other tiles into the empty position — cost equals
   (number of tiles jumped) + 1 → i.e., distance 2→cost2, 3→cost3, 4→cost4.

Goal condition: **All White tiles must be to the RIGHT of all Black tiles**.
The position of the blank is irrelevant for the goal.

This script is the ONLY artifact students will receive. Implement only where
`# TODO` comments appear. Do not change function names or printed labels.

Grading (how points are awarded in code):
-----------------------------------------
TOTAL: 150 points — all points are attached to the TODOs below.
- is_goal:                15 pts
- successors:             45 pts
- action_cost:            10 pts
- heuristic_h1 (h1):      40 pts
- heuristic_h2 (h2):      40 pts
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Dict
import heapq

# -------------------------
# Fixed assignment constants (do not modify)
# -------------------------

# DO NOT CHANGE: state uses 9 positions with four 'B', four 'W', and one '_' (blank)
State = Tuple[str, ...]
N_POS = 9
TILES = {'B', 'W', '_'}

# DO NOT CHANGE: fixed initial state
INITIAL_STATE: State = ('W','W','W','W','_','B','B','B','B')

# Legal move distances (cost equals distance)
ALLOWED_DISTANCES = {1, 2, 3, 4}

# -------------------------
# Required helper utilities
# -------------------------

def is_goal(state: State) -> bool:
    """
    Goal: All 'W' tiles are to the RIGHT of all 'B' tiles (blank position irrelevant).
    Return True iff the goal condition holds for 'state'.

    Implement WITHOUT printing and WITHOUT side effects.
    """
    # TODO (is_goal – 15 pts total):
    # [7 pts] Decide the condition that makes the goal true for a given 'state'.
    goal_condition = ...        # bool: True if goal satisfied, else False
    # [3 pts] Do not use prints or assertions; just compute and return the result.
    # [5 pts] Handle all valid inputs of length N_POS with symbols from TILES.
    return goal_condition


def _swap_positions(t: Tuple[str, ...], i: int, j: int) -> Tuple[str, ...]:
    """Return a new tuple with positions i and j swapped. (Provided utility; do not modify)"""
    lst = list(t)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def successors(state: State) -> Iterable[Tuple[State, int]]:
    """
    Generate all legal successors (next_state, step_cost) from 'state'.

    Legal moves:
      - Let e be the index of '_' in 'state'.
      - A tile at index i can move into e if distance d = |i - e| ∈ {1,2,3,4}.
      - Step cost = d.
    Yield every legal (next_state, d) pair. Do NOT print.
    """
    # TODO (successors – 45 pts total):
    # [5 pts] Locate the blank index in 'state'.
    e = ...
    # [5 pts] Iterate over all tile indices to consider potential movers.
    for i in ...:
        # [5 pts] Compute the move distance d = abs(i - e).
        d = ...
        # [10 pts] Check both legality conditions:
        #          (a) d is in ALLOWED_DISTANCES
        #          (b) the moving position is not the blank
        if ...:
            # [10 pts] Construct the next state by swapping positions (i, e).
            next_state = ...
            # [10 pts] Yield the pair (next_state, d) for every legal move.
            yield ...


def action_cost(state: State, next_state: State) -> int:
    """
    Return the step cost between 'state' and 'next_state'.
    Exactly one swap (tile with blank) should have occurred.
    """
    # TODO (action_cost – 10 pts total):
    # [4 pts] Find the blank index before and after the move.
    e_before = ...
    e_after  = ...
    # [3 pts] Determine from which index the moved tile came in 'state'.
    moved_from = ...
    # [3 pts] Return |moved_from - e_before| as the integer cost.
    return ...


# -------------------------
# Admissible, consistent heuristics
# -------------------------

def heuristic_h1(state: State) -> int:
    """
    h1: Lower bound via split boundary.
    For each boundary b (0..N_POS), consider:
      - wrong_left  = # of 'W' in positions [0..b-1]
      - wrong_right = # of 'B' in positions [b..N_POS-1]
    h1 is the minimum over b of wrong_left + wrong_right.

    Must be admissible & consistent. Return a nonnegative integer.
    """
    # TODO (heuristic_h1 – 40 pts total):
    # [8 pts] Initialize a variable to track the best (minimum) value.
    best = ...
    # [8 pts] Loop over all boundaries b from 0 to N_POS (inclusive of 0, inclusive of N_POS).
    for b in ...:
        # [10 pts] Compute wrong_left for the left segment [0..b-1].
        wrong_left = ...
        # [10 pts] Compute wrong_right for the right segment [b..N_POS-1].
        wrong_right = ...
        # [4 pts] Update best using wrong_left + wrong_right.
        best = ...
    # Return the best integer lower bound.
    return ...


def heuristic_h2(state: State) -> int:
    """
    h2: Lower bound via color inversions.
    Count pairs (i, j) with i < j such that state[i] == 'W' and state[j] == 'B'.
    Return that count as a nonnegative integer.
    """
    # TODO (heuristic_h2 – 40 pts total):
    # [8 pts] Initialize the inversion counter.
    inv = ...
    # [12 pts] Double loop over i < j to inspect ordered pairs.
    for i in ...:
        for j in ...:
            # [12 pts] If the pair contributes to inversions (W before B), update the counter.
            if ...:
                inv = ...
    # [8 pts] Return the final integer count.
    return ...


# -------------------------
# A* (Best-First-Search style) — Provided
# -------------------------

class PriorityQueue:
    """Min-heap priority queue for (priority, state)."""
    def __init__(self):
        self._data = []
        self._push_count = 0  # tie-breaker

    def push(self, priority: int, item):
        self._push_count += 1
        heapq.heappush(self._data, (priority, self._push_count, item))

    def pop(self):
        return heapq.heappop(self._data)[-1]

    def __len__(self):
        return len(self._data)


def reconstruct_path(came_from: Dict[State, Optional[State]], goal: State) -> List[State]:
    path = []
    s = goal
    while s is not None:
        path.append(s)
        s = came_from[s]
    path.reverse()
    return path


def astar(start: State, heuristic: Callable[[State], int]):
    """
    Run A* from 'start' using the provided admissible & consistent 'heuristic'.

    Returns:
        path: List[State] optimal path from start to goal (inclusive)
        cost: int         optimal path cost
        expanded_count: int
        frontier_size_at_goal: int
        total_pushed: int
    """
    # Provided implementation — do not modify.
    g: Dict[State, int] = { start: 0 }
    came_from: Dict[State, Optional[State]] = { start: None }

    frontier = PriorityQueue()
    frontier.push(heuristic(start), start)

    expanded_count = 0
    total_pushed = 1  # start pushed once

    closed: set[State] = set()

    while len(frontier) > 0:
        current = frontier.pop()

        if current in closed:
            continue

        if is_goal(current):
            path = reconstruct_path(came_from, current)
            cost = g[current]
            frontier_size_at_goal = len(frontier)
            return path, cost, expanded_count, frontier_size_at_goal, total_pushed

        closed.add(current)

        expanded_count += 1
        for nxt, step_cost in successors(current):
            new_g = g[current] + step_cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                came_from[nxt] = current
                f = new_g + heuristic(nxt)
                frontier.push(f, nxt)
                total_pushed += 1

    return [], float('inf'), expanded_count, len(frontier), total_pushed


# -------------------------
# Pretty-print utilities — Provided
# -------------------------

def format_state(state: State) -> str:
    """Return a compact tuple-like string for printing (autograder relies on this)."""
    return "(" + ",".join(f"'{c}'" for c in state) + ")"


def print_run_banner(title: str):
    print(title)


def run_and_report(heur_name: str, heuristic: Callable[[State], int]):
    title = f"=== A* with {heur_name} ==="
    print_run_banner(title)
    path, cost, expanded, frontier_size_at_goal, total_pushed = astar(INITIAL_STATE, heuristic)

    print("Optimal solution (sequence of states):")
    for s in path:
        print(format_state(s))
    print(f"Optimal cost (f(goal)): {cost}")
    print(f"Expanded nodes: {expanded}")
    print(f"Frontier size at goal: {frontier_size_at_goal}")
    print(f"Total nodes ever added to frontier: {total_pushed}")
    print()

    return {
        "path": path,
        "cost": cost,
        "expanded": expanded,
        "frontier_size_at_goal": frontier_size_at_goal,
        "total_pushed": total_pushed
    }


def main():
    # --- h1 run ---
    res1 = run_and_report("h1", heuristic_h1)

    # --- h2 run ---
    res2 = run_and_report("h2", heuristic_h2)

    # --- comparison ---
    print("=== Comparison (h1 vs h2) ===")
    print(f"Total nodes ever added (h1): {res1['total_pushed']}")
    print(f"Total nodes ever added (h2): {res2['total_pushed']}")
    print()


if __name__ == "__main__":
    main()
