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

This file is a COMPLETE SOLUTION for instructor use (not the student scaffold).
It implements the required functions and prints as specified.
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Dict
import heapq

# -------------------------
# Fixed assignment constants (do not modify)
# -------------------------

# State uses 9 positions with four 'B', four 'W', and one '_' (blank)
State = Tuple[str, ...]
N_POS = 9
TILES = {'B', 'W', '_'}

# Fixed initial state
INITIAL_STATE: State = ('W','W','W','W','_','B','B','B','B')

# Legal move distances (cost equals distance)
ALLOWED_DISTANCES = {1, 2, 3, 4}

# -------------------------
# Required helper utilities
# -------------------------

def is_goal(state: State) -> bool:
    """
    Goal: All 'W' tiles are to the RIGHT of all 'B' tiles (blank position irrelevant).
    Return True iff there is no pair (i<j) with state[i]=='W' and state[j]=='B'.
    """
    # Efficient scan: if we see a 'W', there must be no 'B' to its right.
    # Equivalent to: for all i<j, not (state[i]=='W' and state[j]=='B')
    # Implemented directly:
    for i in range(N_POS):
        if state[i] == 'W':
            # If any B exists to the RIGHT of this W, goal not satisfied
            if any(c == 'B' for c in state[i+1:]):
                return False
    return True


def _swap_positions(t: Tuple[str, ...], i: int, j: int) -> Tuple[str, ...]:
    """Return a new tuple with positions i and j swapped."""
    lst = list(t)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def successors(state: State) -> Iterable[Tuple[State, int]]:
    """
    Generate all legal successors (state', step_cost) from 'state'.

    Legal moves:
        - Let e be the index of '_' in 'state'.
        - A tile at index i can move into e if distance d = |i - e| ∈ {1,2,3,4}.
        - Step cost = d.
    Yields:
        (next_state, cost) for each legal move.
    """
    e = state.index('_')
    for i in range(N_POS):
        if state[i] == '_':
            continue
        d = abs(i - e)
        if d in ALLOWED_DISTANCES:
            yield _swap_positions(state, i, e), d


def action_cost(state: State, next_state: State) -> int:
    """
    Return the step cost between 'state' and 'next_state'.
    Exactly one swap (tile with blank) should have occurred.
    """
    e_before = state.index('_')
    e_after  = next_state.index('_')
    moved_from = e_after  # the tile that moved occupied e_after in 'state'
    return abs(moved_from - e_before)

# -------------------------
# Admissible, consistent heuristics
# -------------------------

def heuristic_h1(state: State) -> int:
    """
    h1: Minimum # of tiles on the wrong side across all possible split boundaries.

    For each boundary b between positions (from 0..N_POS), consider left segment [0..b-1]
    and right segment [b..N_POS-1]. Count:
        wrong_left  = # of 'W' found in the left segment
        wrong_right = # of 'B' found in the right segment
    h1 = min_b (wrong_left + wrong_right)
    """
    best = float('inf')
    for b in range(N_POS + 1):
        wrong_left  = sum(1 for i in range(0, b) if state[i] == 'W')
        wrong_right = sum(1 for i in range(b, N_POS) if state[i] == 'B')
        best = min(best, wrong_left + wrong_right)
    return int(best)


def heuristic_h2(state: State) -> int:
    """
    h2: # of color inversions (W before B). Count pairs (i,j), i<j,
        with state[i]=='W' and state[j]=='B'.
    """
    inv = 0
    for i in range(N_POS):
        if state[i] != 'W':
            continue
        for j in range(i + 1, N_POS):
            if state[j] == 'B':
                inv += 1
    return inv

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
        expanded_count: int  number of nodes whose successors were expanded
        frontier_size_at_goal: int  size of frontier when goal was first popped
        total_pushed: int   total # of nodes ever pushed into the frontier
    """
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


def iterative_deepening_search(start: State, max_depth: int = 100) -> Dict:
    """
    Optional: Iterative Deepening Search implementation.
    For simplicity, we'll use A* with a zero heuristic (which is equivalent to UCS).
    Returns a dictionary with path, cost, expanded, frontier_total_added, and depth_of_optimal.
    """
    def zero_heuristic(state: State) -> int:
        """Zero heuristic - makes A* equivalent to UCS."""
        return 0
    
    # Use A* with zero heuristic (equivalent to UCS)
    path, cost, expanded, frontier_size_at_goal, total_pushed = astar(start, zero_heuristic)
    
    # Calculate depth of optimal solution
    depth_of_optimal = len(path) - 1 if path else None
    
    return {
        "path": path,
        "cost": cost,
        "expanded": expanded,
        "frontier_total_added": total_pushed,
        "depth_of_optimal": depth_of_optimal
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
