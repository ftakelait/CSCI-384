#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSci 384 — Artificial Intelligence
Home Assignment 1 (200 points total)

Sliding-Tile “Black/White + Blank” Puzzle
-----------------------------------------
We have 10 positions in a 1-D row: five Black tiles ('B'), four White tiles ('W'),
and one Blank ('_'). Legal moves and costs:

1) A tile may move into an adjacent empty location — cost = 1.
2) A tile may hop over 1, 2, or 3 other tiles into the empty position — cost equals
   (number of tiles jumped) + 1 → i.e., distance 2→cost2, 3→cost3, 4→cost4.

Goal condition: **All White tiles must be to the RIGHT of all Black tiles**.
The position of the blank is irrelevant for the goal.

This script is the ONLY artifact students will receive. Implement only where
`# TODO` comments appear. Do not change function names or printed labels.

Grading (mirrors the handout):
------------------------------
1. [30] Problem formulation:
   (a) [5] State representation
   (b) [5] Initial state
   (c) [5] Goal test
   (d) [10] Actions (successor function)
   (e) [5] Action/path cost

2. [170] A* with BEST-FIRST-SEARCH (AIMA-4th Fig. 3.7 style)
   (a) [10] Define admissible & consistent heuristic h1
   (b) [Optional, 10] (not required here) proof sketch for h1
   (c) [100] Correct implementation that finds an optimal solution
   (d) [25] Required printed outputs for h1 run:
       (d1) [7]  Optimal solution as a sequence of states
       (d2) [7]  Optimal cost (f(Goal))
       (d3) [6]  (a) #expanded nodes, (b) #nodes remaining in frontier at goal
       (d4) [5]  Total #nodes ever added to the frontier
   (e) [25] Repeat with heuristic h2 (admissible & consistent):
       (e1) [10] Define h2
       (e2) [5]  Optimal solution (sequence of states)
       (e3) [5]  Optimal cost
       (e4) [5]  Total #nodes ever added to the frontier
   (f) [10] Compare h1 vs h2 using (d4)/(e4)

3. [Optional, 50] IDS (m=100):
   (a)-(e) See handout; stubs provided below.

Implementation/Autograding Notes:
---------------------------------
- Keep the public API exactly as defined (function names/print labels).
- Fill ONLY where `# TODO` appears. Each TODO line has a comment telling you what to write.
- Do not import extra libraries.
- Do not alter constants, print labels, or the main() flow.
- The autograder will import and call:
    - INITIAL_STATE
    - is_goal(state)
    - successors(state)
    - action_cost(state, next_state)
    - heuristic_h1(state)
    - heuristic_h2(state)
    - astar(start_state, heuristic)
  and will parse the printed sections delineated by banners in main().

State Representation (fixed for the assignment):
------------------------------------------------
A state is a tuple[str, ...] of length 10 with elements in {'B','W','_'}.
Example: ('W','W','W','W','_','B','B','B','B','B')

Initial State (fixed for the assignment):
-----------------------------------------
To make the assignment self-contained (the figure is not provided), we define:
INITIAL_STATE = ('W','W','W','W','_','B','B','B','B','B')

This is a challenging but solvable start: all whites are to the LEFT and must end
to the RIGHT of all blacks; the blank can end anywhere.

Heuristics (you must implement h1 and h2; both must be admissible and consistent):
----------------------------------------------------------------------------------
- h1 idea (guidance): minimum # of tiles on the “wrong side” of some split point.
  For every boundary between positions, count how many 'W' are on the left side
  PLUS how many 'B' are on the right side; take the minimum across all boundaries.
  This lower-bounds the # of tiles that must cross that boundary, and each move
  moves a single tile; step cost ≥ 1. This is admissible and consistent.

- h2 idea (guidance): number of inversions between colors (pairs (i,j) with i<j,
  state[i]=='W' and state[j]=='B'). Each such W-B pair must be “uncrossed” at some
  positive cost; a single move that jumps k tiles costs k+1 ≥ k, and reduces at most
  k such crossings. Summing gives a valid lower bound. This is admissible and
  consistent for this move/cost model.

Print Format (do not modify labels):
------------------------------------
For each heuristic run (h1 then h2), print:

=== A* with h1 ===
Optimal solution (sequence of states):
<one state per line, tuple format>
Optimal cost (f(goal)): <number>
Expanded nodes: <number>
Frontier size at goal: <number>
Total nodes ever added to frontier: <number>

=== A* with h2 ===
... (same labels)

=== Comparison (h1 vs h2) ===
Total nodes ever added (h1): <number>
Total nodes ever added (h2): <number>

[Optional IDS section printed only if implemented/enabled]

"""

from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Dict
import heapq

# -------------------------
# Fixed assignment constants
# -------------------------

# DO NOT CHANGE: state uses 10 positions with five 'B', four 'W', and one '_' (blank)
State = Tuple[str, ...]
N_POS = 10
TILES = {'B', 'W', '_'}

# DO NOT CHANGE: fixed initial state (students implement algorithms to solve from here)
INITIAL_STATE: State = ('W','W','W','W','_','B','B','B','B','B')

# -------------------------
# Required helper utilities
# -------------------------

def is_goal(state: State) -> bool:
    """
    Goal: All 'W' tiles are to the RIGHT of all 'B' tiles. Blank position is irrelevant.

    Returns:
        True if no 'W' appears to the left of any 'B'; otherwise False.
    """
    # TODO: Return True iff every index i<j with state[i]=='W' and state[j]=='B' is absent.
    # In other words, there is no W before any B. Implement in 1-3 lines.
    # Write a boolean expression that checks the condition directly.
    # Example hint (do not copy literally): return not any( ... your condition ... )
    raise NotImplementedError("TODO: implement is_goal")


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
        - Step cost = d (since d-1 tiles are jumped plus 1) → matches the handout.
    Yields:
        (next_state, cost) for each legal move.
    """
    # TODO: Implement:
    # 1) Find the index of the blank: e = state.index('_')
    # 2) For each i in range(N_POS):
    #       d = abs(i - e)
    #       If d in {1,2,3,4} and state[i] != '_', then yield (swap(i,e), d)
    # Make sure to return all possible legal moves.
    raise NotImplementedError("TODO: implement successors")


def action_cost(state: State, next_state: State) -> int:
    """
    Return the step cost between 'state' and 'next_state'.

    (We also pass costs out of 'successors', but the autograder will call this too.)
    """
    # TODO: Compute |i - e| where 'i' is the index of the moved tile in 'state' and
    # 'e' is the index of the blank in 'state'. Only one swap happens per move.
    # Steps:
    #   - find blank index in both states
    #   - the moved tile's index in 'state' is the blank index in 'next_state'
    #   - distance d = abs(i - e) ; return d
    raise NotImplementedError("TODO: implement action_cost")


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

    This lower-bounds the # of tiles that must cross some boundary, each move moves one
    tile and costs ≥ 1 ⇒ admissible; moving one tile changes this count by ≤ 1 and
    each step has cost ≥ 1 ⇒ consistent.
    """
    # TODO: Implement exactly as described above.
    # Hint:
    #   best = +infinity
    #   for b in range(N_POS+1):
    #       wrong_left  = sum(1 for i in range(0, b)   if state[i] == 'W')
    #       wrong_right = sum(1 for i in range(b, N_POS) if state[i] == 'B')
    #       best = min(best, wrong_left + wrong_right)
    #   return best
    raise NotImplementedError("TODO: implement heuristic_h1")


def heuristic_h2(state: State) -> int:
    """
    h2: # of color inversions (W before B). Count pairs (i,j), i<j, with state[i]=='W' and
        state[j]=='B'. Each such pair must be “uncrossed.” A move that jumps k tiles costs
        k+1 ≥ k and reduces at most k inversions, so total path cost ≥ total inversions.

    Hence h2 is admissible and consistent for this move/cost model.
    """
    # TODO: Implement inversion count between 'W' and 'B':
    # For all i<j: if state[i]=='W' and state[j]=='B', increment a counter.
    # Return that counter.
    raise NotImplementedError("TODO: implement heuristic_h2")


# -------------------------
# A* (Best-First-Search style)
# -------------------------

class PriorityQueue:
    """Min-heap priority queue for (f,state)."""
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
    # g-costs
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
            # Skip stale entry
            continue

        # Goal test upon pop (A* tree-search style frontier pop)
        if is_goal(current):
            path = reconstruct_path(came_from, current)
            cost = g[current]
            frontier_size_at_goal = len(frontier)
            return path, cost, expanded_count, frontier_size_at_goal, total_pushed

        closed.add(current)

        # Expand successors
        expanded_count += 1
        for nxt, step_cost in successors(current):
            new_g = g[current] + step_cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                came_from[nxt] = current
                f = new_g + heuristic(nxt)
                frontier.push(f, nxt)
                total_pushed += 1

    # If no solution (should not happen for this assignment)
    return [], float('inf'), expanded_count, len(frontier), total_pushed


# -------------------------
# Optional: Iterative Deepening Search (IDS)
# -------------------------

def _depth_limited_search(state: State, depth_limit: int,
                          path_cost: int, visited: set[State],
                          best_solution: Dict[str, object]) -> None:
    """
    Helper for IDS. You may implement if you choose to do the optional part.
    Not used by the autograder unless you enable it in main().
    """
    # Optional TODO: Implement DLS with cost accumulation and counters
    return


def iterative_deepening_search(start: State, max_depth: int = 100):
    """
    Optional IDS implementation.
    Returns a dict with keys similar to A* so you can print comparable results.
    """
    # Optional TODO: Implement IDS (not required).
    return {
        "path": [],
        "cost": None,
        "expanded": 0,
        "frontier_total_added": 0,
        "depth_of_optimal": None,
    }


# -------------------------
# Pretty-print utilities
# -------------------------

def format_state(state: State) -> str:
    """Return a compact tuple-like string for printing (autograder relies on this)."""
    # Example: "('W','W','W','W','_','B','B','B','B','B')"
    return "(" + ",".join(f"'{c}'" for c in state) + ")"


def print_run_banner(title: str):
    print(title)
    # exactly one line break after title
    # Do not modify the banner format (used by autograder)


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
    print()  # newline after each section

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

    # --- Optional IDS (uncomment to enable once implemented) ---
    # print("=== Optional IDS ===")
    # ids_res = iterative_deepening_search(INITIAL_STATE, max_depth=100)
    # print("Optimal solution (sequence of states):")
    # for s in ids_res["path"]:
    #     print(format_state(s))
    # print(f"Optimal cost (f(goal)): {ids_res['cost']}")
    # print(f"Depth of optimal goal: {ids_res['depth_of_optimal']}")
    # print(f"Total nodes ever added to frontier: {ids_res['frontier_total_added']}")
    # print()

if __name__ == "__main__":
    main()
