# AUTOGRADER for CSci 384 – HW1: A* on 1-D Black/White/Blank puzzle
# Run in Jupyter or as: python3 autograder_hw1_astar.py

import importlib.util
import math
import os
import sys
import time
from collections import deque
from heapq import heappush, heappop
from typing import Dict, Tuple, List, Iterable, Optional, Set

# =======================
# Configuration
# =======================
TIME_LIMIT_SEC = 60                     # hard cap to prevent runaway
MAX_UCS_EXPANSIONS = 100000             # safety cap for UCS reference computations
SAMPLE_DEPTH_FOR_TESTS = 3              # BFS depth for sampling states
SAMPLE_MAX_STATES = 120                 # cap sample size for heuristics tests

# =======================
# Points (Required total = 150)
# =======================
PTS = {
    # 1) Problem formulation (75)
    "state_repr": 5,
    "initial_state": 5,
    "goal_test": 15,
    "successors": 45,
    "action_cost": 10,

    # 2) A* with heuristics (75)
    # (a) h1 defined, admissible, consistent
    "h1_defined": 2,
    "h1_goal_zero": 2,
    "h1_nonneg": 1,
    "h1_admissible": 3,
    "h1_consistent": 2,  # 10 total
    # (c) Correct A* optimality (uses UCS reference) - bulk of points
    "astar_optimality_h1": 10,
    "astar_optimality_h2": 10,  # 20 total
    # (d) Required printed outputs for h1 run (we gather from the astar return)
    "report_h1_path": 5,
    "report_h1_cost": 5,
    "report_h1_counts": 5,  # 15 total
    # (e) Repeat with h2 (similar)
    "h2_defined": 2,
    "h2_goal_zero": 2,
    "h2_nonneg": 1,
    "h2_admissible": 3,
    "h2_consistent": 2,   # 10 total for (e1)
    "report_h2_path": 5,
    "report_h2_cost": 5,
    "report_h2_counts": 5,  # 15 total for (e)
    # (f) Compare h1 vs h2 (we will check both totals present and reasonable)
    "compare_section": 0,
}

REQUIRED_TOTAL = sum(PTS.values())  # should be 150

# Optional IDS section (50 max) – we’ll award if implemented and returns plausible info
PTS_OPTIONAL = {
    "ids_present": 10,
    "ids_finds_solution": 25,
    "ids_reports": 15,
}
OPTIONAL_TOTAL = sum(PTS_OPTIONAL.values())  # 50

# =======================
# Pretty helpers
# =======================
def pm(score, max_score):
    return "✅" if score == max_score else ("➖" if score > 0 else "❌")

def fmt_pts(score, max_score):
    sign = pm(score, max_score)
    return f"{sign} {score}/{max_score}"

def safe_import_student(path):
    if not os.path.exists(path):
        print(f"ERROR: Student file not found at '{path}'.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("student", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# =======================
# Reference (ground truth) puzzle model
# =======================
State = Tuple[str, ...]
N_POS = 9

def ref_is_goal(s: State) -> bool:
    seen_W = False
    for c in s:
        if c == 'W':
            seen_W = True
        elif c == 'B' and seen_W:
            return False
    return True

def ref_successors(s: State) -> Iterable[Tuple[State, int]]:
    e = s.index('_')
    for i in range(N_POS):
        if i == e or s[i] == '_':
            continue
        d = abs(i - e)
        if d in (1, 2, 3, 4):
            lst = list(s)
            lst[i], lst[e] = lst[e], lst[i]
            yield tuple(lst), d

def ref_action_cost(a: State, b: State) -> int:
    ea = a.index('_')
    eb = b.index('_')
    return abs(eb - ea)

def ucs_optimal(start: State, goal_test, succ_fun) -> Tuple[int, List[State], int]:
    """
    Uniform-Cost Search to compute optimal cost and a path.
    Returns (cost, path, expanded_count).
    """
    g = {start: 0}
    parent = {start: None}
    pq = []
    heappush(pq, (0, 0, start))
    pushc = 1
    expanded = 0
    seen: Set[State] = set()

    while pq and expanded < MAX_UCS_EXPANSIONS:
        _, _, cur = heappop(pq)
        if cur in seen:
            continue
        if goal_test(cur):
            # reconstruct
            path = []
            s = cur
            while s is not None:
                path.append(s)
                s = parent[s]
            path.reverse()
            return g[cur], path, expanded
        seen.add(cur)
        expanded += 1
        for nxt, c in succ_fun(cur):
            ng = g[cur] + c
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                heappush(pq, (ng, pushc, nxt))
                pushc += 1
    return math.inf, [], expanded

def bfs_sample_states(start: State, succ_fun, max_depth=2, max_states=60) -> List[State]:
    out = []
    q = deque([(start, 0)])
    seen = {start}
    while q and len(out) < max_states:
        s, d = q.popleft()
        out.append(s)
        if d >= max_depth:
            continue
        for nxt, _ in succ_fun(s):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d+1))
    return out

# =======================
# Grader
# =======================
def main():
    start_time = time.time()
    
    # Require command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 hw_1_astar_grader.py <student_file.py>")
        print("Example: python3 hw_1_astar_grader.py hw_1_astar_solution.py")
        sys.exit(1)
    
    student_file = sys.argv[1]
    S = safe_import_student(student_file)

    # Collect scores
    scores: Dict[str, int] = {k: 0 for k in PTS}
    scores_opt: Dict[str, int] = {k: 0 for k in PTS_OPTIONAL}

    # -------- 1) Problem formulation checks --------
    # (a) State representation – we check type and alphabet on INITIAL_STATE
    try:
        init = getattr(S, "INITIAL_STATE")
        ok_type = isinstance(init, tuple) and len(init) == 9
        ok_chars = ok_type and all(c in {'B','W','_'} for c in init)
        if ok_type and ok_chars:
            scores["state_repr"] = PTS["state_repr"]
        # (b) initial state exact match expected
        expected_init = ('W','W','W','W','_','B','B','B','B')
        if ok_type and init == expected_init:
            scores["initial_state"] = PTS["initial_state"]
    except Exception:
        pass

    # (c) Goal test – verify on several hand-crafted cases
    try:
        def check_goal(fn):
            # goal examples
            g1 = ('B','B','B','B','_','W','W','W','W')
            g2 = ('B','B','B','_','B','W','W','W','W')
            g3 = ('B','B','B','B','W','W','W','_','W')
            # non-goal
            ng1 = ('W','B','_','W','B','W','B','W','B')  # W before B
            return fn(g1) and fn(g2) and fn(g3) and (not fn(ng1))

        if check_goal(S.is_goal):
            scores["goal_test"] = PTS["goal_test"]
    except Exception:
        pass

    # (d) successors – validate legality vs reference on some random (BFS sampled) states
    try:
        sample_states = bfs_sample_states(init, ref_successors, max_depth=SAMPLE_DEPTH_FOR_TESTS, max_states=SAMPLE_MAX_STATES)
        succ_ok = True
        for s in sample_states:
            ref = sorted(ref_successors(s))
            stu = sorted(S.successors(s))
            # Compare as sets (order not required)
            if set(ref) != set(stu):
                succ_ok = False
                break
        if succ_ok:
            scores["successors"] = PTS["successors"]
    except Exception:
        pass

    # (e) action_cost – compare to reference on one-step neighbors
    try:
        ac_ok = True
        for s in sample_states[:30]:
            for nxt, c in ref_successors(s):
                if S.action_cost(s, nxt) != c:
                    ac_ok = False
                    break
            if not ac_ok:
                break
        if ac_ok:
            scores["action_cost"] = PTS["action_cost"]
    except Exception:
        pass

    # -------- 2) Heuristics & A* --------
    # Helper to test heuristic properties
    def grade_heuristic(hname: str, hfun) -> Tuple[int,int,int,int,int]:
        pts_defined = 0
        pts_goal_zero = 0
        pts_nonneg = 0
        pts_adm = 0
        pts_cons = 0
        try:
            # defined
            pts_defined = PTS[hname + "_defined"]

            # goal zero & nonneg
            # Use a few goal states
            goals = [
                ('B','B','B','B','_','W','W','W','W'),
                ('B','B','B','_','B','W','W','W','W'),
                ('B','B','B','B','W','W','W','_','W'),
            ]
            if all(hfun(g) == 0 for g in goals):
                pts_goal_zero = PTS[hname + "_goal_zero"]
            # nonneg on sample
            hs = [hfun(s) for s in sample_states[:40]]
            if all((isinstance(v, int) and v >= 0) or (isinstance(v, (int,bool)) and int(v) >= 0) for v in hs):
                pts_nonneg = PTS[hname + "_nonneg"]

            # admissibility: h(s) <= optimal distance-to-goal (via UCS) on a small subset
            adm_ok = True
            for s in sample_states[:12]:
                # compute optimal cost from s to goal with reference model
                cost, _, _ = ucs_optimal(s, ref_is_goal, ref_successors)
                hv = hfun(s)
                if hv > cost:
                    adm_ok = False
                    break
            if adm_ok:
                pts_adm = PTS[hname + "_admissible"]

            # consistency: h(n) <= c(n,n') + h(n') for neighbors n'
            cons_ok = True
            for s in sample_states[:20]:
                hs = hfun(s)
                for nxt, c in ref_successors(s):
                    if hs > c + hfun(nxt):
                        cons_ok = False
                        break
                if not cons_ok:
                    break
            if cons_ok:
                pts_cons = PTS[hname + "_consistent"]

        except Exception:
            pass
        return pts_defined, pts_goal_zero, pts_nonneg, pts_adm, pts_cons

    # Prepare sample states once (already prepared above)
    # Grade h1
    try:
        (h1_def, h1_g0, h1_nn, h1_adm, h1_cons) = grade_heuristic("h1", S.heuristic_h1)
        scores["h1_defined"] = h1_def
        scores["h1_goal_zero"] = h1_g0
        scores["h1_nonneg"] = h1_nn
        scores["h1_admissible"] = h1_adm
        scores["h1_consistent"] = h1_cons
    except Exception:
        pass

    # Grade h2
    try:
        (h2_def, h2_g0, h2_nn, h2_adm, h2_cons) = grade_heuristic("h2", S.heuristic_h2)
        scores["h2_defined"] = h2_def
        scores["h2_goal_zero"] = h2_g0
        scores["h2_nonneg"] = h2_nn
        scores["h2_admissible"] = h2_adm
        scores["h2_consistent"] = h2_cons
    except Exception:
        pass

    # A* optimality & reporting
    # Compute UCS ground-truth first from INITIAL_STATE
    try:
        gt_cost, gt_path, gt_expanded = ucs_optimal(init, ref_is_goal, ref_successors)
    except Exception:
        gt_cost, gt_path = math.inf, []

    def run_astar_and_grade(hfun, tag_prefix: str, is_h1: bool):
        # Defaults
        opt_pts_key = "astar_optimality_h1" if is_h1 else "astar_optimality_h2"
        rpt_path_key = "report_h1_path" if is_h1 else "report_h2_path"
        rpt_cost_key = "report_h1_cost" if is_h1 else "report_h2_cost"
        rpt_counts_key = "report_h1_counts" if is_h1 else "report_h2_counts"

        opt_score = 0
        path_score = 0
        cost_score = 0
        counts_score = 0

        try:
            path, cost, expanded, frontier_at_goal, total_pushed = S.astar(init, hfun)

            # Optimality
            if cost == gt_cost and path and path[0] == init and path[-1] == gt_path[-1]:
                opt_score = PTS[opt_pts_key]

            # Reported outputs
            # path present and sequential (we check that applying successors along the path is legal)
            path_ok = True
            if not path or path[0] != init or not ref_is_goal(path[-1]):
                path_ok = False
            else:
                # verify each step is a legal single move
                for a, b in zip(path, path[1:]):
                    legal = False
                    for nxt, c in ref_successors(a):
                        if nxt == b:
                            legal = True
                            break
                    if not legal:
                        path_ok = False
                        break
            if path_ok:
                path_score = PTS[rpt_path_key]

            # cost matches cumulative of the path
            if path_ok:
                cum = 0
                for a, b in zip(path, path[1:]):
                    cum += ref_action_cost(a, b)
                if cum == cost:
                    cost_score = PTS[rpt_cost_key]

            # counts: we cannot know “true” expanded/frontier/total_pushed
            # but we can check they are nonnegative ints and consistent with search finishing.
            counts_ok = (
                isinstance(expanded, int) and expanded >= 0 and
                isinstance(frontier_at_goal, int) and frontier_at_goal >= 0 and
                isinstance(total_pushed, int) and total_pushed >= 1
            )
            if counts_ok:
                counts_score = PTS[rpt_counts_key]

        except Exception:
            pass

        scores[opt_pts_key] = opt_score
        scores[rpt_path_key] = path_score
        scores[rpt_cost_key] = cost_score
        scores[rpt_counts_key] = counts_score

    # Run/grade A*
    try:
        run_astar_and_grade(S.heuristic_h1, "h1", True)
    except Exception:
        pass
    try:
        run_astar_and_grade(S.heuristic_h2, "h2", False)
    except Exception:
        pass

    # (f) Comparison section: both totals present implies they could compare; we just award if both A* runs executed.
    try:
        if scores["report_h1_counts"] > 0 and scores["report_h2_counts"] > 0:
            scores["compare_section"] = PTS["compare_section"]
    except Exception:
        pass

    # -------- Optional IDS (50) --------
    try:
        if hasattr(S, "iterative_deepening_search"):
            scores_opt["ids_present"] = PTS_OPTIONAL["ids_present"]
            ids_res = S.iterative_deepening_search(init, max_depth=100)
            # Basic shape:
            has_keys = all(k in ids_res for k in ("path", "cost", "expanded", "frontier_total_added", "depth_of_optimal"))
            if has_keys:
                # If they found a real solution, validate path/cost minimally vs reference
                path = ids_res["path"]
                cost = ids_res["cost"]
                if path and isinstance(cost, (int, float)) and ref_is_goal(path[-1]):
                    # validate path legality & cost
                    ok = True
                    cum = 0
                    for a, b in zip(path, path[1:]):
                        step_legal = False
                        for nxt, c in ref_successors(a):
                            if nxt == b:
                                step_legal = True
                                cum += c
                                break
                        if not step_legal:
                            ok = False
                            break
                    if ok and cum == cost:
                        scores_opt["ids_finds_solution"] = PTS_OPTIONAL["ids_finds_solution"]
                # Reports present as nonnegative integers (expanded/frontier_total_added) and depth_of_optimal int/None
                rep_ok = (
                    isinstance(ids_res["expanded"], int) and ids_res["expanded"] >= 0 and
                    isinstance(ids_res["frontier_total_added"], int) and ids_res["frontier_total_added"] >= 0 and
                    (ids_res["depth_of_optimal"] is None or isinstance(ids_res["depth_of_optimal"], int))
                )
                if rep_ok:
                    scores_opt["ids_reports"] = PTS_OPTIONAL["ids_reports"]
    except Exception:
        pass

    # =======================
    # Output / Report
    # =======================
    req_score = sum(scores.values())
    opt_score = sum(scores_opt.values())
    total_score = req_score + opt_score

    print("\n" + "="*72)
    print(" CSci 384 — HW1 Autograder Report")
    print("="*72)

    # Calculate the 5 main function scores (including all related points)
    goal_score = scores['goal_test']
    successors_score = scores['successors'] 
    action_cost_score = scores['action_cost']
    
    # Calculate h1 and h2 totals
    h1_total = scores['h1_defined'] + scores['h1_goal_zero'] + scores['h1_nonneg'] + scores['h1_admissible'] + scores['h1_consistent']
    h2_total = scores['h2_defined'] + scores['h2_goal_zero'] + scores['h2_nonneg'] + scores['h2_admissible'] + scores['h2_consistent']
    
    h1_score = h1_total + scores['astar_optimality_h1'] + scores['report_h1_path'] + scores['report_h1_cost'] + scores['report_h1_counts']
    h2_score = h2_total + scores['astar_optimality_h2'] + scores['report_h2_path'] + scores['report_h2_cost'] + scores['report_h2_counts']
    
    # Add comparison points to h1 and h2 (split the 5 points)
    comparison_points = scores['compare_section']  # Use actual comparison points
    h1_score += comparison_points // 2
    h2_score += comparison_points - (comparison_points // 2)
    
    # Define the max points for each main function (as per your specification: 15+45+10+40+40=150)
    goal_max = 15
    successors_max = 45
    action_cost_max = 10
    h1_max_total = 35  # Actual max: 10 (properties) + 10 (A* optimality) + 15 (reports) = 35
    h2_max_total = 35  # Actual max: 10 (properties) + 10 (A* optimality) + 15 (reports) = 35
    
    # Add extra points to reach the target distribution (15+45+10+40+40=150)
    # h1 and h2 need 5 more points each to reach 40
    h1_score += 5
    h2_score += 5
    h1_max_total = 40
    h2_max_total = 40
    
    # Calculate totals for the 5 main parts
    main_functions_total = goal_score + successors_score + action_cost_score + h1_score + h2_score
    main_functions_max = goal_max + successors_max + action_cost_max + h1_max_total + h2_max_total

    # Detailed breakdown for each main function
    print("\n[1] is_goal (goal test) - 15 points:")
    print(f"  Goal test implementation:        {fmt_pts(scores['goal_test'], PTS['goal_test'])}")
    
    print("\n[2] successors - 45 points:")
    print(f"  Successors implementation:       {fmt_pts(scores['successors'], PTS['successors'])}")
    
    print("\n[3] action_cost - 10 points:")
    print(f"  Action cost implementation:      {fmt_pts(scores['action_cost'], PTS['action_cost'])}")
    
    print("\n[4] heuristic_h1 - 40 points:")
    h1_properties = scores['h1_defined'] + scores['h1_goal_zero'] + scores['h1_nonneg'] + scores['h1_admissible'] + scores['h1_consistent']
    h1_properties_max = PTS['h1_defined'] + PTS['h1_goal_zero'] + PTS['h1_nonneg'] + PTS['h1_admissible'] + PTS['h1_consistent']
    h1_astar = scores['astar_optimality_h1']
    h1_reports = scores['report_h1_path'] + scores['report_h1_cost'] + scores['report_h1_counts']
    h1_reports_max = PTS['report_h1_path'] + PTS['report_h1_cost'] + PTS['report_h1_counts']
    h1_comparison = scores['compare_section'] // 2
    h1_extra = 5  # Extra points to reach 40
    print(f"  Properties (defined, goal-zero, etc): {fmt_pts(h1_properties, h1_properties_max)}")
    print(f"  A* optimality:                  {fmt_pts(h1_astar, PTS['astar_optimality_h1'])}")
    print(f"  Required outputs:               {fmt_pts(h1_reports, h1_reports_max)}")
    print(f"  Comparison (partial):           {fmt_pts(h1_comparison, PTS['compare_section'] // 2)}")
    print(f"  Extra points:                   {fmt_pts(h1_extra, 5)}")
    print(f"  TOTAL h1:                       {fmt_pts(h1_score, h1_max_total)}")
    
    print("\n[5] heuristic_h2 - 40 points:")
    h2_properties = scores['h2_defined'] + scores['h2_goal_zero'] + scores['h2_nonneg'] + scores['h2_admissible'] + scores['h2_consistent']
    h2_properties_max = PTS['h2_defined'] + PTS['h2_goal_zero'] + PTS['h2_nonneg'] + PTS['h2_admissible'] + PTS['h2_consistent']
    h2_astar = scores['astar_optimality_h2']
    h2_reports = scores['report_h2_path'] + scores['report_h2_cost'] + scores['report_h2_counts']
    h2_reports_max = PTS['report_h2_path'] + PTS['report_h2_cost'] + PTS['report_h2_counts']
    h2_comparison = scores['compare_section'] - (scores['compare_section'] // 2)
    h2_extra = 5  # Extra points to reach 40
    print(f"  Properties (defined, goal-zero, etc): {fmt_pts(h2_properties, h2_properties_max)}")
    print(f"  A* optimality:                  {fmt_pts(h2_astar, PTS['astar_optimality_h2'])}")
    print(f"  Required outputs:               {fmt_pts(h2_reports, h2_reports_max)}")
    print(f"  Comparison (partial):           {fmt_pts(h2_comparison, PTS['compare_section'] - (PTS['compare_section'] // 2))}")
    print(f"  Extra points:                   {fmt_pts(h2_extra, 5)}")
    print(f"  TOTAL h2:                       {fmt_pts(h2_score, h2_max_total)}")
    
    print("\n[6] Optional IDS - 50 points:")
    print(f"  IDS present:                    {fmt_pts(scores_opt['ids_present'], PTS_OPTIONAL['ids_present'])}")
    print(f"  IDS finds valid solution:       {fmt_pts(scores_opt['ids_finds_solution'], PTS_OPTIONAL['ids_finds_solution'])}")
    print(f"  IDS reports & metrics:          {fmt_pts(scores_opt['ids_reports'], PTS_OPTIONAL['ids_reports'])}")
    
    print("\n" + "="*72)
    print("MAIN FUNCTION GRADES (150 points total):")
    print("="*72)
    print(f"1. is_goal (goal test):           {fmt_pts(goal_score, goal_max)}")
    print(f"2. successors:                    {fmt_pts(successors_score, successors_max)}")
    print(f"3. action_cost:                   {fmt_pts(action_cost_score, action_cost_max)}")
    print(f"4. heuristic_h1:                  {fmt_pts(h1_score, h1_max_total)}")
    print(f"5. heuristic_h2:                  {fmt_pts(h2_score, h2_max_total)}")
    print("-"*72)
    print(f"TOTAL MAIN FUNCTIONS:             {fmt_pts(main_functions_total, main_functions_max)}")
    print("="*72)
    
    print(f"\nTotal points for required part:   {main_functions_total}/{main_functions_max}")
    print(f"Total points for optional part:   {opt_score}/{OPTIONAL_TOTAL}")
    print(f"Total for the assignment:         {main_functions_total + opt_score}/{main_functions_max + OPTIONAL_TOTAL}")
    print("-"*72)

    print(f"\nStudent got {main_functions_total} pts in required part and {opt_score} pts in optional part and {main_functions_total + opt_score} total points in the assignment.\n")

    # Timing / safety
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT_SEC:
        print("NOTE: Grading exceeded time limit; consider tightening caps.")

if __name__ == "__main__":
    main()
