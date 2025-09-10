# 🧩 A\* Search: Black–White Sliding Puzzle

**Programming Project Assignment: CSCI 384 AI**

Implement **A\*** search to solve a 1-D sliding puzzle with **Black (B)** tiles, **White (W)** tiles, and one **Blank (\_)**.
You will complete a single Python script with clearly marked `# TODO` sections and produce the required outputs for two **admissible & consistent** heuristics.

This is a structured, step-by-step project: define the problem, implement the successor function and costs, write two heuristics, and run **A\***—reporting the optimal path and metrics for each heuristic.

---

## 📚 What You’ll Practice

* Formal problem modeling (state, goal, actions, path costs)
* Implementing **A\*** (best-first) with `f(n) = g(n) + h(n)`
* Designing and validating **admissible & consistent** heuristics
* Measuring search performance (expanded nodes, frontier size, total pushes)
* (Optional) Implementing **Iterative Deepening Search (IDS)**

---

## 📁 Folder / File Layout

| File           | Description                                         |
| -------------- | --------------------------------------------------- |
| `hw1_astar.py` | **Starter** with `# TODO` sections (the only file). |

> Only `hw1_astar.py` is provided to and submitted by students.

---

## 🧠 Puzzle Overview

* **Board:** a 1-D row of **10** positions
* **Tiles:** **5 `'B'`**, **4 `'W'`**, **1 `'_'`** (blank)
* **Initial state (fixed in code):**
  `('W','W','W','W','_','B','B','B','B','B')`

### Legal Moves & Costs

1. Slide a tile **adjacent** into the blank → **cost = 1**
2. **Hop** over **1–3** tiles into the blank → **cost = (# jumped) + 1**
   (equivalently, moving a tile by distance **d ∈ {1,2,3,4}** costs **d**)

### Goal Condition

All **White** tiles must be strictly to the **right** of all **Black** tiles.
The **blank position is irrelevant**.

---

## 🧩 What You Implement in `hw1_astar.py`

* `is_goal(state)` — goal test
* `successors(state)` — legal next states + step costs
* `action_cost(state, next_state)` — distance-based cost
* `heuristic_h1(state)` — boundary misplacements (admissible & consistent)
* `heuristic_h2(state)` — W–B inversion count (admissible & consistent)
* `astar(start_state, heuristic)` — A\* search core

> Complete **only** the blocks marked with `# TODO`. Do **not** rename functions, constants, or print labels.

---

## 📤 Required Output (Exact Labels)

Your program must print **two sections** (h1 then h2) and a comparison:

```
=== A* with h1 ===
Optimal solution (sequence of states):
('W','W','W','W','_','B','B','B','B','B')
...
Optimal cost (f(goal)): <number>
Expanded nodes: <number>
Frontier size at goal: <number>
Total nodes ever added to frontier: <number>

=== A* with h2 ===
... same labels ...

=== Comparison (h1 vs h2) ===
Total nodes ever added (h1): <number>
Total nodes ever added (h2): <number>
```

⚠️ **Do not** change banner text or labels—grading relies on them.

---

## 🚀 How to Run

From the folder containing the file:

```bash
python3 hw1_astar.py
```

You should see the h1 section, h2 section, and the comparison printed to the terminal.

---

## 📦 Dependencies

* **Python 3.8+**
* No external packages required.

---

## ✅ What to Submit

* Submit **only** your completed `hw1_astar.py`.
* Your file must:

  * Run without errors
  * Produce the **exact** output sections and labels shown above
  * Implement the required functions listed in **What You Implement**

Follow any naming or portal instructions from your instructor.

---

## 🏗️ Grading Breakdown (200 pts)

**1) Problem Formulation (30 pts)**

* State representation (5)
* Initial state (5)
* Goal test (5)
* Successors (10)
* Action/path cost (5)

**2) A\* with Heuristics (170 pts)**

* h1 properties (admissible/consistent) (10)
* A\* optimality across h1 & h2 (100)
* Required printed outputs for **h1** (25)
* h2 properties + outputs (25)
* Comparison (h1 vs h2) (10)

**3) Optional IDS (+50 pts)**

* If you implement IDS, you may print its results in the commented section.

---

## 🧪 Tips & Troubleshooting

* Both heuristics must return **integers ≥ 0** and **0 at goal**.
* Check the **path cost** by summing step distances along your printed path.
* Ensure A\* performs the **goal test on pop** from the frontier (tree-search style).
* Keep function names and print labels unchanged.

---

## 🔒 Academic Integrity

Write your own implementation. Discuss high-level ideas only; code must be original. Cite any references used.

---
