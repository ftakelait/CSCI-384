# 🧩 Constraint Satisfaction Problems - Programming Assignment

## 📋 Assignment Overview

This assignment will teach you how to implement **Constraint Satisfaction Problems (CSPs)** and various algorithms to solve them. You'll work with classic CSP problems, implement advanced solving techniques, and analyze algorithm performance.

**Points**: 150 total (120 core + 15 conceptual + 15 bonus)

## 🎯 What You'll Learn

By completing this assignment, you will:
- ✅ Understand the mathematical foundations of CSPs
- ✅ Implement backtracking search algorithms
- ✅ Apply constraint propagation techniques
- ✅ Use advanced heuristics (MRV, Degree, LCV)
- ✅ Analyze CSP structure and difficulty
- ✅ Compare algorithm performance
- ✅ Solve real-world CSP applications

## 📁 Project Files

```
📦 Your Assignment Files:
├── 📄 src/constraint_satisfaction_project.py     ← MAIN FILE (you complete this)
├── 📄 requirements.txt                           ← Python packages needed
├── 📄 README.md                                 
│
📁 data/ (CSP Datasets):
├── 📄 sample_csp_problems.json                  ← Sample CSP problems
└── 📄 csp_parameters.csv                        ← CSP generation parameters
│
📁 src/ (Helper Files - DON'T EDIT):
├── 📄 csp_generator.py                          ← CSP generation functions
├── 📄 csp_inference.py                          ← CSP solving algorithms
├── 📄 csp_utils.py                              ← Utility functions
└── 📄 csp_visualizer.py                         ← Visualization tools
```

## 🚀 Getting Started

### Step 1: Setup Your Environment
```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Understand Constraint Satisfaction Problems

**What are CSPs?** 🤔

A Constraint Satisfaction Problem consists of:
- **Variables**: Set of variables that need values
- **Domains**: Possible values for each variable
- **Constraints**: Restrictions on variable assignments

#### 🧩 **Mathematical Formulation**

A CSP is formally defined as a triple (X, D, C) where:

- **X = {X₁, X₂, ..., Xₙ}** is a finite set of variables
- **D = {D₁, D₂, ..., Dₙ}** is a finite set of domains where Dᵢ is the domain of variable Xᵢ
- **C = {C₁, C₂, ..., Cₘ}** is a finite set of constraints

Each constraint Cᵢ is a pair (scope, relation) where:
- **scope** is a tuple of variables that the constraint restricts
- **relation** is a set of tuples of values that satisfy the constraint

#### 🔍 **Constraint Types**

1. **Unary Constraints**: Restrict values of a single variable
   - Example: X₁ ≠ 0

2. **Binary Constraints**: Restrict pairs of variables
   - Example: X₁ ≠ X₂

3. **Global Constraints**: Involve multiple variables
   - Example: AllDifferent(X₁, X₂, ..., Xₙ)

#### 📊 **Constraint Tightness**

Constraint tightness measures how restrictive a constraint is:

```
p = |incompatible_pairs| / |total_possible_pairs|
```

Where:
- **p** = constraint tightness (0 ≤ p ≤ 1)
- **incompatible_pairs** = pairs of values that violate the constraint
- **total_possible_pairs** = all possible value pairs

### Step 3: Complete the Assignment

Open `src/constraint_satisfaction_project.py` and complete these sections:

## 📊 **Detailed Point Breakdown (150 points total)**

### 🔧 Core Requirements (120 points)

#### **[20 pts] STEP 1: Sample Problems Analysis**
- Load and analyze sample CSP problems (N-Queens, Map Coloring, Scheduling)
- Convert problem formats and perform structure analysis
- Visualize constraint graphs and problem solutions

#### **[25 pts] STEP 2: Advanced CSP Solving Techniques**
- Implement MRV (Minimum Remaining Values) heuristic
- Implement Degree heuristic for variable selection
- Implement LCV (Least Constraining Value) heuristic for value ordering

#### **[20 pts] STEP 3: CSP Structure Analysis**
- Generate CSPs with different parameters (n, p, α, r)
- Analyze constraint density and tightness
- Study the relationship between structure and difficulty

#### **[25 pts] STEP 4: Algorithm Performance Comparison**
- Compare basic backtracking vs. forward checking vs. MAC
- Measure solving time and success rates
- Analyze search tree depth and node expansion

#### **[15 pts] STEP 5: Real-world CSP Applications**
- Implement course scheduling CSP
- Implement resource allocation CSP
- Solve and visualize real-world problems

#### **[20 pts] STEP 6: CSP Optimization and Heuristics**
- Implement constraint ordering heuristics
- Implement domain reduction techniques
- Test optimization effectiveness

#### **[15 pts] STEP 7: Advanced Constraint Propagation**
- Implement path consistency (PC)
- Implement k-consistency algorithms
- Compare propagation techniques

#### **[10 pts] STEP 8: Final Analysis and Report Generation**
- Generate comprehensive performance reports
- Create visualization summaries
- Save results to files

### [CONCEPTUAL 15 pts] CONCEPTUAL QUESTIONS

Answer these questions in your code comments:

1. **Q1 (5 pts)**: What is the time complexity of backtrack search in the worst case?
2. **Q2 (5 pts)**: How does constraint propagation improve the efficiency of CSP solving?
3. **Q3 (5 pts)**: What is the relationship between constraint tightness and problem difficulty?

### [BONUS 15 pts] BONUS: Advanced CSP Analysis

Complete advanced analysis:
- Phase transition phenomenon in random CSPs
- Heuristic effectiveness analysis
- Parallel CSP solving techniques

## 🧮 **Mathematical Concepts**

### **Backtrack Search Complexity**

The time complexity of backtrack search is:
```
O(b^d)
```
Where:
- **b** = branching factor (average domain size)
- **d** = depth of search tree (number of variables)

In the worst case: **O(dⁿ)** where d is domain size and n is number of variables.

### **Arc Consistency Algorithm (AC-3)**

The AC-3 algorithm has complexity:
```
O(cd³)
```
Where:
- **c** = number of constraints
- **d** = domain size

### **Model RB Parameters**

For generating random CSPs using Model RB:
- **n** = number of variables
- **p** = constraint tightness (0 < p < 1)
- **α** = domain size parameter (0 < α < 1)
- **r** = constraint density parameter (0 < r < 1)

Domain size: **d = n^α**
Number of constraints: **m = r × n × ln(n)**

### **Phase Transition**

The satisfiability threshold occurs when:
```
p_c ≈ 1 - e^(-α/r)
```

## 📊 **Visualization and Analysis**

### 🎨 **Available Visualization Functions**
- `visualize_constraint_graph()`: Show CSP structure
- `visualize_n_queens_solution()`: Display N-Queens solutions
- `visualize_map_coloring_solution()`: Show map coloring results
- `plot_algorithm_performance()`: Compare algorithm performance
- `plot_csp_analysis()`: Analyze CSP properties

### 📈 **Performance Metrics**
- **Solving Time**: Average time to find a solution
- **Success Rate**: Percentage of problems solved
- **Search Nodes**: Number of nodes explored
- **Backtracks**: Number of backtracking steps

## 💡 Tips for Success

### 🔍 Understanding the Code
- Read the helper functions in `csp_utils.py`
- Study the CSP generation in `csp_generator.py`
- Use the visualization tools in `csp_visualizer.py`

### 🧩 CSP Problem Types
- **N-Queens**: Place queens on chessboard without attacks
- **Map Coloring**: Color regions with different colors
- **Scheduling**: Assign tasks to time slots with constraints
- **Sudoku**: Fill grid following number placement rules

### 📝 Implementation Tips
- Start with basic backtracking before adding heuristics
- Test with small problems before scaling up
- Use visualization to understand problem structure
- Measure performance systematically

## 🆘 Getting Help

### Common Issues:
1. **Import Errors**: Make sure you installed all requirements
2. **Data Loading**: Check file paths in the data/ folder
3. **Algorithm Errors**: Start with simple implementations first

### Debugging Tips:
- Print intermediate results to understand data flow
- Use small CSP instances for testing
- Check constraint satisfaction manually
- Verify heuristics with known examples

## 📋 Submission Checklist

Before submitting your assignment, make sure you have:

- ✅ Completed all 8 core steps in `constraint_satisfaction_project.py`
- ✅ Answered all 3 conceptual questions in code comments
- ✅ Implemented bonus advanced analysis functions for extra points
- ✅ Documented your algorithmic insights and interpretations
- ✅ Used meaningful variable names and clear comments
- ✅ Generated performance reports and visualizations

## 🚀 **Running the Assignment**

```bash
# Run the main assignment
python src/constraint_satisfaction_project.py
```

## 📚 **References**

1. Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Xu, K., & Li, W. (2000). Exact Phase Transitions in Random Constraint Satisfaction Problems. *Journal of Artificial Intelligence Research*, 12, 93-103.
3. Dechter, R. (2003). *Constraint Processing*. Morgan Kaufmann.

---

**Good luck with your Constraint Satisfaction Problems assignment!** 🧩✨
