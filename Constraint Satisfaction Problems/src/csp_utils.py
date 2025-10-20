# csp_utils.py
#
# DESCRIPTION
# -----------
# Utility functions for Constraint Satisfaction Problems
# Includes data loading, CSP analysis, and performance metrics

import json
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import time

def load_sample_csp_problems():
    """
    Load sample CSP problems from data files
    
    Returns:
        dict: Dictionary containing different types of CSP problems
    """
    problems = {
        'n_queens': load_n_queens_problem(),
        'map_coloring': load_map_coloring_problem(),
        'sudoku': load_sudoku_problem(),
        'scheduling': load_scheduling_problem()
    }
    return problems

def load_n_queens_problem(n=8):
    """
    Generate N-Queens CSP problem
    
    Args:
        n (int): Size of the chessboard (default: 8)
    
    Returns:
        tuple: (variables, domains, constraints)
    """
    variables = list(range(n))
    domains = {i: list(range(n)) for i in variables}
    constraints = {}
    
    # Add constraints for N-Queens problem
    for i in range(n):
        for j in range(i + 1, n):
            incompatible_pairs = []
            for val_i in range(n):
                for val_j in range(n):
                    # Same row, same column, or same diagonal
                    if (val_i == val_j or 
                        abs(val_i - val_j) == abs(i - j)):
                        incompatible_pairs.append((val_i, val_j))
            constraints[(i, j)] = incompatible_pairs
    
    return variables, domains, constraints

def load_map_coloring_problem():
    """
    Generate Map Coloring CSP problem (Australia map)
    
    Returns:
        tuple: (variables, domains, constraints)
    """
    # Variables represent Australian states
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    
    # Domain: 3 colors (Red, Green, Blue)
    domains = {var: ['R', 'G', 'B'] for var in variables}
    
    # Constraints: adjacent states cannot have same color
    constraints = {
        ('WA', 'NT'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('WA', 'SA'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('NT', 'SA'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('NT', 'Q'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('SA', 'Q'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('SA', 'NSW'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('SA', 'V'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('Q', 'NSW'): [('R', 'R'), ('G', 'G'), ('B', 'B')],
        ('NSW', 'V'): [('R', 'R'), ('G', 'G'), ('B', 'B')]
    }
    
    return variables, domains, constraints

def load_sudoku_problem():
    """
    Generate Sudoku CSP problem (9x9 grid)
    
    Returns:
        tuple: (variables, domains, constraints)
    """
    variables = [(i, j) for i in range(9) for j in range(9)]
    domains = {var: list(range(1, 10)) for var in variables}
    constraints = {}
    
    # Add row, column, and box constraints
    for i in range(9):
        for j in range(9):
            var1 = (i, j)
            
            # Row constraints
            for k in range(9):
                if k != j:
                    var2 = (i, k)
                    if (var1, var2) not in constraints and (var2, var1) not in constraints:
                        constraints[(var1, var2)] = [(val, val) for val in range(1, 10)]
            
            # Column constraints
            for k in range(9):
                if k != i:
                    var2 = (k, j)
                    if (var1, var2) not in constraints and (var2, var1) not in constraints:
                        constraints[(var1, var2)] = [(val, val) for val in range(1, 10)]
            
            # Box constraints
            box_i, box_j = i // 3, j // 3
            for bi in range(box_i * 3, (box_i + 1) * 3):
                for bj in range(box_j * 3, (box_j + 1) * 3):
                    if bi != i or bj != j:
                        var2 = (bi, bj)
                        if (var1, var2) not in constraints and (var2, var1) not in constraints:
                            constraints[(var1, var2)] = [(val, val) for val in range(1, 10)]
    
    return variables, domains, constraints

def load_scheduling_problem():
    """
    Generate Job Scheduling CSP problem
    
    Returns:
        tuple: (variables, domains, constraints)
    """
    # Variables represent jobs
    variables = ['J1', 'J2', 'J3', 'J4', 'J5']
    
    # Domain: time slots (0-7 hours)
    domains = {var: list(range(8)) for var in variables}
    
    # Constraints: resource conflicts and precedence
    constraints = {
        ('J1', 'J2'): [(t, t) for t in range(8)],  # Same resource
        ('J1', 'J3'): [(t, t) for t in range(8)],  # Same resource
        ('J2', 'J4'): [(t, t) for t in range(8)],  # Same resource
        ('J3', 'J5'): [(t, t) for t in range(8)],  # Same resource
        ('J1', 'J4'): [(t1, t2) for t1 in range(8) for t2 in range(t1, 8)]  # J1 must finish before J4
    }
    
    return variables, domains, constraints

def analyze_csp_structure(variables, domains, constraints):
    """
    Analyze the structure of a CSP
    
    Args:
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
    
    Returns:
        dict: Analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['num_variables'] = len(variables)
    analysis['num_constraints'] = len(constraints)
    analysis['avg_domain_size'] = np.mean([len(domains[var]) for var in variables])
    analysis['total_search_space'] = np.prod([len(domains[var]) for var in variables])
    
    # Constraint density
    max_constraints = len(variables) * (len(variables) - 1) // 2
    analysis['constraint_density'] = len(constraints) / max_constraints if max_constraints > 0 else 0
    
    # Constraint tightness
    total_incompatible = sum(len(incompatible_pairs) for incompatible_pairs in constraints.values())
    total_possible = sum(len(domains[var1]) * len(domains[var2]) 
                        for var1, var2 in constraints.keys())
    analysis['constraint_tightness'] = total_incompatible / total_possible if total_possible > 0 else 0
    
    # Degree analysis
    degrees = defaultdict(int)
    for var1, var2 in constraints.keys():
        degrees[var1] += 1
        degrees[var2] += 1
    
    analysis['avg_degree'] = np.mean(list(degrees.values())) if degrees else 0
    analysis['max_degree'] = max(degrees.values()) if degrees else 0
    
    return analysis

def visualize_csp_structure(variables, domains, constraints, title="CSP Structure"):
    """
    Visualize the constraint graph of a CSP
    
    Args:
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
        title (str): Title for the plot
    """
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(variables)
    
    # Add edges
    G.add_edges_from(constraints.keys())
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def measure_performance(solver_func, variables, domains, constraints, iterations=10):
    """
    Measure the performance of a CSP solver
    
    Args:
        solver_func (function): The solver function to test
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
        iterations (int): Number of iterations to average over
    
    Returns:
        dict: Performance metrics
    """
    times = []
    solutions_found = 0
    
    for _ in range(iterations):
        start_time = time.time()
        solution = solver_func({}, variables, domains, constraints)
        end_time = time.time()
        
        times.append(end_time - start_time)
        if solution != "Failure":
            solutions_found += 1
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'success_rate': solutions_found / iterations,
        'total_solutions_found': solutions_found
    }

def compare_algorithms(variables, domains, constraints, algorithms):
    """
    Compare different CSP solving algorithms
    
    Args:
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
        algorithms (dict): Dictionary mapping algorithm names to solver functions
    
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for name, solver_func in algorithms.items():
        print(f"Testing {name}...")
        results[name] = measure_performance(solver_func, variables, domains, constraints)
    
    return results

def plot_performance_comparison(results):
    """
    Plot performance comparison of different algorithms
    
    Args:
        results (dict): Results from compare_algorithms
    """
    algorithms = list(results.keys())
    avg_times = [results[alg]['avg_time'] for alg in algorithms]
    success_rates = [results[alg]['success_rate'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average time comparison
    ax1.bar(algorithms, avg_times, color='skyblue', alpha=0.7)
    ax1.set_title('Average Solving Time Comparison', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    ax2.bar(algorithms, success_rates, color='lightgreen', alpha=0.7)
    ax2.set_title('Success Rate Comparison', fontweight='bold')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def save_results_to_csv(results, filename):
    """
    Save performance results to CSV file
    
    Args:
        results (dict): Performance results
        filename (str): Output filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Algorithm', 'Avg_Time', 'Std_Time', 'Min_Time', 'Max_Time', 'Success_Rate'])
        
        for algorithm, metrics in results.items():
            writer.writerow([
                algorithm,
                metrics['avg_time'],
                metrics['std_time'],
                metrics['min_time'],
                metrics['max_time'],
                metrics['success_rate']
            ])

def print_csp_summary(variables, domains, constraints, analysis):
    """
    Print a summary of CSP analysis
    
    Args:
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
        analysis (dict): Analysis results
    """
    print("=" * 60)
    print("CONSTRAINT SATISFACTION PROBLEM SUMMARY")
    print("=" * 60)
    print(f"Variables: {analysis['num_variables']}")
    print(f"Constraints: {analysis['num_constraints']}")
    print(f"Average Domain Size: {analysis['avg_domain_size']:.2f}")
    print(f"Total Search Space: {analysis['total_search_space']:,}")
    print(f"Constraint Density: {analysis['constraint_density']:.3f}")
    print(f"Constraint Tightness: {analysis['constraint_tightness']:.3f}")
    print(f"Average Degree: {analysis['avg_degree']:.2f}")
    print(f"Maximum Degree: {analysis['max_degree']}")
    print("=" * 60)
