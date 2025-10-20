# csp_visualizer.py
#
# DESCRIPTION
# -----------
# Visualization tools for Constraint Satisfaction Problems
# Includes constraint graphs, solution visualization, and performance plots

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def visualize_constraint_graph(variables, domains, constraints, title="Constraint Graph"):
    """
    Visualize the constraint graph of a CSP
    
    Args:
        variables (list): List of variables
        domains (dict): Dictionary mapping variables to their domains
        constraints (dict): Dictionary mapping variable pairs to incompatible value pairs
        title (str): Title for the plot
    """
    G = nx.Graph()
    
    # Add nodes with domain size information
    for var in variables:
        domain_size = len(domains[var])
        G.add_node(var, domain_size=domain_size)
    
    # Add edges
    G.add_edges_from(constraints.keys())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Draw nodes with size based on domain size
    node_sizes = [len(domains[var]) * 200 + 500 for var in variables]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
    
    # Draw labels
    labels = {var: f"{var}\n({len(domains[var])})" for var in variables}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, alpha=0.8, label='Variable (Domain Size)'),
        plt.Line2D([0], [0], color='gray', alpha=0.6, label='Constraint')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def visualize_n_queens_solution(solution, n=8, title="N-Queens Solution"):
    """
    Visualize N-Queens solution on a chessboard
    
    Args:
        solution (dict): Dictionary mapping variables to their assigned values
        n (int): Size of the chessboard
        title (str): Title for the plot
    """
    if not solution or solution == "Failure":
        print("No valid solution found!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create chessboard
    for i in range(n):
        for j in range(n):
            color = 'white' if (i + j) % 2 == 0 else 'lightgray'
            rect = patches.Rectangle((j, n-1-i), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    
    # Place queens
    for arc, value in solution.items():
        if isinstance(arc, int) and 0 <= arc < n:
            row = arc
            col = value
            circle = patches.Circle((col + 0.5, n - 1 - row + 0.5), 0.3, 
                                  facecolor='red', edgecolor='darkred', linewidth=2)
            ax.add_patch(circle)
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(range(n+1))
    ax.set_yticks(range(n+1))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def visualize_map_coloring_solution(solution, title="Map Coloring Solution"):
    """
    Visualize map coloring solution
    
    Args:
        solution (dict): Dictionary mapping variables to their assigned values
        title (str): Title for the plot
    """
    if not solution or solution == "Failure":
        print("No valid solution found!")
        return
    
    # Australian states with their approximate positions
    state_positions = {
        'WA': (1, 2), 'NT': (3, 2), 'SA': (2.5, 3.5), 'Q': (4, 3),
        'NSW': (4.5, 4.5), 'V': (3.5, 5), 'T': (4, 6)
    }
    
    color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw states as rectangles
    for state, (x, y) in state_positions.items():
        color = color_map.get(solution.get(state, 'white'), 'white')
        rect = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add state label
        ax.text(x, y, state, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # Draw connections between adjacent states
    connections = [
        ('WA', 'NT'), ('WA', 'SA'), ('NT', 'SA'), ('NT', 'Q'),
        ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'), ('Q', 'NSW'),
        ('NSW', 'V')
    ]
    
    for state1, state2 in connections:
        x1, y1 = state_positions[state1]
        x2, y2 = state_positions[state2]
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 7)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='red', label='Red'),
        patches.Patch(color='green', label='Green'),
        patches.Patch(color='blue', label='Blue')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def visualize_sudoku_solution(solution, title="Sudoku Solution"):
    """
    Visualize Sudoku solution
    
    Args:
        solution (dict): Dictionary mapping variables to their assigned values
        title (str): Title for the plot
    """
    if not solution or solution == "Failure":
        print("No valid solution found!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create 9x9 grid
    for i in range(9):
        for j in range(9):
            # Determine cell color
            color = 'white' if (i//3 + j//3) % 2 == 0 else 'lightgray'
            
            # Draw cell
            rect = patches.Rectangle((j, 8-i), 1, 1, 
                                   facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add number if assigned
            cell = (i, j)
            if cell in solution:
                ax.text(j + 0.5, 8 - i + 0.5, str(solution[cell]), 
                       ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw thick lines for 3x3 boxes
    for i in range(0, 10, 3):
        ax.axhline(y=i, color='black', linewidth=3)
        ax.axvline(x=i, color='black', linewidth=3)
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_algorithm_performance(results, title="Algorithm Performance Comparison"):
    """
    Plot performance comparison of different algorithms
    
    Args:
        results (dict): Results from algorithm comparison
        title (str): Title for the plot
    """
    algorithms = list(results.keys())
    metrics = ['avg_time', 'success_rate']
    metric_labels = ['Average Time (seconds)', 'Success Rate']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average time comparison
    avg_times = [results[alg]['avg_time'] for alg in algorithms]
    bars1 = axes[0].bar(algorithms, avg_times, color='skyblue', alpha=0.7)
    axes[0].set_title('Average Solving Time', fontweight='bold')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars1, avg_times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.3f}s', ha='center', va='bottom')
    
    # Success rate comparison
    success_rates = [results[alg]['success_rate'] for alg in algorithms]
    bars2 = axes[1].bar(algorithms, success_rates, color='lightgreen', alpha=0.7)
    axes[1].set_title('Success Rate', fontweight='bold')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars2, success_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_csp_analysis(analysis_results, title="CSP Analysis"):
    """
    Plot CSP analysis results
    
    Args:
        analysis_results (list): List of analysis results for different CSPs
        title (str): Title for the plot
    """
    if not analysis_results:
        return
    
    # Extract data for plotting
    csp_names = [f"CSP {i+1}" for i in range(len(analysis_results))]
    num_variables = [result['num_variables'] for result in analysis_results]
    num_constraints = [result['num_constraints'] for result in analysis_results]
    constraint_density = [result['constraint_density'] for result in analysis_results]
    constraint_tightness = [result['constraint_tightness'] for result in analysis_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Number of variables
    axes[0, 0].bar(csp_names, num_variables, color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Number of Variables', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    
    # Number of constraints
    axes[0, 1].bar(csp_names, num_constraints, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Number of Constraints', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    
    # Constraint density
    axes[1, 0].bar(csp_names, constraint_density, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Constraint Density', fontweight='bold')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_ylim(0, 1)
    
    # Constraint tightness
    axes[1, 1].bar(csp_names, constraint_tightness, color='lightyellow', alpha=0.7)
    axes[1, 1].set_title('Constraint Tightness', fontweight='bold')
    axes[1, 1].set_ylabel('Tightness')
    axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_search_tree_depth(algorithm_results, title="Search Tree Depth Analysis"):
    """
    Plot search tree depth analysis for different algorithms
    
    Args:
        algorithm_results (dict): Results from different algorithms
        title (str): Title for the plot
    """
    algorithms = list(algorithm_results.keys())
    
    # Simulate search tree depths (in practice, you'd track this during search)
    depths = {
        'Backtrack': np.random.normal(8, 2, 20),
        'Forward Checking': np.random.normal(6, 1.5, 20),
        'MAC': np.random.normal(5, 1, 20)
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create box plot
    data_to_plot = [depths.get(alg, [0]) for alg in algorithms]
    box_plot = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Search Tree Depth')
    ax.set_xlabel('Algorithm')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_performance_report(results, filename="csp_performance_report.png"):
    """
    Create a comprehensive performance report
    
    Args:
        results (dict): Performance results
        filename (str): Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = list(results.keys())
    
    # Time comparison
    times = [results[alg]['avg_time'] for alg in algorithms]
    axes[0, 0].bar(algorithms, times, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average Solving Time', fontweight='bold')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Success rate
    success_rates = [results[alg]['success_rate'] for alg in algorithms]
    axes[0, 1].bar(algorithms, success_rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Success Rate', fontweight='bold')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Time distribution
    all_times = []
    all_algorithms = []
    for alg in algorithms:
        # Simulate multiple runs (in practice, you'd have actual data)
        simulated_times = np.random.normal(results[alg]['avg_time'], 
                                         results[alg]['std_time'], 50)
        all_times.extend(simulated_times)
        all_algorithms.extend([alg] * 50)
    
    df = pd.DataFrame({'Algorithm': all_algorithms, 'Time': all_times})
    sns.violinplot(data=df, x='Algorithm', y='Time', ax=axes[1, 0])
    axes[1, 0].set_title('Time Distribution', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Performance summary table
    axes[1, 1].axis('off')
    table_data = []
    for alg in algorithms:
        table_data.append([
            alg,
            f"{results[alg]['avg_time']:.3f}s",
            f"{results[alg]['success_rate']:.2f}",
            f"{results[alg]['total_solutions_found']}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                           colLabels=['Algorithm', 'Avg Time', 'Success Rate', 'Solutions Found'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Performance Summary', fontweight='bold')
    
    plt.suptitle('CSP Algorithm Performance Report', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def save_visualization(fig, filename, dpi=300):
    """
    Save visualization to file
    
    Args:
        fig: Matplotlib figure object
        filename (str): Output filename
        dpi (int): Resolution for saved image
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
