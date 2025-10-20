# main.py
#
# Simple main script to run basic CSP examples
# This demonstrates the basic functionality before students complete the assignment

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_basic_examples():
    """Run basic CSP examples"""
    print("🧩 Constraint Satisfaction Problems - Basic Examples")
    print("=" * 60)
    
    try:
        from csp_generator import generate_csp
        from csp_inference import backtrack, arc_consistency
        from csp_utils import load_sample_csp_problems, analyze_csp_structure, print_csp_summary
        
        # Example 1: Generate and solve a random CSP
        print("\n📋 Example 1: Random CSP Generation and Solving")
        print("-" * 50)
        
        variables, domains, constraints = generate_csp(5, 0.3, 0.8, 0.3)
        print(f"Generated CSP with {len(variables)} variables")
        print(f"Variables: {variables}")
        print(f"Domain sizes: {[len(domains[v]) for v in variables]}")
        print(f"Number of constraints: {len(constraints)}")
        
        # Analyze the CSP
        analysis = analyze_csp_structure(variables, domains, constraints)
        print_csp_summary(variables, domains, constraints, analysis)
        
        # Try to solve the CSP
        print("\nSolving CSP...")
        solution = backtrack({}, variables, domains, constraints)
        if solution and solution != "Failure":
            print(f"✅ Solution found: {solution}")
        else:
            print("❌ No solution found")
        
        # Example 2: Load sample problems
        print("\n📋 Example 2: Sample CSP Problems")
        print("-" * 50)
        
        sample_problems = load_sample_csp_problems()
        for name, problem in sample_problems.items():
            variables, domains, constraints = problem
            print(f"\n{name}:")
            print(f"  Variables: {variables}")
            print(f"  Number of variables: {len(variables)}")
            print(f"  Number of constraints: {len(constraints)}")
        
        # Example 3: Arc consistency
        print("\n📋 Example 3: Arc Consistency")
        print("-" * 50)
        
        # Create a simple CSP for arc consistency testing
        variables = [0, 1, 2]
        domains = {0: [1, 2], 1: [1, 2], 2: [1, 2]}
        constraints = {(0, 1): [(1, 1)], (1, 2): [(2, 2)]}  # X0≠X1, X1≠X2
        
        print(f"Before AC: {domains}")
        result = arc_consistency(variables, domains, constraints)
        print(f"After AC: {domains}")
        print(f"AC result: {result}")
        
        print("\n✅ Basic examples completed successfully!")
        print("\nNext steps:")
        print("1. Complete the full assignment in src/constraint_satisfaction_project.py")
        print("2. Run: python src/constraint_satisfaction_project.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Check the setup and try again.")

if __name__ == "__main__":
    run_basic_examples()
