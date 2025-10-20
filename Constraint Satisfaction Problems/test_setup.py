# test_setup.py
#
# Simple test script to verify the CSP assignment setup

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from csp_generator import generate_csp
        print("✅ csp_generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import csp_generator: {e}")
        return False
    
    try:
        from csp_inference import backtrack, arc_consistency
        print("✅ csp_inference imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import csp_inference: {e}")
        return False
    
    try:
        from csp_utils import load_sample_csp_problems, analyze_csp_structure
        print("✅ csp_utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import csp_utils: {e}")
        return False
    
    try:
        from csp_visualizer import visualize_constraint_graph
        print("✅ csp_visualizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import csp_visualizer: {e}")
        return False
    
    return True

def test_csp_generation():
    """Test CSP generation"""
    print("\nTesting CSP generation...")
    
    try:
        from csp_generator import generate_csp
        
        # Generate a small CSP
        variables, domains, constraints = generate_csp(4, 0.3, 0.8, 0.3)
        
        print(f"✅ Generated CSP with {len(variables)} variables")
        print(f"✅ Domains: {domains}")
        print(f"✅ Constraints: {len(constraints)} constraint pairs")
        
        return True
    except Exception as e:
        print(f"❌ CSP generation failed: {e}")
        return False

def test_csp_solving():
    """Test CSP solving"""
    print("\nTesting CSP solving...")
    
    try:
        from csp_generator import generate_csp
        from csp_inference import backtrack
        
        # Generate and solve a small CSP
        variables, domains, constraints = generate_csp(3, 0.2, 0.8, 0.3)
        solution = backtrack({}, variables, domains, constraints)
        
        if solution and solution != "Failure":
            print(f"✅ Found solution: {solution}")
        else:
            print("⚠️ No solution found (this may be normal for some CSPs)")
        
        return True
    except Exception as e:
        print(f"❌ CSP solving failed: {e}")
        return False

def test_sample_problems():
    """Test sample problems loading"""
    print("\nTesting sample problems...")
    
    try:
        from csp_utils import load_sample_csp_problems
        
        problems = load_sample_csp_problems()
        print(f"✅ Loaded {len(problems)} sample problems")
        
        for name, problem in problems.items():
            print(f"  - {name}: {len(problem['variables'])} variables")
        
        return True
    except Exception as e:
        print(f"❌ Sample problems loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧩 CSP Assignment Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test CSP generation
    if not test_csp_generation():
        all_tests_passed = False
    
    # Test CSP solving
    if not test_csp_solving():
        all_tests_passed = False
    
    # Test sample problems
    if not test_sample_problems():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✅ All tests passed! Setup is ready.")
        print("You can now start working on the assignment.")
    else:
        print("❌ Some tests failed. Please check the setup.")
    
    print("\nNext steps:")
    print("1. Complete the assignment in src/constraint_satisfaction_project.py")
    print("2. Run the grading script: python grade_script.py")

if __name__ == "__main__":
    main()

