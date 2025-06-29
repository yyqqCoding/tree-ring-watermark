"""
Simple verification script for CoSDA implementation
This script checks if the CoSDA modules can be imported correctly.
"""

import sys
import os

def test_imports():
    """Test if all CoSDA modules can be imported."""
    print("Testing CoSDA module imports...")
    
    try:
        # Test basic Python imports
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        # Test CoSDA module structure
        import CoSDA
        print("✓ CoSDA package imported successfully")
        
        # Test individual modules
        from CoSDA.compensation_sampling import CompensationSampler
        print("✓ CompensationSampler imported successfully")
        
        from CoSDA.drift_alignment import DriftAlignmentNetwork
        print("✓ DriftAlignmentNetwork imported successfully")
        
        from CoSDA.schedulers import CoSDADDIMScheduler
        print("✓ CoSDADDIMScheduler imported successfully")
        
        from CoSDA.utils import create_distortions, evaluate_inversion_error
        print("✓ Utility functions imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ CoSDA import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without PyTorch."""
    print("\nTesting basic functionality...")
    
    try:
        # Test CompensationSampler initialization
        from CoSDA.compensation_sampling import CompensationSampler
        sampler = CompensationSampler(p=0.8)
        assert sampler.p == 0.8
        print("✓ CompensationSampler initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def check_file_structure():
    """Check if all required files exist."""
    print("\nChecking file structure...")
    
    required_files = [
        "__init__.py",
        "compensation_sampling.py",
        "drift_alignment.py",
        "schedulers.py",
        "cosda_pipeline.py",
        "utils.py",
        "tree_ring_integration.py",
        "train_drift_alignment.py",
        "demo_cosda_tree_ring.py",
        "README.md"
    ]
    
    cosda_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_exist = True
    for file in required_files:
        file_path = os.path.join(cosda_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - Missing")
            all_exist = False
    
    return all_exist

def main():
    """Main verification function."""
    print("=" * 60)
    print("CoSDA Implementation Verification")
    print("=" * 60)
    
    # Check file structure
    files_ok = check_file_structure()
    
    # Test imports (may fail if PyTorch not installed)
    imports_ok = test_imports()
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("Verification Results:")
    print(f"File structure: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Module imports: {'✓ PASS' if imports_ok else '✗ FAIL (may need PyTorch)'}")
    print(f"Basic functionality: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print("=" * 60)
    
    if files_ok and basic_ok:
        print("🎉 CoSDA implementation structure is correct!")
        if not imports_ok:
            print("💡 Install PyTorch and other dependencies to run full tests.")
    else:
        print("⚠️ Some issues found in the implementation.")
    
    return files_ok and basic_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
