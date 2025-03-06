"""
Gann Square of 9 Test and Visualization Runner
This script runs the test suite and visualization tools for the Gann Square of 9 implementation.
"""

import os
import sys
import argparse
from typing import List


def setup_environment() -> None:
    """Create required directories if they don't exist"""
    dirs = ['core', 'core/gann', 'tests', 'tools']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {directory.replace('/', '.')} package\n")


def run_comparison_tests() -> None:
    """Run the comparison tests between original and new implementation"""
    print("Running Gann implementation comparison tests...")
    test_file = 'tests/test_gann_implementation.py'
    if os.path.exists(test_file):
        os.system(f"python {test_file}")
    else:
        print(f"Error: Test file {test_file} not found.")


def run_visualization(price: float = None) -> None:
    """Run the Gann visualization tools"""
    print("Running Gann visualization tools...")
    
    viz_file = 'tools/gann_visualization.py'
    if os.path.exists(viz_file):
        if price:
            os.system(f"python {viz_file} --price {price}")
        else:
            os.system(f"python {viz_file}")
    else:
        print(f"Error: Visualization file {viz_file} not found.")


def main(args: List[str] = None) -> None:
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Gann Square of 9 Test and Visualization Runner')
    parser.add_argument('--tests', action='store_true', help='Run comparison tests')
    parser.add_argument('--visualize', action='store_true', help='Run visualization tools')
    parser.add_argument('--price', type=float, help='Specific price for Gann wheel visualization')
    
    parsed_args = parser.parse_args(args)
    
    # Setup environment
    setup_environment()
    
    # Run selected tools
    if parsed_args.tests or not (parsed_args.tests or parsed_args.visualize):
        run_comparison_tests()
        
    if parsed_args.visualize or not (parsed_args.tests or parsed_args.visualize):
        run_visualization(parsed_args.price)
        

if __name__ == "__main__":
    main()