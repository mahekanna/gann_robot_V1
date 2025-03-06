#!/usr/bin/env python3
"""
Simple runner script for Gann Square of 9 tests
"""

import os
import sys

def setup_environment():
    """Ensure required files and directories exist"""
    # First, run the file generator if necessary
    if not os.path.exists("core/gann/square_of_9.py"):
        print("Generating necessary files...")
        if os.path.exists("generate_gann_files.py"):
            os.system("python generate_gann_files.py")
        else:
            print("Error: generate_gann_files.py not found")
            return False
    return True

def run_tests():
    """Run the test script"""
    if os.path.exists("tests/test_gann_implementation.py"):
        print("Running Gann implementation tests...")
        os.system("python tests/test_gann_implementation.py")
    else:
        print("Error: test script not found at tests/test_gann_implementation.py")

def main():
    """Main function"""
    if setup_environment():
        run_tests()
    else:
        print("Failed to set up the environment. Tests cannot be run.")

if __name__ == "__main__":
    main()