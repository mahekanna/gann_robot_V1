# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:23:49 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Simple test for Gann Square of 9 calculation
"""

import math

def simple_gann_calc(price):
    """A very simple Gann calculation to test the environment"""
    root = math.sqrt(price)
    base = math.floor(root)
    central_value = base * base
    
    # Calculate a few values above and below
    values = []
    for i in range(-3, 4):
        val = base + i
        squared = val * val
        values.append(squared)
    
    return {
        'root': root,
        'base': base,
        'central_value': central_value,
        'values': values
    }

def test_simple_calc():
    """Test the simple calculation with a few test prices"""
    test_prices = [22510.85, 1000.0, 9999.99]
    
    for price in test_prices:
        result = simple_gann_calc(price)
        print(f"\nTesting price: {price}")
        print(f"Square root: {result['root']}")
        print(f"Base (floor of root): {result['base']}")
        print(f"Central value (base²): {result['central_value']}")
        print(f"Values around central value: {result['values']}")

if __name__ == "__main__":
    print("Running simple Gann Square of 9 test...")
    test_simple_calc()
    print("\nTest completed successfully!")