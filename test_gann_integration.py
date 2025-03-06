# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:12:25 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Test script to verify proper integration of Gann Square of 9 module
into the project structure
"""

import os
import sys
import argparse

# Ensure proper directory structure
os.makedirs('core/gann', exist_ok=True)
os.makedirs('core/strategy', exist_ok=True)

# Create __init__.py files if they don't exist
for path in ['core', 'core/gann', 'core/strategy']:
    init_file = os.path.join(path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write(f"# {path.replace('/', '.')} package\n")

# Add project root to path
sys.path.append('.')

# Now import from the proper structure
from core.gann.square_of_9 import GannSquareOf9
from core.gann.visualization import GannVisualizer

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Test Gann Square of 9 Integration')
    parser.add_argument('--price', type=float, default=22127.4, help='Price to analyze')
    parser.add_argument('--targets', type=int, default=3, help='Number of target levels')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    args = parser.parse_args()
    
    # Create Gann calculator and get analysis
    gann = GannSquareOf9()
    analysis = gann.get_analysis_report(args.price, args.targets)
    
    if not analysis:
        print(f"Could not generate analysis for price {args.price}")
        return
    
    # Print analysis
    print("\n=== Gann Square of 9 Analysis ===")
    print(f"Price: {analysis['price']:.2f}")
    print(f"Buy above: {analysis['buy_above']:.2f}")
    print(f"Sell below: {analysis['sell_below']:.2f}")
    
    print("\nBuy Targets:")
    for i, (angle, price) in enumerate(analysis['buy_targets']):
        print(f"  Target {i+1}: {price:.2f} ({angle})")
    
    print("\nSell Targets:")
    for i, (angle, price) in enumerate(analysis['sell_targets']):
        print(f"  Target {i+1}: {price:.2f} ({angle})")
    
    print(f"\nLong Stoploss: {analysis['long_stoploss']:.2f}")
    print(f"Short Stoploss: {analysis['short_stoploss']:.2f}")
    
    # Show visualizations if requested
    if args.visualize:
        visualizer = GannVisualizer(gann)
        print("\nShowing price levels...")
        visualizer.plot_price_levels(args.price, args.targets)
        
        print("\nShowing Gann wheel...")
        visualizer.plot_gann_wheel(args.price)
        
        print("\nShowing Gann square...")
        visualizer.plot_gann_square(args.price)

if __name__ == "__main__":
    main()