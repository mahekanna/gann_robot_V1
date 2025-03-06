# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:48:29 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Demo script for Gann Square of 9 visualizations
"""

import argparse
import os
from gann_visualizer import GannVisualizer
from gann_square_of_9 import print_analysis_with_table

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Gann Square of 9 Visualization Demo')
    parser.add_argument('--price', type=float, default=22127.4, help='Price to analyze (previous candle close)')
    parser.add_argument('--targets', type=int, default=3, help='Number of target levels to generate')
    parser.add_argument('--buffer', type=float, default=0.002, help='Buffer percentage for stoploss')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualization images')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    # Add visualization specific options
    visualizations = parser.add_argument_group('visualizations')
    visualizations.add_argument('--levels', action='store_true', help='Plot price levels, targets and stop loss')
    visualizations.add_argument('--wheel', action='store_true', help='Plot Gann wheel')
    visualizations.add_argument('--square', action='store_true', help='Plot Gann square grid')
    visualizations.add_argument('--all', action='store_true', help='Plot all visualizations')
    
    args = parser.parse_args()
    
    # If no specific visualization is selected, default to all
    if not (args.levels or args.wheel or args.square):
        args.all = True
    
    # Create save directory if specified
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Print analysis first
    print(f"Running Gann Square of 9 analysis for price: {args.price}")
    results = print_analysis_with_table(args.price, args.targets, args.buffer)
    
    if not results:
        print("Error: Could not generate analysis. Exiting.")
        return
    
    # Create visualizer
    visualizer = GannVisualizer()
    
    # Plot based on arguments
    if args.levels or args.all:
        print("\nGenerating price levels visualization...")
        save_path = os.path.join(args.save_dir, f"gann_levels_{args.price}.png") if args.save_dir else None
        visualizer.plot_price_levels(args.price, args.targets, args.buffer, not args.no_show, save_path)
    
    if args.wheel or args.all:
        print("\nGenerating Gann wheel visualization...")
        save_path = os.path.join(args.save_dir, f"gann_wheel_{args.price}.png") if args.save_dir else None
        visualizer.plot_gann_wheel(args.price, 8, not args.no_show, save_path)
    
    if args.square or args.all:
        print("\nGenerating Gann square visualization...")
        save_path = os.path.join(args.save_dir, f"gann_square_{args.price}.png") if args.save_dir else None
        visualizer.plot_gann_square(args.price, 15, not args.no_show, save_path)

if __name__ == "__main__":
    main()