# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:30:09 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Demo script for Gann Square of 9 calculations and visualization
"""

import sys
import os
import argparse
import math
from typing import Dict, List, Optional
from gann_square_of_9 import GannSquareOf9
from gann_visualizer import GannVisualizer

def print_gann_table(price: float, gann: GannSquareOf9) -> None:
    """
    Print Gann Square of 9 table
    
    Args:
        price: Price to calculate Gann values from
        gann: Gann calculator instance
    """
    gann_values = gann.calculate_levels(price)
    
    print("\n=== Gann Square of 9 Table ===")
    print(f"Price: {price:.2f}")
    print("\n" + gann.format_table(gann_values))

def print_signal_analysis(current_price: float, previous_close: float, gann: GannSquareOf9) -> None:
    """
    Print Gann signal analysis
    
    Args:
        current_price: Current market price
        previous_close: Previous candle's closing price
        gann: Gann calculator instance
    """
    signals = gann.generate_signals(current_price, previous_close)
    
    print("\n=== Gann Signal Analysis ===")
    print(f"Current Price: {signals['current_price']:.2f}")
    print(f"Previous Close: {signals['previous_close']:.2f}")
    print(f"Buy Above: {signals['buy_above']:.2f if signals['buy_above'] else 'N/A'}")
    print(f"Sell Below: {signals['sell_below']:.2f if signals['sell_below'] else 'N/A'}")
    
    print("\nSignals:")
    if signals['long_signal']:
        print("  LONG SIGNAL ACTIVE (Current price is above Buy Above level)")
    if signals['short_signal']:
        print("  SHORT SIGNAL ACTIVE (Current price is below Sell Below level)")
    if not signals['long_signal'] and not signals['short_signal']:
        print("  No active signals")
    
    print("\nTargets:")
    if signals['buy_targets']:
        print("  Buy Targets:")
        for i, (angle, price) in enumerate(signals['buy_targets']):
            print(f"    Target {i+1}: {price:.2f} ({angle})")
    else:
        print("  No buy targets available")
        
    if signals['sell_targets']:
        print("  Sell Targets:")
        for i, (angle, price) in enumerate(signals['sell_targets']):
            print(f"    Target {i+1}: {price:.2f} ({angle})")
    else:
        print("  No sell targets available")
    
    print("\nStop Loss Levels:")
    print(f"  Long Stop Loss: {signals['long_stoploss']:.2f if signals['long_stoploss'] else 'N/A'}")
    print(f"  Short Stop Loss: {signals['short_stoploss']:.2f if signals['short_stoploss'] else 'N/A'}")

def run_demo(price: float, show_visualizations: bool = True) -> None:
    """
    Run a complete Gann Square of 9 demo
    
    Args:
        price: Base price for calculations
        show_visualizations: Whether to show visualizations
    """
    # Initialize calculator and visualizer
    gann = GannSquareOf9()
    visualizer = GannVisualizer(gann)
    
    # Print Gann table
    print_gann_table(price, gann)
    
    # Show signal analysis
    # For demo, use current price slightly above the price to generate a LONG signal
    current_price = price * 1.01
    previous_close = price
    print_signal_analysis(current_price, previous_close, gann)
    
    # Show visualizations if requested
    if show_visualizations:
        print("\n=== Generating Visualizations ===")
        print("Generating Gann Wheel...")
        visualizer.plot_gann_wheel(price)
        
        print("Generating Gann Square...")
        visualizer.plot_gann_square(price)
        
        print("Generating Signal Analysis...")
        visualizer.visualize_signals(current_price, previous_close)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Gann Square of 9 Demo')
    parser.add_argument('--price', type=float, default=22500.85, help='Base price for calculations')
    parser.add_argument('--current', type=float, help='Current price (defaults to price * 1.01)')
    parser.add_argument('--previous', type=float, help='Previous close price (defaults to price)')
    parser.add_argument('--no-visualization', action='store_true', help='Do not show visualizations')
    
    args = parser.parse_args()
    
    # Initialize Gann calculator and visualizer
    gann = GannSquareOf9()
    visualizer = GannVisualizer(gann)
    
    # Set default values for current and previous prices if not provided
    current_price = args.current if args.current is not None else args.price * 1.01
    previous_close = args.previous if args.previous is not None else args.price
    
    # Print Gann table
    print_gann_table(args.price, gann)
    
    # Show signal analysis
    print_signal_analysis(current_price, previous_close, gann)
    
    # Show visualizations if not disabled
    if not args.no_visualization:
        print("\n=== Generating Visualizations ===")
        print("Generating Gann Wheel...")
        visualizer.plot_gann_wheel(args.price)
        
        print("Generating Gann Square...")
        visualizer.plot_gann_square(args.price)
        
        print("Generating Signal Analysis...")
        visualizer.visualize_signals(current_price, previous_close)

if __name__ == "__main__":
    main()