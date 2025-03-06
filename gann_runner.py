# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:43:56 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Runner script for Gann Square of 9 calculations
"""

import sys
import argparse
from gann_square_of_9 import print_analysis_with_table

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Gann Square of 9 Analysis')
    parser.add_argument('--price', type=float, default=22127.4, help='Price to analyze (previous candle close)')
    parser.add_argument('--targets', type=int, default=8, help='Number of target levels to generate')
    parser.add_argument('--buffer', type=float, default=0.002, help='Buffer percentage for stoploss')
    
    args = parser.parse_args()
    
    print(f"Running Gann Square of 9 analysis for price: {args.price}")
    print(f"Number of targets: {args.targets}")
    print(f"Buffer percentage: {args.buffer}")
    
    print_analysis_with_table(args.price, args.targets, args.buffer)

if __name__ == "__main__":
    main()