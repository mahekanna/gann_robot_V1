# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:36:20 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Demo script for Security Master functionality
"""

import argparse
import logging
import pandas as pd
from tabulate import tabulate
from typing import List, Dict, Any

from api.factory import get_security_master_instance

def display_results(results: List[Dict[str, Any]], title: str) -> None:
    """
    Display results in a nicely formatted table
    
    Args:
        results: List of dictionaries to display
        title: Table title
    """
    if not results:
        print(f"\n{title}: No results found")
        return
        
    print(f"\n{title} ({len(results)} results):")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)
    
    # Select columns to display (limit to most relevant)
    display_cols = []
    all_cols = df.columns.tolist()
    
    # Prioritize important columns
    priority_cols = ['Symbol', 'Security Id', 'Company Name', 'Exchange', 'Series', 
                   'Expiry Date', 'Strike Price', 'Option Type']
                   
    for col in priority_cols:
        if col in all_cols:
            display_cols.append(col)
            
    # Add additional columns if few results
    if len(results) <= 10:
        for col in all_cols:
            if col not in display_cols:
                display_cols.append(col)
                
    # Limit to first 8 columns
    display_cols = display_cols[:8]
    
    # Display table
    print(tabulate(df[display_cols].head(20), headers='keys', tablefmt='grid'))
    
    # If more results than displayed
    if len(results) > 20:
        print(f"... {len(results) - 20} more results not shown")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Security Master Demo')
    parser.add_argument('--search', type=str, help='Search for securities')
    parser.add_argument('--exchange', type=str, help='Filter by exchange')
    parser.add_argument('--symbol', type=str, help='Get details for a specific symbol')
    parser.add_argument('--token', type=str, help='Get symbol for a specific token')
    parser.add_argument('--underlying', type=str, help='Get option chain info for underlying')
    parser.add_argument('--reload', action='store_true', help='Force reload of security master')
    parser.add_argument('--limit', type=int, default=20, help='Limit number of search results')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get Security Master instance
    security_master = get_security_master_instance(force_new=args.reload, log_level=log_level)
    
    # Load security master (force download if requested)
    if not security_master.load_security_master(force_download=args.reload):
        print("Failed to load security master. Exiting.")
        return
    
    # Process commands
    if args.search:
        results = security_master.search_securities(
            query=args.search,
            exchange=args.exchange,
            limit=args.limit
        )
        display_results(results, f"Search results for '{args.search}'")
    
    if args.symbol:
        details = security_master.get_security_details(
            symbol=args.symbol,
            exchange=args.exchange or 'NSE'
        )
        
        if details:
            print(f"\nDetails for {args.symbol}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
                
            # Get token
            token = security_master.get_token(
                symbol=args.symbol,
                exchange=args.exchange or 'NSE'
            )
            if token:
                print(f"  Token: {token}")
        else:
            print(f"\nNo details found for {args.symbol}")
    
    if args.token:
        symbol_info = security_master.get_symbol(args.token)
        if symbol_info:
            print(f"\nSymbol for token {args.token}: {symbol_info['symbol']} ({symbol_info['exchange']})")
        else:
            print(f"\nNo symbol found for token {args.token}")
    
    if args.underlying:
        # Get expiry dates
        expiry_dates = security_master.get_expiry_dates(
            underlying=args.underlying,
            exchange=args.exchange or 'NFO'
        )
        
        if expiry_dates:
            print(f"\nExpiry dates for {args.underlying}:")
            for date in expiry_dates[:10]:  # Show only first 10
                print(f"  {date}")
                
            if len(expiry_dates) > 10:
                print(f"  ... {len(expiry_dates) - 10} more not shown")
                
            # Get strikes for first expiry
            strikes = security_master.get_option_chain_strikes(
                underlying=args.underlying,
                expiry_date=expiry_dates[0] if expiry_dates else None,
                exchange=args.exchange or 'NFO'
            )
            
            if strikes:
                print(f"\nStrike prices for {args.underlying} (expiry: {expiry_dates[0]}):")
                # Format in multiple columns
                strikes_per_row = 8
                for i in range(0, len(strikes), strikes_per_row):
                    row_strikes = strikes[i:i + strikes_per_row]
                    print("  " + " ".join(f"{strike:8.2f}" for strike in row_strikes))
        else:
            print(f"\nNo option chain data found for {args.underlying}")

if __name__ == "__main__":
    main()