# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:10:22 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Demo for ICICI Direct API
Tests the authentication, market data, and security master functionality
"""

import logging
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from api.factory import get_auth, get_security_master, get_client

def setup_logging(log_level):
    """Setup logging"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/api_demo.log"),
            logging.StreamHandler()
        ]
    )

def authenticate():
    """Authenticate with ICICI Direct API"""
    print("\n=== Testing Authentication ===")
    auth = get_auth()
    
    # Initialize breeze
    if not auth.initialize_breeze():
        print("Failed to initialize BreezeConnect. Make sure breeze-connect is installed.")
        return False
        
    # Get session
    if not auth.get_session():
        print("Failed to authenticate. Please check your credentials and try again.")
        return False
        
    print("Authentication successful!")
    return True

def test_security_master():
    """Test security master functionality"""
    print("\n=== Testing Security Master ===")
    security_master = get_security_master()
    
    # Load security master
    print("Loading security master (this may take a while for first run)...")
    if not security_master.load_security_master():
        print("Failed to load security master. Continuing without it.")
        return False
        
    # Test searching
    print("\nSearching for 'RELI'...")
    results = security_master.search_securities("RELI", limit=5)
    if results:
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  {result['Symbol']} - {result.get('Company Name', 'N/A')}")
    else:
        print("No results found.")
        
    # Test getting token
    test_symbol = "NIFTY"
    print(f"\nGetting token for {test_symbol}...")
    token = security_master.get_token(test_symbol, "NSE")
    if token:
        print(f"Token for {test_symbol}: {token}")
    else:
        print(f"Token not found for {test_symbol}")
        
    return True

def test_market_data():
    """Test market data functionality"""
    print("\n=== Testing Market Data ===")
    client = get_client()
    
    # Connect to API
    if not client.connect():
        print("Failed to connect to API. Please check your credentials and try again.")
        return False
        
    # Test getting historical data
    test_symbol = "NIFTY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nFetching historical data for {test_symbol}...")
    df = client.get_historical_data(
        stock_code=test_symbol,
        exchange_code="NSE",
        from_date=start_date,
        to_date=end_date,
        interval="15minute",
        indices=True
    )
    
    if df.empty:
        print("No historical data found.")
    else:
        print(f"Fetched {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
        print("\nFirst 5 records:")
        print(df.head())
        
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(df['datetime'], df['close'])
        plt.title(f"{test_symbol} - 15minute Chart")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{test_symbol}_chart.png")
        print(f"Chart saved to output/{test_symbol}_chart.png")
        
        # Show the plot if requested
        plt.show()
        
    # Test getting quote
    print(f"\nFetching quote for {test_symbol}...")
    quote = client.get_quote(test_symbol, "NSE", indices=True)
    
    if quote:
        print("Quote:")
        for key, value in sorted(quote.items())[:10]:  # Show first 10 items
            print(f"  {key}: {value}")
            
        if len(quote) > 10:
            print(f"  ... ({len(quote) - 10} more items)")
    else:
        print("Failed to fetch quote.")
        
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ICICI Direct API Demo')
    parser.add_argument('--auth-only', action='store_true', help='Only test authentication')
    parser.add_argument('--security-master-only', action='store_true', help='Only test security master')
    parser.add_argument('--market-data-only', action='store_true', help='Only test market data')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    # Create directories
    Path('logs').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    # Run tests
    if args.auth_only:
        authenticate()
    elif args.security_master_only:
        test_security_master()
    elif args.market_data_only:
        test_market_data()
    else:
        # Run all tests
        if authenticate():
            test_security_master()
            test_market_data()

if __name__ == "__main__":
    main()