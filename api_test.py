# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 20:47:45 2025

@author: mahes
"""

# File: api_test.py
"""
Test script for ICICI Direct API integration
"""

import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

from api.factory import get_auth, get_client
from utils.logger import setup_logger

def test_auth():
    """Test authentication"""
    logger = setup_logger("api_test", log_level="INFO")
    logger.info("Testing authentication...")
    
    auth = get_auth()
    if auth.get_session():
        logger.info("Authentication successful")
        return True
    else:
        logger.error("Authentication failed")
        return False

def test_historical_data(symbol: str, days: int = 7, indices: bool = False):
    """
    Test retrieving historical data
    
    Args:
        symbol: Trading symbol
        days: Number of days of data to retrieve
        indices: Whether symbol is an index
    """
    logger = setup_logger("api_test", log_level="INFO")
    logger.info(f"Testing historical data for {symbol}...")
    
    # Get client
    client = get_client()
    
    # Connect to API
    if not client.connect():
        logger.error("Failed to connect to API")
        return
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get historical data
    df = client.get_historical_data(
        stock_code=symbol,
        exchange_code="NSE",
        from_date=start_date,
        to_date=end_date,
        interval="1day",
        indices=indices
    )
    
    if df.empty:
        logger.error("No historical data retrieved")
        return
    
    logger.info(f"Retrieved {len(df)} rows of data")
    print("\nHistorical Data:")
    print(df.head())
    
    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['close'])
    plt.title(f"{symbol} - Close Price")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def test_quote(symbol: str, indices: bool = False):
    """
    Test retrieving a quote
    
    Args:
        symbol: Trading symbol
        indices: Whether symbol is an index
    """
    logger = setup_logger("api_test", log_level="INFO")
    logger.info(f"Testing quote for {symbol}...")
    
    # Get client
    client = get_client()
    
    # Connect to API
    if not client.connect():
        logger.error("Failed to connect to API")
        return
    
    # Get quote
    quote = client.get_quote(symbol, "NSE", indices=indices)
    
    if not quote:
        logger.error("Failed to retrieve quote")
        return
    
    print("\nQuote:")
    for key, value in quote.items():
        print(f"  {key}: {value}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ICICI Direct API Test')
    parser.add_argument('--auth-only', action='store_true', help='Test authentication only')
    parser.add_argument('--symbol', type=str, default='NIFTY', help='Symbol to test')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data')
    parser.add_argument('--indices', action='store_true', help='Symbol is an index')
    
    args = parser.parse_args()
    
    if args.auth_only:
        test_auth()
    else:
        if test_auth():
            test_quote(args.symbol, args.indices)
            test_historical_data(args.symbol, args.days, args.indices)

if __name__ == "__main__":
    main()