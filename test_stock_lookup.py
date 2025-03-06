# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:39:14 2025

@author: mahes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Stock Lookup client
"""

import sys
from pathlib import Path
from stock_lookup import StockLookup

def test_stock_lookup():
    """Test stock lookup functionality"""
    # Initialize client
    lookup = StockLookup()
    
    # Test stock symbols
    test_stocks = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN",
        "NIFTY", "BANKNIFTY", "TATASTEEL", "ICICIBANK"
    ]
    
    print("\n=== NSE STOCK TOKENS ===")
    for symbol in test_stocks:
        token = lookup.get_stock_token(symbol)
        if token:
            details = lookup.get_token_details(token)
            print(f"✅ {symbol}: {token}" + (f" ({details['type']})" if details else ""))
        else:
            print(f"❌ {symbol}: Not found")
    
    # Test option lookup
    print("\n=== OPTION CHAIN LOOKUP ===")
    for underlying in ["NIFTY", "BANKNIFTY", "RELIANCE"]:
        # Get expiry dates
        expiry_dates = lookup.get_expiry_dates(underlying)
        if not expiry_dates:
            print(f"❌ {underlying}: No expiry dates found")
            continue
        
        # Use first expiry
        expiry = expiry_dates[0]
        print(f"\n{underlying} (Expiry: {expiry}):")
        
        # Get option chain
        chain = lookup.get_option_chain(underlying, expiry)
        
        # Count options
        ce_count = len(chain.get('CE', {}))
        pe_count = len(chain.get('PE', {}))
        
        if ce_count or pe_count:
            print(f"  Found {ce_count} call options and {pe_count} put options")
            
            # Show some strike prices
            ce_strikes = list(chain.get('CE', {}).keys())[:5]
            pe_strikes = list(chain.get('PE', {}).keys())[:5]
            
            if ce_strikes:
                print(f"  Call Strikes: {', '.join(ce_strikes)}")
            
            if pe_strikes:
                print(f"  Put Strikes: {', '.join(pe_strikes)}")
            
            # Test nearest strikes
            current_price = float(ce_strikes[0]) if ce_strikes else 0
            if current_price > 0:
                nearest = lookup.get_nearest_strikes(underlying, expiry, current_price, 3)
                print(f"  Nearest to {current_price}:")
                print(f"    Above: {nearest['above']}")
                print(f"    Below: {nearest['below']}")
                
                # Test ATM strike
                atm = lookup.get_atm_strike(underlying, expiry, current_price)
                if atm:
                    print(f"  ATM Strike: {atm}")
                    
                    # Test option contract lookup
                    ce_token = lookup.get_option_contract(underlying, expiry, atm, 'CE')
                    pe_token = lookup.get_option_contract(underlying, expiry, atm, 'PE')
                    
                    if ce_token:
                        details = lookup.get_token_details(ce_token)
                        print(f"  ATM Call: {ce_token}" + (f" ({details['symbol']})" if details else ""))
                    
                    if pe_token:
                        details = lookup.get_token_details(pe_token)
                        print(f"  ATM Put: {pe_token}" + (f" ({details['symbol']})" if details else ""))
        else:
            print("  No option chains found")
    
    # Test stock search
    print("\n=== STOCK SEARCH ===")
    search_queries = ["REL", "BANK", "NIF", "TCS"]
    
    for query in search_queries:
        results = lookup.search_stocks(query)
        print(f"\nSearch results for '{query}':")
        
        if results:
            for i, result in enumerate(results[:5]):
                symbol = result.get('symbol', 'Unknown')
                name = result.get('name', 'Unknown')
                print(f"  {i+1}. {symbol}: {name}")
                
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more results")
        else:
            print("  No results found")

def main():
    """Main function"""
    # Check if mappings directory exists
    if not Path('data/mappings').exists():
        print("Mappings directory not found. Please run generate_stock_mappings.py first.")
        return 1
    
    # Run tests
    test_stock_lookup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())