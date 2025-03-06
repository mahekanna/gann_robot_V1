# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 21:06:59 2025

@author: mahes
"""

# File: test_api.py - Updated
"""
Simple test script for ICICI Direct API
"""

import logging
import sys
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import directly from api module
from api.auth import get_client

def test_api():
    """Test API authentication and basic functionality"""
    print("\n=== ICICI Direct API Test ===\n")
    
    # Get authenticated client
    client = get_client()
    
    if client:
        print("✓ Successfully authenticated!")
        
        # Test a simple API call
        try:
            print("\nFetching customer details...")
            details = client.get_customer_details()
            print(f"✓ API call successful!")
            print(f"Status: {details.get('Status', 'Unknown')}")
            
            # Try getting some market data
            print("\nFetching quote for NIFTY...")
            # The correct parameters might be different than what we tried
            # Let's try without the indices parameter
            quote = client.get_quotes(
                stock_code="NIFTY",
                exchange_code="NSE"
            )
            
            if quote and "Success" in quote and quote["Success"]:
                print("✓ NIFTY quote retrieved successfully!")
                # Show some key data
                data = quote["Success"][0]
                print(f"Last Price: {data.get('last_price', 'N/A')}")
                print(f"Change: {data.get('change', 'N/A')}")
                print(f"% Change: {data.get('percentage_change', 'N/A')}%")
            else:
                print("× Failed to get NIFTY quote")
                print(f"Response: {quote}")
            
        except Exception as e:
            print(f"× API call failed: {e}")
    else:
        print("× Authentication failed")

if __name__ == "__main__":
    test_api()