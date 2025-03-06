# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 21:33:57 2025

@author: mahes
"""

# File: test_session_reuse.py
"""
Test if session reuse works correctly
"""

import logging
import sys
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import directly from api module
from api.auth import get_client

def test_session_reuse():
    """Test if session is reused correctly"""
    print("\n=== Testing Session Reuse ===\n")
    
    print("First authentication attempt:")
    client1 = get_client()
    if client1:
        print("✓ First authentication successful")
    else:
        print("× First authentication failed")
        return
    
    print("\nSecond authentication attempt (should reuse session):")
    client2 = get_client()
    if client2:
        print("✓ Second authentication successful")
        
        # Verify it works with a simple API call
        try:
            details = client2.get_customer_details()
            print(f"✓ API call successful (Status: {details.get('Status', 'Unknown')})")
        except Exception as e:
            print(f"× API call failed: {e}")
    else:
        print("× Second authentication failed")

if __name__ == "__main__":
    test_session_reuse()