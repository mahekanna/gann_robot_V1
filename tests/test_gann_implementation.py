import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

# Add parent directory to path to import both implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original implementation
from original_gann import (
    gann_square_of_9 as original_gann_square_of_9,
    find_buy_sell_levels as original_find_buy_sell_levels,
    get_unique_targets_from_angles as original_get_unique_targets_from_angles,
    calculate_stoploss as original_calculate_stoploss
)

# Import our new implementation
from core.gann.square_of_9 import GannSquareOf9

class TestGannImplementation:
    """
    Test class to verify that our Gann Square of 9 implementation
    produces the same results as the original code.
    """
    
    def __init__(self):
        """Initialize the test class with both implementations"""
        self.gann = GannSquareOf9(buffer_percentage=0.002)
        self.test_prices = [
            22510.85, 22127.4, 1482.75, 500.0, 2500.0, 1000.0, 999.99, 1.23, 9999.99
        ]
        
    def test_gann_values(self):
        """Test that gann_square_of_9 produces the same values"""
        print("\n=== Testing Gann Square of 9 Values ===")
        
        for price in self.test_prices:
            print(f"\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            
            # Run our implementation
            our_values = self.gann.calculate_levels(price, increments)
            
            # Check if all angles are present in both
            angles_match = set(original_values.keys()) == set(our_values.keys())
            print(f"Angles match: {angles_match}")
            
            # Check if values match for each angle
            all_values_match = True
            for angle in original_values.keys():
                original_angle_values = original_values[angle]
                our_angle_values = our_values[angle]
                
                # Check length
                length_match = len(original_angle_values) == len(our_angle_values)
                if not length_match:
                    print(f"  {angle}: Length mismatch. Original: {len(original_angle_values)}, Ours: {len(our_angle_values)}")
                    all_values_match = False
                    continue
                
                # Check values
                values_match = all(
                    abs(original_angle_values[i] - our_angle_values[i]) < 0.01  # Allow small floating point differences
                    for i in range(len(original_angle_values))
                )
                
                if not values_match:
                    print(f"  {angle}: Values mismatch.")
                    print(f"    Original: {original_angle_values[:5]}...")
                    print(f"    Ours    : {our_angle_values[:5]}...")
                    all_values_match = False
            
            print(f"All values match: {all_values_match}")
    
    def test_find_key_levels(self):
        """Test that find_buy_sell_levels produces the same values"""
        print("\n=== Testing Find Buy/Sell Levels ===")
        
        for price in self.test_prices:
            print(f"\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0�': original_values['0�']})
            
            # Run our implementation
            our_values = self.gann.calculate_levels(price, increments)
            our_buy_above, our_sell_below = self.gann.find_key_levels(price, our_values)
            
            # Compare results
            buy_above_match = (
                original_buy_above is None and our_buy_above is None
            ) or (
                original_buy_above is not None and our_buy_above is not None and
                original_buy_above[0] == our_buy_above[0] and  # Check angle
                abs(original_buy_above[1] - our_buy_above[1]) < 0.01  # Check price with tolerance
            )
            
            sell_below_match = (
                original_sell_below is None and our_sell_below is None
            ) or (
                original_sell_below is not None and our_sell_below is not None and
                original_sell_below[0] == our_sell_below[0] and  # Check angle
                abs(original_sell_below[1] - our_sell_below[1]) < 0.01  # Check price with tolerance
            )
            
            print(f"Buy Above match: {buy_above_match}")
            if original_buy_above and our_buy_above:
                print(f"  Original: {original_buy_above[0]}, {original_buy_above[1]}")
                print(f"  Ours    : {our_buy_above[0]}, {our_buy_above[1]}")
                
            print(f"Sell Below match: {sell_below_match}")
            if original_sell_below and our_sell_below:
                print(f"  Original: {original_sell_below[0]}, {original_sell_below[1]}")
                print(f"  Ours    : {our_sell_below[0]}, {our_sell_below[1]}")
    
    def test_targets(self):
        """Test that get_unique_targets_from_angles produces the same targets"""
        print("\n=== Testing Unique Targets ===")
        
        for price in self.test_prices:
            print(f"\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0�': original_values['0�']})
            
            if not original_buy_above or not original_sell_below:
                print(f"Skipping price {price} - no buy/sell levels found")
                continue
                
            original_buy_targets, original_sell_targets = original_get_unique_targets_from_angles(
                original_buy_above[1], original_values, 3, price, original_sell_below[1]
            )
            
            # Run our implementation
            our_values = self.gann.calculate_levels(price, increments)
            our_buy_above, our_sell_below = self.gann.find_key_levels(price, our_values)
            
            if not our_buy_above or not our_sell_below:
                print(f"Skipping price {price} - no buy/sell levels found in our implementation")
                continue
                
            our_buy_targets, our_sell_targets = self.gann.get_targets(
                our_buy_above[1], our_values, 3, price, our_sell_below[1]
            )
            
            # Compare results
            buy_targets_match = len(original_buy_targets) == len(our_buy_targets)
            sell_targets_match = len(original_sell_targets) == len(our_sell_targets)
            
            if buy_targets_match:
                for i in range(len(original_buy_targets)):
                    if (original_buy_targets[i][0] != our_buy_targets[i][0] or 
                        abs(original_buy_targets[i][1] - our_buy_targets[i][1]) >= 0.01):
                        buy_targets_match = False
                        break
            
            if sell_targets_match:
                for i in range(len(original_sell_targets)):
                    if (original_sell_targets[i][0] != our_sell_targets[i][0] or 
                        abs(original_sell_targets[i][1] - our_sell_targets[i][1]) >= 0.01):
                        sell_targets_match = False
                        break
            
            print(f"Buy Targets match: {buy_targets_match}")
            if not buy_targets_match:
                print(f"  Original Buy Targets: {original_buy_targets}")
                print(f"  Our Buy Targets    : {our_buy_targets}")
                
            print(f"Sell Targets match: {sell_targets_match}")
            if not sell_targets_match:
                print(f"  Original Sell Targets: {original_sell_targets}")
                print(f"  Our Sell Targets    : {our_sell_targets}")
    
    def test_stoploss(self):
        """Test that calculate_stoploss produces the same values"""
        print("\n=== Testing Stoploss Calculation ===")
        
        for price in self.test_prices:
            print(f"\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0�': original_values['0�']})
            
            if not original_buy_above or not original_sell_below:
                print(f"Skipping price {price} - no buy/sell levels found")
                continue
                
            original_long_sl, original_short_sl = original_calculate_stoploss(
                original_buy_above, original_sell_below, 0.002
            )
            
            # Run our implementation
            our_values = self.gann.calculate_levels(price, increments)
            our_buy_above, our_sell_below = self.gann.find_key_levels(price, our_values)
            
            if not our_buy_above or not our_sell_below:
                print(f"Skipping price {price} - no buy/sell levels found in our implementation")
                continue
                
            our_long_sl, our_short_sl = self.gann.calculate_stoploss(our_buy_above, our_sell_below)
            
            # Compare results
            long_sl_match = (
                original_long_sl is None and our_long_sl is None
            ) or (
                original_long_sl is not None and our_long_sl is not None and
                abs(original_long_sl - our_long_sl) < 0.01  # Check with tolerance
            )
            
            short_sl_match = (
                original_short_sl is None and our_short_sl is None
            ) or (
                original_short_sl is not None and our_short_sl is not None and
                abs(original_short_sl - our_short_sl) < 0.01  # Check with tolerance
            )
            
            print(f"Long Stoploss match: {long_sl_match}")
            if original_long_sl and our_long_sl:
                print(f"  Original: {original_long_sl}")
                print(f"  Ours    : {our_long_sl}")
                
            print(f"Short Stoploss match: {short_sl_match}")
            if original_short_sl and our_short_sl:
                print(f"  Original: {original_short_sl}")
                print(f"  Ours    : {our_short_sl}")
    
    def test_generate_signals(self):
        """Test the complete signal generation process"""
        print("\n=== Testing Complete Signal Generation ===")
        
        for price in self.test_prices:
            print(f"\nTesting price: {price}")
            
            # Use current price as current and price-10 as previous close for testing
            current_price = price
            previous_close = price - 10 if price > 20 else price * 0.9
            
            # Generate signals with our implementation
            signals = self.gann.generate_signals(current_price, previous_close)
            
            # Verify signal structure
            expected_keys = [
                "current_price", "previous_close", "buy_above", "sell_below",
                "long_signal", "short_signal", "buy_targets", "sell_targets",
                "long_stoploss", "short_stoploss"
            ]
            
            missing_keys = [key for key in expected_keys if key not in signals]
            if missing_keys:
                print(f"Missing keys in signal output: {missing_keys}")
            else:
                print("Signal structure is correct")
                
            # Print out the signal for inspection
            print(f"Signal output:")
            print(f"  Current Price: {signals['current_price']}")
            print(f"  Previous Close: {signals['previous_close']}")
            print(f"  Buy Above: {signals['buy_above']}")
            print(f"  Sell Below: {signals['sell_below']}")
            print(f"  Long Signal: {signals['long_signal']}")
            print(f"  Short Signal: {signals['short_signal']}")
            print(f"  Long Stoploss: {signals['long_stoploss']}")
            print(f"  Short Stoploss: {signals['short_stoploss']}")
            print(f"  Buy Targets: {signals['buy_targets'][:3] if signals['buy_targets'] else []}")
            print(f"  Sell Targets: {signals['sell_targets'][:3] if signals['sell_targets'] else []}")
    
    def run_tests(self):
        """Run all tests"""
        self.test_gann_values()
        self.test_find_key_levels()
        self.test_targets()
        self.test_stoploss()
        self.test_generate_signals()
        
        print("\n=== Tests completed ===")


if __name__ == "__main__":
    test = TestGannImplementation()
    test.run_tests()
