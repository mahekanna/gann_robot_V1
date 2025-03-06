#!/usr/bin/env python3
"""
Generate the necessary files for Gann Square of 9 testing
"""

import os

def ensure_directories():
    """Create required directories if they don't exist"""
    dirs = ['core', 'core/gann', 'tests', 'tools']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {directory.replace('/', '.')} package\n")

def create_original_gann():
    """Create the original Gann implementation file"""
    with open("original_gann.py", "w", encoding="utf-8") as f:
        f.write("""
import math
import json

# -------------------------------
# Gann Square of 9 Functions
# -------------------------------

def gann_square_of_9(price, increments, num_values=20, include_lower=True):
    \"\"\"Generates Gann Square of 9 levels for different angles.\"\"\"
    gann_values = {}
    angles = ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']

    root = math.sqrt(price)
    base = math.floor(root)
    central_value = base * base

    for angle, increment in zip(angles, increments):
        gann_values[angle] = []
        is_cardinal = angle.replace('deg', '').isdigit() and int(angle.replace('deg', '')) % 90 == 0
        base_mult = 1.0 if is_cardinal else 1.125

        if include_lower:
            lower_count = num_values // 2
            for i in range(lower_count, 0, -1):
                if is_cardinal:
                    val = base - (i * increment)
                    if val > 0:
                        squared = val * val
                        gann_values[angle].insert(0, round(squared, 2))
                else:
                    val = base - (i * increment * base_mult)
                    if val > 0:
                        squared = val * val
                        gann_values[angle].insert(0, round(squared, 2))

        gann_values[angle].append(round(central_value, 2))

        for i in range(1, num_values + 1):
            if is_cardinal:
                val = base + (i * increment)
                squared = val * val
            else:
                val = base + (i * increment * base_mult)
                squared = val * val
            gann_values[angle].append(round(squared, 2))

    return gann_values

def find_buy_sell_levels(price, gann_values):
    \"\"\"Finds the nearest Buy and Sell levels from the 0deg angle.\"\"\"
    buy_above = None
    sell_below = None
    closest_above = None
    closest_below = None

    if '0deg' in gann_values:
        for value in gann_values['0deg']:
            if value > price and (closest_above is None or value < closest_above):
                closest_above = value
                buy_above = ('0deg', value)
            if value < price and (closest_below is None or value > closest_below):
                closest_below = value
                sell_below = ('0deg', value)

    return buy_above, sell_below

def get_unique_targets_from_angles(entry_value, gann_values, num_levels, current_price, sell_below_value=None):
    \"\"\"Fetch unique buy and sell targets, ensuring unique buy targets per angle.\"\"\"
    angles = ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']
    buy_targets = []
    sell_targets = []
    used_values_buy = set()
    used_values_sell = set()

    # Buy targets: Ensure unique values, one per angle
    for angle in angles:
        values_above = [v for v in gann_values[angle] if v > entry_value and v not in used_values_buy]
        if values_above:
            closest_above = min(values_above)
            buy_targets.append((angle, closest_above))
            used_values_buy.add(closest_above)

    # Sell targets: Start with central value, then unique below sell_below_value
    central_value = math.floor(math.sqrt(current_price)) ** 2
    if sell_below_value is not None and central_value < sell_below_value:
        sell_targets.append(('0deg', central_value))
        used_values_sell.add(central_value)

    for angle in angles:
        if sell_below_value is not None:
            values_below = [v for v in gann_values[angle] if v < sell_below_value and v not in used_values_sell]
            if values_below:
                highest_below = max(values_below)
                sell_targets.append((angle, highest_below))
                used_values_sell.add(highest_below)

    # Sort and limit to num_levels
    buy_targets = sorted(buy_targets, key=lambda x: x[1])[:num_levels]
    sell_targets = sorted(sell_targets, key=lambda x: x[1], reverse=True)[:num_levels]

    return buy_targets, sell_targets

def calculate_stoploss(buy_above, sell_below, buffer_percentage):
    \"\"\"Calculate stoploss for long and short trades.\"\"\"
    long_stoploss = round(sell_below[1] * (1 - buffer_percentage), 2) if sell_below else None
    short_stoploss = round(buy_above[1] * (1 + buffer_percentage), 2) if buy_above else None
    return long_stoploss, short_stoploss
""")

def create_new_gann_implementation():
    """Create our new Gann implementation file"""
    with open("core/gann/square_of_9.py", "w", encoding="utf-8") as f:
        f.write("""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class GannSquareOf9:
    \"\"\"
    Implementation of W.D. Gann's Square of 9 calculation for trading levels.
    This class provides methods to calculate support/resistance levels, buy/sell 
    signals, and targets based on price action.
    \"\"\"

    def __init__(self, buffer_percentage: float = 0.002):
        \"\"\"
        Initialize the Gann Square of 9 calculator.
        
        Args:
            buffer_percentage: Buffer percentage for stoploss calculation to reduce whipsaws
        \"\"\"
        self.buffer_percentage = buffer_percentage
        # Default increments for different angles
        # Cardinal (0deg, 90deg, 180deg, 270deg) and Ordinal (45deg, 135deg, 225deg, 315deg)
        self.default_increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
        self.angles = ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']

    def calculate_levels(self, price: float, increments: Optional[List[float]] = None, 
                         num_values: int = 20, include_lower: bool = True) -> Dict[str, List[float]]:
        \"\"\"
        Generates Gann Square of 9 levels for different angles.
        
        Args:
            price: The price to calculate levels from (usually previous candle close)
            increments: Custom increments for each angle, if None uses default
            num_values: Number of values to generate above the central value
            include_lower: Whether to include values below the central value
            
        Returns:
            Dictionary with angles as keys and lists of price levels as values
        \"\"\"
        if increments is None:
            increments = self.default_increments
            
        gann_values = {}
        
        root = math.sqrt(price)
        base = math.floor(root)
        central_value = base * base

        for angle, increment in zip(self.angles, increments):
            gann_values[angle] = []
            is_cardinal = angle.replace('deg', '').isdigit() and int(angle.replace('deg', '')) % 90 == 0
            base_mult = 1.0 if is_cardinal else 1.125

            if include_lower:
                lower_count = num_values // 2
                for i in range(lower_count, 0, -1):
                    if is_cardinal:
                        val = base - (i * increment)
                        if val > 0:
                            squared = val * val
                            gann_values[angle].insert(0, round(squared, 2))
                    else:
                        val = base - (i * increment * base_mult)
                        if val > 0:
                            squared = val * val
                            gann_values[angle].insert(0, round(squared, 2))

            gann_values[angle].append(round(central_value, 2))

            for i in range(1, num_values + 1):
                if is_cardinal:
                    val = base + (i * increment)
                    squared = val * val
                else:
                    val = base + (i * increment * base_mult)
                    squared = val * val
                gann_values[angle].append(round(squared, 2))

        return gann_values

    def find_key_levels(self, price: float, gann_values: Optional[Dict[str, List[float]]] = None) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
        \"\"\"
        Finds the nearest Buy and Sell levels from the 0deg angle.
        
        Args:
            price: Current price to find levels around
            gann_values: Pre-calculated Gann values, if None will calculate
            
        Returns:
            Tuple of (buy_above, sell_below) where each is (angle, price) or None
        \"\"\"
        if gann_values is None:
            gann_values = self.calculate_levels(price)
            
        buy_above = None
        sell_below = None
        closest_above = None
        closest_below = None

        if '0deg' in gann_values:
            for value in gann_values['0deg']:
                if value > price and (closest_above is None or value < closest_above):
                    closest_above = value
                    buy_above = ('0deg', value)
                if value < price and (closest_below is None or value > closest_below):
                    closest_below = value
                    sell_below = ('0deg', value)

        return buy_above, sell_below

    def get_targets(self, entry_value: float, gann_values: Dict[str, List[float]], 
                   num_levels: int, current_price: float, 
                   sell_below_value: Optional[float] = None) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        \"\"\"
        Fetch unique buy and sell targets, ensuring unique buy targets per angle.
        
        Args:
            entry_value: Entry price point (usually the buy_above value)
            gann_values: Pre-calculated Gann values
            num_levels: Number of target levels to return
            current_price: Current market price
            sell_below_value: Value for sell_below level from find_key_levels
            
        Returns:
            Tuple of (buy_targets, sell_targets) where each is a list of (angle, price)
        \"\"\"
        buy_targets = []
        sell_targets = []
        used_values_buy = set()
        used_values_sell = set()

        # Buy targets: Ensure unique values, one per angle
        for angle in self.angles:
            values_above = [v for v in gann_values[angle] if v > entry_value and v not in used_values_buy]
            if values_above:
                closest_above = min(values_above)
                buy_targets.append((angle, closest_above))
                used_values_buy.add(closest_above)

        # Sell targets: Start with central value, then unique below sell_below_value
        if sell_below_value:
            central_value = math.floor(math.sqrt(current_price)) ** 2
            if central_value < sell_below_value:
                sell_targets.append(('0deg', central_value))
                used_values_sell.add(central_value)

            for angle in self.angles:
                values_below = [v for v in gann_values[angle] if v < sell_below_value and v not in used_values_sell]
                if values_below:
                    highest_below = max(values_below)
                    sell_targets.append((angle, highest_below))
                    used_values_sell.add(highest_below)

        # Sort and limit to num_levels
        buy_targets = sorted(buy_targets, key=lambda x: x[1])[:num_levels]
        sell_targets = sorted(sell_targets, key=lambda x: x[1], reverse=True)[:num_levels]

        return buy_targets, sell_targets

    def calculate_stoploss(self, buy_above: Optional[Tuple[str, float]], 
                         sell_below: Optional[Tuple[str, float]]) -> Tuple[Optional[float], Optional[float]]:
        \"\"\"
        Calculate stoploss for long and short trades.
        
        Args:
            buy_above: Buy above level as (angle, price)
            sell_below: Sell below level as (angle, price)
            
        Returns:
            Tuple of (long_stoploss, short_stoploss)
        \"\"\"
        long_stoploss = round(sell_below[1] * (1 - self.buffer_percentage), 2) if sell_below else None
        short_stoploss = round(buy_above[1] * (1 + self.buffer_percentage), 2) if buy_above else None
        return long_stoploss, short_stoploss

    def generate_signals(self, current_price: float, previous_close: float, 
                        num_target_levels: int = 3) -> Dict[str, Union[float, str, List[Tuple[str, float]]]]:
        \"\"\"
        Generate trading signals based on Gann Square of 9.
        
        Args:
            current_price: Current market price
            previous_close: Previous candle's closing price
            num_target_levels: Number of target levels to generate
            
        Returns:
            Dictionary with signal information including targets and stoploss
        \"\"\"
        # Calculate Gann levels based on previous close
        gann_values = self.calculate_levels(previous_close)
        
        # Find key levels
        buy_above, sell_below = self.find_key_levels(current_price, gann_values)
        
        # Default values if no signals are found
        result = {
            "current_price": current_price,
            "previous_close": previous_close,
            "buy_above": None,
            "sell_below": None,
            "long_signal": False,
            "short_signal": False,
            "buy_targets": [],
            "sell_targets": [],
            "long_stoploss": None,
            "short_stoploss": None
        }
        
        if not buy_above or not sell_below:
            return result
            
        # Check for signals
        long_signal = current_price > buy_above[1]
        short_signal = current_price < sell_below[1]
        
        # Get targets if we have key levels
        buy_targets, sell_targets = self.get_targets(
            buy_above[1], gann_values, num_target_levels, current_price, sell_below[1]
        )
        
        # Calculate stop loss levels
        long_stoploss, short_stoploss = self.calculate_stoploss(buy_above, sell_below)
        
        # Update result with calculated values
        result.update({
            "buy_above": buy_above[1],
            "sell_below": sell_below[1],
            "long_signal": long_signal,
            "short_signal": short_signal,
            "buy_targets": buy_targets,
            "sell_targets": sell_targets,
            "long_stoploss": long_stoploss,
            "short_stoploss": short_stoploss
        })
        
        return result

    def format_table(self, gann_values: Dict[str, List[float]]) -> str:
        \"\"\"
        Creates a formatted table showing both cardinal and ordinal angle values.
        
        Args:
            gann_values: Dictionary of Gann values by angle
            
        Returns:
            Formatted table as string
        \"\"\"
        table = ""
        col_widths = [max(len(str(value)) for value in values_list) for values_list in gann_values.values()]

        table += "Angle | " + " | ".join(f"{angle}".center(width) for angle, width in zip(gann_values.keys(), col_widths)) + "\\n"
        table += "-" * len(table.split("\\n")[0]) + "\\n"

        num_values = len(next(iter(gann_values.values())))
        for i in range(num_values):
            row = f"{i+1:4d} | "
            for angle, values_list in gann_values.items():
                value = values_list[i]
                row += f"{str(value).rjust(col_widths[list(gann_values.keys()).index(angle)])} | "
            table += row + "\\n"

        return table
""")

def create_test_script():
    """Create the test script"""
    with open("tests/test_gann_implementation.py", "w", encoding="utf-8") as f:
        f.write("""
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
    \"\"\"
    Test class to verify that our Gann Square of 9 implementation
    produces the same results as the original code.
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the test class with both implementations\"\"\"
        self.gann = GannSquareOf9(buffer_percentage=0.002)
        self.test_prices = [
            22510.85, 22127.4, 1482.75, 500.0, 2500.0, 1000.0, 999.99, 1.23, 9999.99
        ]
        
    def test_gann_values(self):
        \"\"\"Test that gann_square_of_9 produces the same values\"\"\"
        print("\\n=== Testing Gann Square of 9 Values ===")
        
        for price in self.test_prices:
            print(f"\\nTesting price: {price}")
            
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
        \"\"\"Test that find_buy_sell_levels produces the same values\"\"\"
        print("\\n=== Testing Find Buy/Sell Levels ===")
        
        for price in self.test_prices:
            print(f"\\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0deg': original_values['0deg']})
            
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
        \"\"\"Test that get_unique_targets_from_angles produces the same targets\"\"\"
        print("\\n=== Testing Unique Targets ===")
        
        for price in self.test_prices:
            print(f"\\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0deg': original_values['0deg']})
            
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
        \"\"\"Test that calculate_stoploss produces the same values\"\"\"
        print("\\n=== Testing Stoploss Calculation ===")
        
        for price in self.test_prices:
            print(f"\\nTesting price: {price}")
            
            # Run original implementation
            increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            original_values = original_gann_square_of_9(price, increments)
            original_buy_above, original_sell_below = original_find_buy_sell_levels(price, {'0deg': original_values['0deg']})
            
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
        \"\"\"Test the complete signal generation process\"\"\"
        print("\\n=== Testing Complete Signal Generation ===")
        
        for price in self.test_prices:
            print(f"\\nTesting price: {price}")
            
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
        \"\"\"Run all tests\"\"\"
        self.test_gann_values()
        self.test_find_key_levels()
        self.test_targets()
        self.test_stoploss()
        self.test_generate_signals()
        
        print("\\n=== Tests completed ===")


if __name__ == "__main__":
    test = TestGannImplementation()
    test.run_tests()
""")

def main():
    """Main function"""
    print("Generating Gann Square of 9 test files...")
    
    # Create directories
    ensure_directories()
    
    # Create files
    create_original_gann()
    create_new_gann_implementation()
    create_test_script()
    
    print("Files created successfully!")
    print("""
To run tests:
1. Change to the project directory
2. Run: python tests/test_gann_implementation.py
""")

if __name__ == "__main__":
    main()