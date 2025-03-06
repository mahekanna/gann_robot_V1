"""
Gann Square of 9 implementation
Core calculations for the Gann-based trading strategy
"""

import math
import json
from typing import Dict, List, Tuple, Optional, Union, Any

class GannSquareOf9:
    """Implementation of W.D. Gann's Square of 9 calculation for trading levels."""
    
    def __init__(self, buffer_percentage: float = 0.002):
        """
        Initialize the Gann Square of 9 calculator.
        
        Args:
            buffer_percentage: Buffer percentage for stoploss calculation
        """
        self.buffer_percentage = buffer_percentage
        # Default increments for different angles
        self.default_increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
        self.angles = ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']

    def calculate_levels(self, price: float, increments: Optional[List[float]] = None, 
                         num_values: int = 20, include_lower: bool = True) -> Dict[str, List[float]]:
        """
        Generates Gann Square of 9 levels for different angles.
        
        Args:
            price: The price to calculate levels from (usually previous candle close)
            increments: Custom increments for each angle, if None uses default
            num_values: Number of values to generate above the central value
            include_lower: Whether to include values below the central value
            
        Returns:
            Dictionary with angles as keys and lists of price levels as values
        """
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
        """
        Finds the nearest Buy and Sell levels from the 0deg angle.
        
        Args:
            price: Current price to find levels around
            gann_values: Pre-calculated Gann values, if None will calculate
            
        Returns:
            Tuple of (buy_above, sell_below) where each is (angle, price) or None
        """
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
        """
        Fetch unique buy and sell targets, ensuring unique buy targets per angle.
        
        Args:
            entry_value: Entry price point (usually the buy_above value)
            gann_values: Pre-calculated Gann values
            num_levels: Number of target levels to return
            current_price: Current market price
            sell_below_value: Value for sell_below level from find_key_levels
            
        Returns:
            Tuple of (buy_targets, sell_targets) where each is a list of (angle, price)
        """
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
        central_value = math.floor(math.sqrt(current_price)) ** 2
        if sell_below_value is not None and central_value < sell_below_value:
            sell_targets.append(('0deg', central_value))
            used_values_sell.add(central_value)

        for angle in self.angles:
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

    def calculate_stoploss(self, buy_above: Optional[Tuple[str, float]], 
                          sell_below: Optional[Tuple[str, float]]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stoploss for long and short trades.
        
        Args:
            buy_above: Buy above level as (angle, price)
            sell_below: Sell below level as (angle, price)
            
        Returns:
            Tuple of (long_stoploss, short_stoploss)
        """
        long_stoploss = round(sell_below[1] * (1 - self.buffer_percentage), 2) if sell_below else None
        short_stoploss = round(buy_above[1] * (1 + self.buffer_percentage), 2) if buy_above else None
        return long_stoploss, short_stoploss

    def generate_signals(self, price: float, previous_close: float, 
                         num_target_levels: int = 3) -> Dict[str, Any]:
        """
        Generate trading signals based on Gann Square of 9.
        
        Args:
            price: Current market price
            previous_close: Previous candle's closing price
            num_target_levels: Number of target levels to generate
            
        Returns:
            Dictionary with signal information including targets and stoploss
        """
        # Calculate Gann levels based on previous close
        gann_values = self.calculate_levels(previous_close)
        
        # Find key levels
        buy_above, sell_below = self.find_key_levels(price, gann_values)
        
        # Default values if no signals are found
        result = {
            "current_price": price,
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
        long_signal = price > buy_above[1]
        short_signal = price < sell_below[1]
        
        # Get targets if we have key levels
        buy_targets, sell_targets = self.get_targets(
            buy_above[1], gann_values, num_target_levels, price, sell_below[1]
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
        """
        Creates a formatted table showing both cardinal and ordinal angle values.
        
        Args:
            gann_values: Dictionary of Gann values by angle
            
        Returns:
            Formatted table as string
        """
        table = ""
        col_widths = [max(len(str(value)) for value in values_list) for values_list in gann_values.values()]

        table += "Angle | " + " | ".join(f"{angle}".center(width) for angle, width in zip(gann_values.keys(), col_widths)) + "\n"
        table += "-" * len(table.split("\n")[0]) + "\n"

        num_values = len(next(iter(gann_values.values())))
        for i in range(num_values):
            row = f"{i+1:4d} | "
            for angle, values_list in gann_values.items():
                value = values_list[i]
                row += f"{str(value).rjust(col_widths[list(gann_values.keys()).index(angle)])} | "
            table += row + "\n"

        return table

    def get_analysis_report(self, price: float, num_levels: int = 3) -> Dict[str, Any]:
        """
        Run full analysis and produce a report for a given price
        
        Args:
            price: Price to analyze (previous candle close)
            num_levels: Number of target levels
            
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
        gann_values = self.calculate_levels(price, increments)

        buy_level_0, sell_level_0 = self.find_key_levels(price, {'0deg': gann_values['0deg']})

        if buy_level_0 and sell_level_0:
            buy_targets, sell_targets = self.get_targets(
                buy_level_0[1],
                gann_values,
                num_levels,
                price,
                sell_level_0[1]
            )

            buy_targets_str = "\n".join([f"{angle}: {value}" for angle, value in buy_targets])
            sell_targets_str = "\n".join([f"{angle}: {value}" for angle, value in sell_targets])

            long_stoploss, short_stoploss = self.calculate_stoploss(buy_level_0, sell_level_0)

            return {
                "price": price,
                "gann_values": gann_values,
                "gann_table": self.format_table(gann_values),
                "buy_above": buy_level_0[1],
                "sell_below": sell_level_0[1],
                "buy_targets": buy_targets,
                "sell_targets": sell_targets,
                "buy_targets_str": buy_targets_str,
                "sell_targets_str": sell_targets_str,
                "long_stoploss": long_stoploss,
                "short_stoploss": short_stoploss
            }
        else:
            return None