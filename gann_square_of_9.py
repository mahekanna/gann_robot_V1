#!/usr/bin/env python3
"""
Gann Square of 9 implementation based on the original code
"""

import math
import json
from typing import Dict, List, Tuple, Optional

def gann_square_of_9(price, increments, num_values=20, include_lower=True):
    """Generates Gann Square of 9 levels for different angles."""
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
    """Finds the nearest Buy and Sell levels from the 0deg angle."""
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
    """Fetch unique buy and sell targets, ensuring unique buy targets per angle."""
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
    """Calculate stoploss for long and short trades."""
    long_stoploss = round(sell_below[1] * (1 - buffer_percentage), 2) if sell_below else None
    short_stoploss = round(buy_above[1] * (1 + buffer_percentage), 2) if buy_above else None
    return long_stoploss, short_stoploss

def gann_square_of_9_table(gann_values):
    """Creates a formatted table showing both cardinal and ordinal angle values."""
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

def print_analysis_with_table(price, num_levels=3, buffer_percentage=0.002):
    """Runs the Gann Square of 9 Analysis with a configurable number of target levels."""
    increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
    gann_values = gann_square_of_9(price, increments)
    gann_table_str = gann_square_of_9_table(gann_values)

    buy_level_0, sell_level_0 = find_buy_sell_levels(price, {'0deg': gann_values['0deg']})

    if buy_level_0 and sell_level_0:
        buy_targets, sell_targets = get_unique_targets_from_angles(
            buy_level_0[1],
            gann_values,
            num_levels,
            price,
            sell_level_0[1]
        )

        buy_targets_str = "\n".join([f"{angle}: {value}" for angle, value in buy_targets])
        sell_targets_str = "\n".join([f"{angle}: {value}" for angle, value in sell_targets])

        results = {
            "Current Price": price,
            "Buy above (0deg Angle Only)": buy_level_0[1],
            "Sell below (0deg Angle Only)": sell_level_0[1],
            "Buy Targets": buy_targets_str if buy_targets else "No Buy Targets",
            "Sell Targets": sell_targets_str if sell_targets else "No Sell Targets",
            "Stoploss Long": calculate_stoploss(buy_level_0, sell_level_0, buffer_percentage)[0],
            "Stoploss Short": calculate_stoploss(buy_level_0, sell_level_0, buffer_percentage)[1],
        }

        print("\nGann Square of 9 Table:")
        print(gann_table_str)
        print("\nGann Analysis Results:\n")
        for key, value in results.items():
            if isinstance(value, str) and "\n" in value:
                print(f"{key}:\n{value}\n")
            else:
                print(f"{key}: {value}")

        return results
    else:
        print(f"Could not find buy/sell levels for price {price}")
        return None