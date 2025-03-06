
import math
import json

# -------------------------------
# Gann Square of 9 Functions
# -------------------------------

def gann_square_of_9(price, increments, num_values=20, include_lower=True):
    """Generates Gann Square of 9 levels for different angles."""
    gann_values = {}
    angles = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']

    root = math.sqrt(price)
    base = math.floor(root)
    central_value = base * base

    for angle, increment in zip(angles, increments):
        gann_values[angle] = []
        is_cardinal = angle.replace('°', '').isdigit() and int(angle.replace('°', '')) % 90 == 0
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
    """Finds the nearest Buy and Sell levels from the 0° angle."""
    buy_above = None
    sell_below = None
    closest_above = None
    closest_below = None

    if '0°' in gann_values:
        for value in gann_values['0°']:
            if value > price and (closest_above is None or value < closest_above):
                closest_above = value
                buy_above = ('0°', value)
            if value < price and (closest_below is None or value > closest_below):
                closest_below = value
                sell_below = ('0°', value)

    return buy_above, sell_below

def get_unique_targets_from_angles(entry_value, gann_values, num_levels, current_price, sell_below_value=None):
    """Fetch unique buy and sell targets, ensuring unique buy targets per angle."""
    angles = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
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
    if central_value < sell_below_value:
        sell_targets.append(('0°', central_value))
        used_values_sell.add(central_value)

    for angle in angles:
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
