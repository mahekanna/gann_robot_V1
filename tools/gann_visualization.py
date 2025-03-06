import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math
from datetime import datetime, timedelta

# Add parent directory to path to import both implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our implementation
from core.gann.square_of_9 import GannSquareOf9

def generate_sample_data(symbol: str = "NIFTY", days: int = 30, start_price: float = 22000.0, volatility: float = 0.015) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing
    
    Args:
        symbol: Symbol name
        days: Number of days of data
        start_price: Starting price
        volatility: Daily volatility as percentage
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    today = datetime.now().date()
    dates = [(today - timedelta(days=days-i)) for i in range(days)]
    
    # Generate prices with random walk
    close_prices = [start_price]
    for i in range(1, days):
        # Random daily return with slight upward bias
        daily_return = np.random.normal(0.0005, volatility)
        close_prices.append(close_prices[-1] * (1 + daily_return))
    
    # Generate OHLCV data
    data = []
    for i, date in enumerate(dates):
        close = close_prices[i]
        # Generate random daily range
        daily_range = close * np.random.uniform(0.01, 0.03)
        high = close + daily_range/2
        low = close - daily_range/2
        # Open between previous close and current close
        if i == 0:
            open_price = close * 0.995
        else:
            prev_close = close_prices[i-1]
            open_price = prev_close + 0.3 * (close - prev_close)
        
        # Volume varies with price movement
        volume = int(np.random.uniform(100000, 500000) * (1 + abs(close - (open_price)) / close))
        
        data.append({
            'datetime': pd.Timestamp(date),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'symbol': symbol
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def visualize_gann_levels(data: pd.DataFrame, start_idx: int = None, num_bars: int = 20) -> None:
    """
    Visualize Gann Square of 9 levels and signals on candlestick chart
    
    Args:
        data: DataFrame with OHLCV data
        start_idx: Index to start visualization from
        num_bars: Number of bars to display
    """
    if start_idx is None:
        start_idx = max(0, len(data) - num_bars)
    
    end_idx = min(start_idx + num_bars, len(data))
    display_data = data.iloc[start_idx:end_idx].copy()
    
    # Initialize Gann calculator
    gann = GannSquareOf9()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot candlesticks
    dates = list(range(len(display_data)))
    
    # Plot each candle
    for i, (idx, row) in enumerate(display_data.iterrows()):
        # Candle body
        if row['close'] >= row['open']:
            # Bullish candle
            color = 'green'
            body_bottom = row['open']
            body_height = row['close'] - row['open']
        else:
            # Bearish candle
            color = 'red'
            body_bottom = row['close']
            body_height = row['open'] - row['close']
            
        # Draw body
        ax.add_patch(plt.Rectangle((dates[i] - 0.4, body_bottom), 0.8, body_height, fill=True, color=color))
        
        # Draw wick
        ax.plot([dates[i], dates[i]], [row['low'], row['high']], color='black', linewidth=1)
    
    # Calculate and plot Gann levels for each bar
    for i, (idx, row) in enumerate(display_data.iterrows()):
        if i < len(display_data) - 1:  # Not the last bar
            # Get next bar for current price
            next_bar = display_data.iloc[i+1]
            
            # Calculate Gann levels using previous bar's close
            gann_signals = gann.generate_signals(next_bar['close'], row['close'])
            
            # Plot buy_above and sell_below levels if they exist
            if gann_signals['buy_above']:
                ax.axhline(y=gann_signals['buy_above'], color='blue', linestyle='--', 
                          alpha=0.5, linewidth=1, xmin=i/len(display_data), xmax=(i+1)/len(display_data))
                
            if gann_signals['sell_below']:
                ax.axhline(y=gann_signals['sell_below'], color='red', linestyle='--', 
                          alpha=0.5, linewidth=1, xmin=i/len(display_data), xmax=(i+1)/len(display_data))
    
    # Add annotations for the last bar's Gann levels
    last_bar_idx = len(display_data) - 2  # Second to last bar for calculation
    if last_bar_idx >= 0:
        prev_close = display_data.iloc[last_bar_idx]['close']
        current_price = display_data.iloc[last_bar_idx + 1]['close']
        
        gann_signals = gann.generate_signals(current_price, prev_close)
        
        if gann_signals['buy_above']:
            ax.axhline(y=gann_signals['buy_above'], color='blue', linestyle='-', 
                      linewidth=1.5, xmin=last_bar_idx/len(display_data), xmax=1)
            ax.text(len(display_data) - 1, gann_signals['buy_above'], 
                   f"Buy Above: {gann_signals['buy_above']:.2f}", 
                   color='blue', verticalalignment='bottom')
            
        if gann_signals['sell_below']:
            ax.axhline(y=gann_signals['sell_below'], color='red', linestyle='-', 
                      linewidth=1.5, xmin=last_bar_idx/len(display_data), xmax=1)
            ax.text(len(display_data) - 1, gann_signals['sell_below'], 
                   f"Sell Below: {gann_signals['sell_below']:.2f}", 
                   color='red', verticalalignment='top')
            
        # Plot target levels for the last signal
        for i, (angle, price) in enumerate(gann_signals['buy_targets']):
            ax.axhline(y=price, color='green', linestyle=':', linewidth=1, xmin=0.9)
            ax.text(len(display_data) - 0.5, price, 
                   f"Target {i+1}: {price:.2f} ({angle})", 
                   color='green', verticalalignment='bottom')
                   
        for i, (angle, price) in enumerate(gann_signals['sell_targets']):
            ax.axhline(y=price, color='purple', linestyle=':', linewidth=1, xmin=0.9)
            ax.text(len(display_data) - 0.5, price, 
                   f"Target {i+1}: {price:.2f} ({angle})", 
                   color='purple', verticalalignment='top')
            
        # Plot stop loss levels
        if gann_signals['long_stoploss']:
            ax.axhline(y=gann_signals['long_stoploss'], color='red', linestyle='-.',
                      linewidth=1, xmin=0.9)
            ax.text(len(display_data) - 0.5, gann_signals['long_stoploss'], 
                   f"Long SL: {gann_signals['long_stoploss']:.2f}", 
                   color='red', verticalalignment='top')
                   
        if gann_signals['short_stoploss']:
            ax.axhline(y=gann_signals['short_stoploss'], color='blue', linestyle='-.',
                      linewidth=1, xmin=0.9)
            ax.text(len(display_data) - 0.5, gann_signals['short_stoploss'], 
                   f"Short SL: {gann_signals['short_stoploss']:.2f}", 
                   color='blue', verticalalignment='bottom')
    
    # Set x-axis labels to dates
    ax.set_xticks(dates)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in display_data['datetime']], rotation=45)
    
    # Set title and labels
    plt.title(f"Gann Square of 9 Levels for {display_data.iloc[0]['symbol']}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add signal information in a text box
    if last_bar_idx >= 0:
        signal_text = (
            f"Current Price: {current_price:.2f}\n"
            f"Previous Close: {prev_close:.2f}\n"
            f"Buy Above: {gann_signals['buy_above']:.2f}\n"
            f"Sell Below: {gann_signals['sell_below']:.2f}\n"
            f"Long Signal: {gann_signals['long_signal']}\n"
            f"Short Signal: {gann_signals['short_signal']}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, signal_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    plt.show()

def plot_gann_wheel(price: float) -> None:
    """
    Create a visual representation of the Gann Square of 9 wheel
    
    Args:
        price: Price to calculate Gann wheel from
    """
    # Initialize Gann calculator
    gann = GannSquareOf9()
    
    # Calculate Gann levels
    gann_values = gann.calculate_levels(price)
    
    # Create polar plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
    
    # Extract values for different angles
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_rad = [math.radians(a) for a in angles_deg]
    
    # Calculate max values for each angle to determine radius
    max_values = [max(gann_values[f"{a}°"]) for a in angles_deg]
    max_value = max(max_values)
    
    # Plot lines for each angle
    for i, angle in enumerate(angles_deg):
        values = gann_values[f"{angle}°"]
        radii = [v / max_value for v in values]  # Normalize values
        
        # Plot each value as a point
        for j, radius in enumerate(radii):
            ax.plot([angles_rad[i]], [radius], 'o', markersize=8, 
                   color=plt.cm.tab10(i % 10), alpha=0.7)
            
            # Add value label for selected points
            if j % 2 == 0:  # Label every other point
                ax.text(angles_rad[i], radius, f"{values[j]:.1f}", 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Plot connecting lines along each angle
    for i, angle in enumerate(angles_deg):
        values = gann_values[f"{angle}°"]
        radii = [v / max_value for v in values]
        ax.plot([angles_rad[i]] * len(radii), radii, '-', 
               color=plt.cm.tab10(i % 10), alpha=0.5)
    
    # Plot connecting circles
    num_circles = 5  # Number of circles to plot
    for i in range(num_circles):
        radius = (i + 1) / num_circles
        circle_points = np.linspace(0, 2*np.pi, 100)
        ax.plot(circle_points, [radius] * len(circle_points), 'k-', alpha=0.2)
    
    # Highlight the central value
    root = math.sqrt(price)
    base = math.floor(root)
    central_value = base * base
    central_radius = central_value / max_value
    
    ax.plot(0, central_radius, 'ro', markersize=12)
    ax.text(0, central_radius, f"{central_value:.1f}", 
           fontsize=10, ha='center', va='center', color='white',
           bbox=dict(facecolor='red', alpha=0.7, boxstyle='round'))
    
    # Set angle labels
    ax.set_thetagrids(angles_deg, [f"{a}°" for a in angles_deg])
    
    # Remove radial labels and set title
    ax.set_yticklabels([])
    ax.set_title(f"Gann Square of 9 Wheel for Price: {price:.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create sample data
    data = generate_sample_data(symbol="NIFTY", days=30, start_price=22500.0)
    
    # Visualize Gann levels on candlestick chart
    visualize_gann_levels(data)
    
    # Visualize Gann wheel for a specific price
    price = data.iloc[-2]['close']  # Use the second-to-last close price
    plot_gann_wheel(price)