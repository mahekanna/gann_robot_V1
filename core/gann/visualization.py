# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""


"""
Visualization utilities for Gann Square of 9 calculations
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Union, Any

from .square_of_9 import GannSquareOf9

class GannVisualizer:
    """
    Visualization tools for Gann Square of 9 calculations
    """
    
    def __init__(self, gann_calculator: Optional[GannSquareOf9] = None):
        """
        Initialize the visualizer
        
        Args:
            gann_calculator: Gann Square of 9 calculator instance
        """
        self.gann = gann_calculator if gann_calculator else GannSquareOf9()
        self.colors = {
            'buy_above': 'blue',
            'sell_below': 'red',
            'buy_targets': 'green',
            'sell_targets': 'purple',
            'long_stoploss': 'orangered',
            'short_stoploss': 'darkblue',
            'price': 'black',
            'cardinal': 'blue',
            'ordinal': 'green'
        }
    
    def plot_price_levels(self, price: float, num_targets: int = 3, 
                         buffer_percentage: float = 0.002, 
                         show_plot: bool = True, 
                         save_path: Optional[str] = None) -> None:
        """
        Plot buy/sell levels, targets, and stop loss levels for a given price
        
        Args:
            price: Previous candle close price
            num_targets: Number of target levels to generate
            buffer_percentage: Buffer percentage for stoploss calculation
            show_plot: Whether to display the plot
            save_path: Path to save the plot image (if provided)
        """
        # Get analysis report
        analysis = self.gann.get_analysis_report(price, num_targets)
        
        if not analysis:
            print(f"Could not generate analysis for price {price}")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up y-axis range
        all_levels = [price, analysis['buy_above'], analysis['sell_below']]
        if analysis['long_stoploss']: all_levels.append(analysis['long_stoploss'])
        if analysis['short_stoploss']: all_levels.append(analysis['short_stoploss'])
        for _, level in analysis['buy_targets']: all_levels.append(level)
        for _, level in analysis['sell_targets']: all_levels.append(level)
        
        min_level = min(all_levels) * 0.995
        max_level = max(all_levels) * 1.005
        
        # Plot current price
        ax.axhline(y=price, color=self.colors['price'], linestyle='-', linewidth=2, 
                  label=f'Current Price: {price:.2f}')
        
        # Plot buy_above level
        ax.axhline(y=analysis['buy_above'], color=self.colors['buy_above'], linestyle='-', linewidth=1.5, 
                  label=f'Buy Above: {analysis["buy_above"]:.2f}')
        
        # Plot sell_below level
        ax.axhline(y=analysis['sell_below'], color=self.colors['sell_below'], linestyle='-', linewidth=1.5, 
                  label=f'Sell Below: {analysis["sell_below"]:.2f}')
        
        # Plot buy targets
        for i, (angle, level) in enumerate(analysis['buy_targets']):
            ax.axhline(y=level, color=self.colors['buy_targets'], linestyle='--', linewidth=1, 
                      alpha=0.7)
            plt.text(0.95, level, f"Buy Target {i+1}: {level:.2f} ({angle})", 
                   ha='right', va='center', color=self.colors['buy_targets'])
        
        # Plot sell targets
        for i, (angle, level) in enumerate(analysis['sell_targets']):
            ax.axhline(y=level, color=self.colors['sell_targets'], linestyle='--', linewidth=1, 
                      alpha=0.7)
            plt.text(0.95, level, f"Sell Target {i+1}: {level:.2f} ({angle})", 
                   ha='right', va='center', color=self.colors['sell_targets'])
        
        # Plot stop loss levels
        if analysis['long_stoploss']:
            ax.axhline(y=analysis['long_stoploss'], color=self.colors['long_stoploss'], linestyle=':', linewidth=1.5, 
                      label=f'Long Stoploss: {analysis["long_stoploss"]:.2f}')
        
        if analysis['short_stoploss']:
            ax.axhline(y=analysis['short_stoploss'], color=self.colors['short_stoploss'], linestyle=':', linewidth=1.5, 
                      label=f'Short Stoploss: {analysis["short_stoploss"]:.2f}')
        
        # Set labels and title
        ax.set_title(f'Gann Square of 9 Levels for Price: {price:.2f}')
        ax.set_ylabel('Price')
        ax.set_ylim(min_level, max_level)
        
        # Set x-axis
        ax.set_xlim(0, 1)
        ax.set_xticks([])  # Hide x-axis ticks
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_gann_wheel(self, price: float, num_values: int = 10, 
                       show_plot: bool = True, 
                       save_path: Optional[str] = None) -> None:
        """
        Create a visual representation of the Gann Square of 9 wheel
        
        Args:
            price: Price to calculate Gann wheel from
            num_values: Number of values to plot for each angle
            show_plot: Whether to display the plot
            save_path: Path to save the plot image (if provided)
        """
        # Calculate Gann levels
        increments = [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
        gann_values = self.gann.calculate_levels(price, increments, num_values=num_values)
        
        # Create polar plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)
        
        # Extract values for different angles
        angles = self.gann.angles
        angles_rad = [math.radians(int(angle.replace('deg', ''))) for angle in angles]
        
        # Calculate max values for each angle to determine radius
        max_values = [max(gann_values[angle]) for angle in angles]
        max_value = max(max_values)
        
        # Plot lines for each angle
        for i, angle in enumerate(angles):
            values = gann_values[angle]
            radii = [v / max_value for v in values]  # Normalize values
            
            # Determine if cardinal or ordinal angle
            is_cardinal = int(angle.replace('deg', '')) % 90 == 0
            color = self.colors['cardinal'] if is_cardinal else self.colors['ordinal']
            
            # Plot each value as a point
            for j, radius in enumerate(radii):
                ax.plot([angles_rad[i]], [radius], 'o', markersize=8, 
                       color=color, alpha=0.7)
                
                # Add value label for selected points
                if j % 2 == 0:  # Label every other point
                    ax.text(angles_rad[i], radius, f"{values[j]:.1f}", 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Plot connecting lines along each angle
        for i, angle in enumerate(angles):
            values = gann_values[angle]
            radii = [v / max_value for v in values]
            is_cardinal = int(angle.replace('deg', '')) % 90 == 0
            color = self.colors['cardinal'] if is_cardinal else self.colors['ordinal']
            ax.plot([angles_rad[i]] * len(radii), radii, '-', 
                   color=color, alpha=0.5)
        
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
        ax.set_thetagrids(np.degrees(angles_rad), [angle for angle in angles])
        
        # Remove radial labels and set title
        ax.set_yticklabels([])
        ax.set_title(f"Gann Square of 9 Wheel for Price: {price:.2f}")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wheel plot saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_gann_square(self, price: float, size: int = 9, 
                        show_plot: bool = True, 
                        save_path: Optional[str] = None) -> None:
        """
        Create a visual representation of the Gann Square of 9 as a grid
        
        Args:
            price: Price to calculate Gann square from
            size: Size of the square (must be odd)
            show_plot: Whether to display the plot
            save_path: Path to save the plot image (if provided)
        """
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
        
        # Calculate square root and base
        root = math.sqrt(price)
        base = math.floor(root)
        central_value = base * base
        
        # Create a square grid around base value
        grid = np.zeros((size, size))
        center = size // 2
        
        # Fill grid with values spiraling outward
        x, y = center, center
        grid[y, x] = central_value
        
        # Directions: right, down, left, up
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        
        value = central_value
        steps = 1
        direction = 0
        
        while x >= 0 and x < size and y >= 0 and y < size:
            for _ in range(2):  # Each direction is repeated for steps
                for _ in range(steps):
                    x += dx[direction]
                    y += dy[direction]
                    
                    if x < 0 or x >= size or y < 0 or y >= size:
                        break
                        
                    # Increment the value based on position
                    value += 1
                    grid[y, x] = value
                    
                direction = (direction + 1) % 4
                
            steps += 1
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a heatmap
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=np.min(grid[grid > 0]), vmax=np.max(grid))
        
        # Plot each cell
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    ax.add_patch(Rectangle((j, size-i-1), 1, 1, 
                                         facecolor=cmap(norm(grid[i, j])), 
                                         edgecolor='k', alpha=0.7))
                    
                    # Add value text
                    plt.text(j+0.5, size-i-0.5, f"{grid[i, j]:.0f}", 
                            ha='center', va='center', 
                            color='white' if norm(grid[i, j]) > 0.5 else 'black',
                            fontsize=9)
        
        # Highlight the price closest to input
        closest_idx = np.unravel_index(np.argmin(np.abs(grid - price)), grid.shape)
        highlight_y, highlight_x = closest_idx
        ax.add_patch(Rectangle((highlight_x, size-highlight_y-1), 1, 1, 
                             facecolor='none', edgecolor='red', linewidth=3))
        
        # Set the limits and ticks
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xticks(np.arange(0, size+1, 1))
        ax.set_yticks(np.arange(0, size+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add grid
        ax.grid(True, linestyle='-', color='k', linewidth=0.5)
        
        # Add title
        plt.title(f"Gann Square of 9 Grid for Price: {price:.2f}")
        
        # Add colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Value')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Square plot saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
