# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

from .base_strategy import BaseStrategy, TimeFrame, SignalType, OptionType, OptionStrikeSelection


class EquityStrategy(BaseStrategy):
    """
    Equity trading strategy with options hedging based on Gann Square of 9
    
    Trading rules:
    - For LONG signals: Buy Equity (cash market) + Buy Call Option (CE) for hedging
    - No short selling of equity, only options for SHORT signals
    """
    
    def __init__(self, 
                 symbol: str, 
                 timeframe: TimeFrame = TimeFrame.MINUTE_15,
                 option_strike_selection: OptionStrikeSelection = OptionStrikeSelection.ATM,
                 num_targets: int = 3,
                 buffer_percentage: float = 0.002,
                 risk_percentage: float = 1.0,
                 trailing_trigger_percentage: float = 0.5,
                 partial_booking_percentages: List[float] = [0.3, 0.3, 0.4],
                 option_lot_size: int = 1):
        """
        Initialize equity strategy
        
        Args:
            symbol: Trading symbol/ticker
            timeframe: Trading timeframe
            option_strike_selection: Method to select option strikes
            num_targets: Number of target levels to generate
            buffer_percentage: Buffer for stoploss calculation
            risk_percentage: Risk percentage per trade
            trailing_trigger_percentage: Percentage of target to trigger trailing
            partial_booking_percentages: List of percentages for partial profit booking
            option_lot_size: Standard lot size for options
        """
        super().__init__(
            symbol, timeframe, option_strike_selection, num_targets,
            buffer_percentage, risk_percentage, trailing_trigger_percentage,
            partial_booking_percentages
        )
        self.option_lot_size = option_lot_size
        
        # Equity-specific state
        self.equity_position = 0
        self.equity_entry_price = 0
        self.option_position = 0
        self.option_entry_price = 0
        self.option_symbol = ""
        self.active_targets = []
        self.stoploss = 0
        self.is_trailing_stoploss_active = False
        
    def process_market_data(self, data: pd.DataFrame) -> SignalType:
        """
        Process market data and generate signals
        
        Args:
            data: DataFrame with market data (OHLCV)
            
        Returns:
            Signal type
        """
        if data.empty or len(data) < 2:
            return SignalType.NO_SIGNAL
            
        # Get current and previous candle
        current_candle = data.iloc[-1]
        previous_candle = data.iloc[-2]
        
        current_price = current_candle['close']
        previous_close = previous_candle['close']
        
        # Check for exit conditions if we have an active position
        if self.equity_position > 0:  # Long position active
            # Check if stop loss is hit
            if current_price <= self.stoploss:
                return SignalType.EXIT_LONG
                
            # Check if we've hit any targets
            for i, (_, target_price) in enumerate(self.active_targets):
                if current_price >= target_price and i not in self.targets_hit:
                    self.targets_hit.append(i)
                    self.handle_target_hit(i, current_price)
                    
            # Update trailing stop if needed
            new_stop = self.update_trailing_stop(current_price)
            if new_stop is not None:
                self.stoploss = new_stop
                
            return SignalType.NO_SIGNAL
                
        # If no position is active, check for new signals
        gann_signals = self.get_gann_signals(current_price, previous_close)
        
        if gann_signals['long_signal']:
            # Save signal info for order execution
            self.active_targets = gann_signals['buy_targets']
            self.stoploss = gann_signals['long_stoploss']
            self.targets_hit = []
            self.is_trailing_stoploss_active = False
            return SignalType.LONG
            
        if gann_signals['short_signal']:
            # For equity strategy, we don't short the equity, only buy put options
            # This would be handled in the option_strategy.py implementation
            pass
            
        return SignalType.NO_SIGNAL
    
    def calculate_position_size(self, capital: float, current_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            capital: Available capital
            current_price: Current market price
            stop_loss: Stop loss price
            
        Returns:
            Position size (quantity)
        """
        # Calculate risk amount based on risk percentage
        risk_amount = capital * (self.risk_percentage / 100)
        
        # Calculate risk per share
        risk_per_share = current_price - stop_loss
        
        if risk_per_share <= 0 or risk_per_share >= current_price * 0.1:
            # Safety check: If risk is negative or too large (>10% of price)
            # Use a default 2% risk per share
            risk_per_share = current_price * 0.02
            
        # Calculate shares to buy
        shares = int(risk_amount / risk_per_share)
        
        # Ensure minimum quantity is 1
        return max(1, shares)
    
    def select_option_strike(self, 
                            spot_price: float, 
                            option_type: OptionType,
                            option_chain: pd.DataFrame) -> str:
        """
        Select option strike based on strategy parameters
        
        Args:
            spot_price: Current spot price
            option_type: Call or Put option
            option_chain: DataFrame with option chain data
            
        Returns:
            Selected option strike price
        """
        # Ensure option_chain has 'strike' and 'type' columns
        if 'strike' not in option_chain.columns or 'type' not in option_chain.columns:
            raise ValueError("Option chain must have 'strike' and 'type' columns")
            
        # Filter option chain by type
        filtered_chain = option_chain[option_chain['type'] == option_type.value]
        
        if filtered_chain.empty:
            raise ValueError(f"No {option_type.value} options found in the option chain")
            
        # Find ATM strike (closest to spot)
        atm_strike = filtered_chain.iloc[(filtered_chain['strike'] - spot_price).abs().argsort()[0]]['strike']
        
        # Select strike based on strategy parameter
        strikes = sorted(filtered_chain['strike'].unique())
        atm_index = strikes.index(atm_strike)
        
        if self.option_strike_selection == OptionStrikeSelection.ATM:
            return atm_strike
            
        if option_type == OptionType.CALL:
            # For calls, ITM strikes are below spot
            if self.option_strike_selection == OptionStrikeSelection.ITM and atm_index > 0:
                return strikes[atm_index - 1]
            elif self.option_strike_selection == OptionStrikeSelection.ITM2 and atm_index > 1:
                return strikes[atm_index - 2]
            elif self.option_strike_selection == OptionStrikeSelection.OTM and atm_index < len(strikes) - 1:
                return strikes[atm_index + 1]
            elif self.option_strike_selection == OptionStrikeSelection.OTM2 and atm_index < len(strikes) - 2:
                return strikes[atm_index + 2]
        else:  # PUT
            # For puts, ITM strikes are above spot
            if self.option_strike_selection == OptionStrikeSelection.ITM and atm_index < len(strikes) - 1:
                return strikes[atm_index + 1]
            elif self.option_strike_selection == OptionStrikeSelection.ITM2 and atm_index < len(strikes) - 2:
                return strikes[atm_index + 2]
            elif self.option_strike_selection == OptionStrikeSelection.OTM and atm_index > 0:
                return strikes[atm_index - 1]
            elif self.option_strike_selection == OptionStrikeSelection.OTM2 and atm_index > 1:
                return strikes[atm_index - 2]
                
        # Default to ATM if the requested strike is not available
        return atm_strike
    
    def handle_target_hit(self, target_index: int, current_price: float) -> Dict[str, Union[float, int]]:
        """
        Handle when a target is hit - implements partial profit booking
        
        Args:
            target_index: Index of the target that was hit
            current_price: Current market price
            
        Returns:
            Dictionary with action details
        """
        # Calculate shares to sell based on partial booking percentages
        if target_index >= len(self.partial_booking_percentages):
            target_index = len(self.partial_booking_percentages) - 1
            
        shares_to_sell = int(self.equity_position * self.partial_booking_percentages[target_index])
        
        # Activate trailing stop after first target is hit
        if target_index == 0:
            self.is_trailing_stoploss_active = True
            
            # Initial trailing stop at entry (or midway between entry and first target)
            first_target_price = self.active_targets[0][1]
            self.stoploss = max(
                self.stoploss,  # Original stop
                self.equity_entry_price + (current_price - self.equity_entry_price) * 0.3  # 30% of current profit
            )
        
        # For later targets, move stop to previous target
        elif target_index > 0:
            previous_target = self.active_targets[target_index - 1][1]
            self.stoploss = max(self.stoploss, previous_target)
        
        return {
            "action": "PARTIAL_EXIT",
            "target_index": target_index,
            "shares_to_sell": shares_to_sell,
            "current_price": current_price,
            "new_stoploss": self.stoploss
        }
    
    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """
        Update trailing stop based on price movement
        
        Args:
            current_price: Current market price
            
        Returns:
            New trailing stop or None if no update
        """
        if not self.is_trailing_stoploss_active:
            return None
            
        # Calculate total profit percentage
        profit_percentage = (current_price - self.equity_entry_price) / self.equity_entry_price * 100
        
        # Calculate new potential stop
        if len(self.targets_hit) == 0:
            # No targets hit yet, don't trail
            return None
        elif len(self.targets_hit) == 1:
            # First target hit, trail at breakeven or higher
            potential_stop = max(self.equity_entry_price, self.stoploss)
        else:
            # Multiple targets hit, trail at previous target
            last_hit_target_index = max(self.targets_hit)
            if last_hit_target_index > 0 and last_hit_target_index < len(self.active_targets):
                previous_target = self.active_targets[last_hit_target_index - 1][1]
                potential_stop = max(previous_target, self.stoploss)
            else:
                potential_stop = self.stoploss
                
        # Only update if new stop is higher
        if potential_stop > self.stoploss:
            return potential_stop
            
        return None