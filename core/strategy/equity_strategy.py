# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
Equity trading strategy with options hedging based on Gann Square of 9
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from .base_strategy import BaseStrategy, TimeFrame, SignalType, OptionType, OptionStrikeSelection
from ..gann.square_of_9 import GannSquareOf9
from ..risk.position_sizing import PositionSizer

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
                 option_lot_size: int = 1,
                 use_options: bool = True):
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
            use_options: Whether to use options for hedging
        """
        super().__init__(
            symbol, timeframe, option_strike_selection, num_targets,
            buffer_percentage, risk_percentage, trailing_trigger_percentage,
            partial_booking_percentages
        )
        self.option_lot_size = option_lot_size
        self.use_options = use_options
        
        # Set up position sizer for risk management
        self.position_sizer = PositionSizer(risk_percentage=risk_percentage)
        
        # Set up logger
        self.logger = logging.getLogger(f"equity_strategy_{symbol}")
        
        # Equity-specific state
        self.equity_position = 0
        self.equity_entry_price = 0
        self.option_position = 0
        self.option_entry_price = 0
        self.option_symbol = ""
        self.active_targets = []
        self.stoploss = 0
        self.current_signal = SignalType.NO_SIGNAL
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
                self.logger.info(f"Stop loss hit at {current_price:.2f} (stop: {self.stoploss:.2f})")
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
                self.logger.info(f"Trailing stop updated to {self.stoploss:.2f}")
                
            return SignalType.NO_SIGNAL
                
        # If no position is active, check for new signals
        gann_signals = self.get_gann_signals(current_price, previous_close)
        
        if gann_signals['long_signal']:
            # Save signal info for order execution
            self.active_targets = gann_signals['buy_targets']
            self.stoploss = gann_signals['long_stoploss']
            self.targets_hit = []
            self.is_trailing_stoploss_active = False
            self.current_signal = SignalType.LONG
            self.logger.info(f"LONG signal generated at {current_price:.2f}")
            return SignalType.LONG
            
        if gann_signals['short_signal']:
            # For equity strategy, we don't short the equity, only buy put options
            self.active_targets = gann_signals['sell_targets']
            self.stoploss = gann_signals['short_stoploss']
            self.targets_hit = []
            self.is_trailing_stoploss_active = False
            self.current_signal = SignalType.SHORT
            self.logger.info(f"SHORT signal generated at {current_price:.2f}")
            return SignalType.SHORT
            
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
        position_info = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=current_price,
            stop_loss=stop_loss,
            slippage=0.1,  # 0.1% slippage assumption
            min_quantity=1
        )
        
        # Log position sizing information
        self.logger.info(f"Position sizing: {position_info}")
        
        return position_info['position_size']
    
    def calculate_option_size(self, capital: float, option_price: float, stop_loss: float) -> int:
        """
        Calculate option position size based on risk parameters
        
        Args:
            capital: Available capital
            option_price: Option premium
            stop_loss: Stop loss price for the option
            
        Returns:
            Number of option contracts
        """
        option_info = self.position_sizer.calculate_option_quantity(
            capital=capital,
            option_price=option_price,
            stop_loss=stop_loss,
            risk_adjustment=1.5,  # Higher risk for options
            min_contracts=1,
            lot_size=self.option_lot_size
        )
        
        # Log option position sizing information
        self.logger.info(f"Option sizing: {option_info}")
        
        return option_info['contracts']
    
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
        # Ensure option_chain has required columns
        required_columns = ['strike', 'type']
        missing_columns = [col for col in required_columns if col not in option_chain.columns]
        if missing_columns:
            raise ValueError(f"Option chain missing required columns: {missing_columns}")
            
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
            self.logger.info(f"First target hit. Activated trailing stop at {self.stoploss:.2f}")
        
        # For later targets, move stop to previous target
        elif target_index > 0:
            previous_target = self.active_targets[target_index - 1][1]
            self.stoploss = max(self.stoploss, previous_target)
            self.logger.info(f"Target {target_index+1} hit. Moved stop to {self.stoploss:.2f}")
        
        self.logger.info(f"Target {target_index+1} hit at {current_price:.2f}. Sell {shares_to_sell} shares.")
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

    def execute_long_strategy(self, 
                             current_price: float, 
                             capital: float, 
                             option_chain: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute long strategy: Buy equity and hedge with call options
        
        Args:
            current_price: Current market price
            capital: Available capital
            option_chain: Option chain data (if using options)
            
        Returns:
            Dictionary with execution details
        """
        # Calculate position size
        equity_quantity = self.calculate_position_size(
            capital=capital,
            current_price=current_price,
            stop_loss=self.stoploss
        )
        
        result = {
            "action": "ENTER_LONG",
            "symbol": self.symbol,
            "price": current_price,
            "stop_loss": self.stoploss,
            "quantity": equity_quantity,
            "targets": self.active_targets,
            "position_type": "EQUITY",
            "orders": [
                {
                    "symbol": self.symbol,
                    "order_type": "MARKET",
                    "transaction_type": "BUY",
                    "quantity": equity_quantity,
                    "price": current_price,
                    "product_type": "CNC"  # Cash and Carry
                }
            ]
        }
        
        # Add option hedge if enabled and option chain is provided
        if self.use_options and option_chain is not None and not option_chain.empty:
            try:
                # Select strike
                call_strike = self.select_option_strike(
                    spot_price=current_price,
                    option_type=OptionType.CALL,
                    option_chain=option_chain
                )
                
                # Get call option details
                call_option = option_chain[
                    (option_chain['type'] == OptionType.CALL.value) & 
                    (option_chain['strike'] == call_strike)
                ].iloc[0]
                
                call_price = call_option.get('last_price', current_price * 0.03)  # Fallback to 3% of spot
                call_symbol = call_option.get('symbol', f"{self.symbol}{call_strike}CE")
                
                # Calculate option stop loss (typically 50% of premium for long options)
                option_stop = call_price * 0.5
                
                # Calculate position size for call option
                option_capital = capital * 0.2  # Use 20% of capital for option hedge
                call_quantity = self.calculate_option_size(
                    capital=option_capital,
                    option_price=call_price,
                    stop_loss=option_stop
                )
                
                # Ensure at least one lot
                call_quantity = max(1, call_quantity)
                
                # Add to result
                result["option_hedge"] = {
                    "symbol": call_symbol,
                    "strike": call_strike,
                    "option_type": "CE",
                    "quantity": call_quantity,
                    "price": call_price,
                    "stop_loss": option_stop
                }
                
                # Add option order
                result["orders"].append({
                    "symbol": call_symbol,
                    "order_type": "MARKET",
                    "transaction_type": "BUY",
                    "quantity": call_quantity * self.option_lot_size,
                    "price": call_price,
                    "product_type": "MIS"  # Intraday
                })
                
            except Exception as e:
                self.logger.error(f"Error setting up option hedge: {e}")
        
        return result
        
    def execute_short_strategy(self, 
                              current_price: float, 
                              capital: float, 
                              option_chain: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute short strategy: Buy put options only
        
        Args:
            current_price: Current market price
            capital: Available capital
            option_chain: Option chain data
            
        Returns:
            Dictionary with execution details
        """
        result = {
            "action": "ENTER_SHORT",
            "symbol": self.symbol,
            "price": current_price,
            "stop_loss": self.stoploss,
            "targets": self.active_targets,
            "position_type": "OPTION",
            "orders": []
        }
        
        # Only use options for short strategy
        if option_chain is not None and not option_chain.empty:
            try:
                # Select strike
                put_strike = self.select_option_strike(
                    spot_price=current_price,
                    option_type=OptionType.PUT,
                    option_chain=option_chain
                )
                
                # Get put option details
                put_option = option_chain[
                    (option_chain['type'] == OptionType.PUT.value) & 
                    (option_chain['strike'] == put_strike)
                ].iloc[0]
                
                put_price = put_option.get('last_price', current_price * 0.03)  # Fallback to 3% of spot
                put_symbol = put_option.get('symbol', f"{self.symbol}{put_strike}PE")
                
                # Calculate option stop loss (typically 50% of premium for long options)
                option_stop = put_price * 0.5
                
                # Calculate position size for put option
                put_quantity = self.calculate_option_size(
                    capital=capital,
                    option_price=put_price,
                    stop_loss=option_stop
                )
                
                # Ensure at least one lot
                put_quantity = max(1, put_quantity)
                
                # Add to result
                result["put_option"] = {
                    "symbol": put_symbol,
                    "strike": put_strike,
                    "option_type": "PE",
                    "quantity": put_quantity,
                    "price": put_price,
                    "stop_loss": option_stop
                }
                
                # Add option order
                result["orders"].append({
                    "symbol": put_symbol,
                    "order_type": "MARKET",
                    "transaction_type": "BUY",
                    "quantity": put_quantity * self.option_lot_size,
                    "price": put_price,
                    "product_type": "MIS"  # Intraday
                })
                
            except Exception as e:
                self.logger.error(f"Error setting up put option: {e}")
                return {"error": f"Failed to create short position: {e}"}
        else:
            return {"error": "Option chain required for short strategy"}
        
        return result
    
    def execute_exit_long(self, 
                         current_price: float, 
                         position_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute exit for long positions
        
        Args:
            current_price: Current market price
            position_details: Current position details
            
        Returns:
            Dictionary with exit details
        """
        result = {
            "action": "EXIT_LONG",
            "symbol": self.symbol,
            "price": current_price,
            "quantity": position_details.get("quantity", 0),
            "pnl": (current_price - position_details.get("entry_price", current_price)) * 
                  position_details.get("quantity", 0),
            "orders": []
        }
        
        # Add equity exit order
        if position_details.get("quantity", 0) > 0:
            result["orders"].append({
                "symbol": self.symbol,
                "order_type": "MARKET",
                "transaction_type": "SELL",
                "quantity": position_details.get("quantity", 0),
                "price": current_price,
                "product_type": "CNC"  # Cash and Carry
            })
        
        # Add option exit order if hedge exists
        if "option_hedge" in position_details:
            hedge = position_details["option_hedge"]
            result["orders"].append({
                "symbol": hedge.get("symbol", ""),
                "order_type": "MARKET",
                "transaction_type": "SELL",
                "quantity": hedge.get("quantity", 0) * self.option_lot_size,
                "price": current_price,  # This will be market price
                "product_type": "MIS"  # Intraday
            })
        
        return result
    
    def execute_exit_short(self, 
                          current_price: float, 
                          position_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute exit for short positions (put options)
        
        Args:
            current_price: Current market price
            position_details: Current position details
            
        Returns:
            Dictionary with exit details
        """
        result = {
            "action": "EXIT_SHORT",
            "symbol": self.symbol,
            "price": current_price,
            "orders": []
        }
        
        # Add put option exit order
        if "put_option" in position_details:
            put = position_details["put_option"]
            result["orders"].append({
                "symbol": put.get("symbol", ""),
                "order_type": "MARKET",
                "transaction_type": "SELL",
                "quantity": put.get("quantity", 0) * self.option_lot_size,
                "price": current_price,  # This will be market price
                "product_type": "MIS"  # Intraday
            })
        
        return result