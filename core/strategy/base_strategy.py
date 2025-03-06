# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd

from ..gann.square_of_9 import GannSquareOf9


class TimeFrame(Enum):
    """Supported timeframes for strategy execution"""
    MINUTE_1 = "1minute"
    MINUTE_3 = "3minute"  
    MINUTE_5 = "5minute"
    MINUTE_10 = "10minute"
    MINUTE_15 = "15minute"
    MINUTE_30 = "30minute"
    HOUR_1 = "1hour"
    HOUR_2 = "2hour"
    HOUR_4 = "4hour"
    DAY_1 = "1day"
    WEEK_1 = "1week"


class SignalType(Enum):
    """Types of trading signals"""
    LONG = "LONG"  # Buy equity + Call option
    SHORT = "SHORT"  # Buy Put option only
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


class OptionType(Enum):
    """Option types"""
    CALL = "CE"
    PUT = "PE"


class OptionStrikeSelection(Enum):
    """Option strike selection methods"""
    ATM = "ATM"  # At The Money
    ITM = "ITM"  # In The Money (1 strike)
    ITM2 = "ITM2"  # In The Money (2 strikes)
    OTM = "OTM"  # Out of The Money (1 strike)
    OTM2 = "OTM2"  # Out of The Money (2 strikes)


class BaseStrategy(ABC):
    """
    Base class for all trading strategies using Gann Square of 9
    """
    
    def __init__(self, 
                 symbol: str, 
                 timeframe: TimeFrame = TimeFrame.MINUTE_15,
                 option_strike_selection: OptionStrikeSelection = OptionStrikeSelection.ATM,
                 num_targets: int = 3,
                 buffer_percentage: float = 0.002,
                 risk_percentage: float = 1.0,
                 trailing_trigger_percentage: float = 0.5,
                 partial_booking_percentages: List[float] = [0.3, 0.3, 0.4]):
        """
        Initialize the base strategy
        
        Args:
            symbol: Trading symbol/ticker
            timeframe: Trading timeframe
            option_strike_selection: Method to select option strikes
            num_targets: Number of target levels to generate
            buffer_percentage: Buffer for stoploss calculation
            risk_percentage: Risk percentage per trade
            trailing_trigger_percentage: Percentage of target to trigger trailing
            partial_booking_percentages: List of percentages for partial profit booking
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.option_strike_selection = option_strike_selection
        self.num_targets = num_targets
        self.risk_percentage = risk_percentage
        self.trailing_trigger_percentage = trailing_trigger_percentage
        self.partial_booking_percentages = partial_booking_percentages
        
        # Ensure partial booking percentages sum to 1.0
        total = sum(partial_booking_percentages)
        if abs(total - 1.0) > 0.01:  # Allow small rounding errors
            self.partial_booking_percentages = [p/total for p in partial_booking_percentages]
            
        # Initialize Gann calculator
        self.gann = GannSquareOf9(buffer_percentage=buffer_percentage)
        
        # State tracking
        self.current_position = None
        self.current_targets = []
        self.targets_hit = []
        self.is_trailing_active = False
        self.trailing_stop = None
    
    @abstractmethod
    def process_market_data(self, data: pd.DataFrame) -> SignalType:
        """
        Process market data and generate signals
        
        Args:
            data: DataFrame with market data (OHLCV)
            
        Returns:
            Signal type
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, capital: float, current_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            capital: Available capital
            current_price: Current market price
            stop_loss: Stop loss price
            
        Returns:
            Position size (quantity/lots)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def handle_target_hit(self, target_index: int, current_price: float) -> Dict[str, Union[float, int]]:
        """
        Handle when a target is hit
        
        Args:
            target_index: Index of the target that was hit
            current_price: Current market price
            
        Returns:
            Dictionary with action details
        """
        pass
    
    @abstractmethod
    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """
        Update trailing stop based on price movement
        
        Args:
            current_price: Current market price
            
        Returns:
            New trailing stop or None if no update
        """
        pass
    
    def get_gann_signals(self, current_price: float, previous_close: float) -> Dict:
        """
        Get Gann signals based on current market data
        
        Args:
            current_price: Current market price
            previous_close: Previous candle's closing price
            
        Returns:
            Dictionary with signal information
        """
        return self.gann.generate_signals(current_price, previous_close, self.num_targets)
    
    def format_signals(self, signals: Dict) -> str:
        """
        Format signals for display or logging
        
        Args:
            signals: Signal dictionary from get_gann_signals
            
        Returns:
            Formatted string with signal information
        """
        output = f"Symbol: {self.symbol} (Timeframe: {self.timeframe.value})\n"
        output += f"Current Price: {signals['current_price']:.2f}\n"
        output += f"Previous Close: {signals['previous_close']:.2f}\n"
        output += f"Buy Above: {signals['buy_above']:.2f}\n"
        output += f"Sell Below: {signals['sell_below']:.2f}\n"
        
        if signals['long_signal']:
            output += "SIGNAL: LONG (Buy Equity + Call Option)\n"
        elif signals['short_signal']:
            output += "SIGNAL: SHORT (Buy Put Option Only)\n"
        else:
            output += "SIGNAL: None\n"
        
        output += "\nBuy Targets:\n"
        for i, (angle, price) in enumerate(signals['buy_targets']):
            output += f"  Target {i+1}: {price:.2f} ({angle})\n"
        
        output += "\nSell Targets:\n"
        for i, (angle, price) in enumerate(signals['sell_targets']):
            output += f"  Target {i+1}: {price:.2f} ({angle})\n"
        
        output += f"\nLong Stop Loss: {signals['long_stoploss']:.2f}\n"
        output += f"Short Stop Loss: {signals['short_stoploss']:.2f}\n"
        
        return output