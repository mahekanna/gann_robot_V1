# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
Position sizing calculator for risk management
"""

from typing import Optional, Dict, Any
import math
import logging

class PositionSizer:
    """
    Position sizer for risk management
    Calculates appropriate position sizes based on risk parameters
    """
    
    def __init__(self, risk_percentage: float = 1.0, max_risk_per_trade: Optional[float] = None):
        """
        Initialize position sizer
        
        Args:
            risk_percentage: Percentage of capital to risk per trade
            max_risk_per_trade: Maximum amount to risk per trade (overrides percentage if set)
        """
        self.risk_percentage = risk_percentage
        self.max_risk_per_trade = max_risk_per_trade
        self.logger = logging.getLogger("position_sizer")
    
    def calculate_position_size(self, 
                              capital: float, 
                              entry_price: float, 
                              stop_loss: float, 
                              slippage: float = 0.0,
                              min_quantity: int = 1,
                              lot_size: int = 1) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters
        
        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss: Stop loss price
            slippage: Expected slippage as percentage
            min_quantity: Minimum position quantity
            lot_size: Lot size for the instrument
            
        Returns:
            Dictionary with position sizing information
        """
        # Calculate risk per share
        if entry_price > stop_loss:  # Long position
            risk_per_share = entry_price - stop_loss
            # Add slippage to entry price
            adjusted_entry = entry_price * (1 + slippage/100)
            position_type = "LONG"
        else:  # Short position
            risk_per_share = stop_loss - entry_price
            # Add slippage to entry price
            adjusted_entry = entry_price * (1 - slippage/100)
            position_type = "SHORT"
            
        # Calculate risk amount
        risk_amount = capital * (self.risk_percentage / 100)
        
        # Cap at max risk if specified
        if self.max_risk_per_trade is not None:
            risk_amount = min(risk_amount, self.max_risk_per_trade)
            
        # Calculate shares to trade
        if risk_per_share <= 0:
            self.logger.warning("Invalid risk per share: Stop loss and entry price may be reversed")
            return {
                "position_size": 0,
                "risk_amount": 0,
                "capital_required": 0,
                "risk_percentage": 0,
                "position_type": position_type,
                "error": "Invalid risk per share"
            }
            
        shares = risk_amount / risk_per_share
        
        # Adjust to lot size
        if lot_size > 1:
            shares = math.floor(shares / lot_size) * lot_size
        else:
            shares = math.floor(shares)
            
        # Ensure minimum quantity
        shares = max(shares, min_quantity)
        
        # Calculate capital required and actual risk
        capital_required = shares * adjusted_entry
        actual_risk = shares * risk_per_share
        actual_risk_percentage = (actual_risk / capital) * 100
        
        return {
            "position_size": int(shares),
            "risk_amount": actual_risk,
            "capital_required": capital_required,
            "risk_percentage": actual_risk_percentage,
            "position_type": position_type
        }
    
    def calculate_option_quantity(self,
                                capital: float,
                                option_price: float,
                                stop_loss: float,
                                risk_adjustment: float = 1.0,
                                min_contracts: int = 1,
                                lot_size: int = 1) -> Dict[str, Any]:
        """
        Calculate option position size based on risk parameters
        
        Args:
            capital: Available capital
            option_price: Option premium
            stop_loss: Stop loss price for the option
            risk_adjustment: Adjustment factor for option risk (typically > 1 for higher-risk options)
            min_contracts: Minimum number of option contracts
            lot_size: Lot size for the option
            
        Returns:
            Dictionary with option position sizing information
        """
        risk_per_contract = (option_price - stop_loss) * lot_size
        
        if risk_per_contract <= 0:
            self.logger.warning("Invalid risk per contract: Stop loss higher than option price")
            return {
                "contracts": 0,
                "risk_amount": 0,
                "capital_required": 0,
                "risk_percentage": 0,
                "error": "Invalid risk per contract"
            }
            
        # Adjust risk percentage for options (typically higher risk)
        adjusted_risk_percentage = self.risk_percentage * risk_adjustment
        
        # Calculate risk amount
        risk_amount = capital * (adjusted_risk_percentage / 100)
        
        # Cap at max risk if specified
        if self.max_risk_per_trade is not None:
            risk_amount = min(risk_amount, self.max_risk_per_trade)
            
        # Calculate number of contracts
        contracts = risk_amount / risk_per_contract
        contracts = math.floor(contracts)
        
        # Ensure minimum contracts
        contracts = max(contracts, min_contracts)
        
        # Calculate capital required and actual risk
        capital_required = contracts * option_price * lot_size
        actual_risk = contracts * risk_per_contract
        actual_risk_percentage = (actual_risk / capital) * 100
        
        return {
            "contracts": int(contracts),
            "risk_amount": actual_risk,
            "capital_required": capital_required,
            "risk_percentage": actual_risk_percentage,
            "option_price": option_price,
            "stop_loss": stop_loss
        }