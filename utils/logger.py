# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

# File: util/logger.py
"""
Logging utilities for the Gann Trading System
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None, 
                log_level: str = 'INFO',
                console: bool = True) -> logging.Logger:
    """
    Set up and configure a logger
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    while logger.handlers:
        logger.handlers.pop()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class TradeLogger:
    """
    Specialized logger for trading activities
    Logs trades, signals, and other trading events to a dedicated file
    """
    
    def __init__(self, 
                strategy_name: str, 
                symbol: str,
                log_dir: str = 'logs/trades',
                log_level: str = 'INFO'):
        """
        Initialize trade logger
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file name
        today = datetime.now().strftime('%Y%m%d')
        log_file = self.log_dir / f"{strategy_name}_{symbol}_{today}.log"
        
        # Set up logger
        self.logger = setup_logger(
            name=f"trade_{strategy_name}_{symbol}",
            log_file=str(log_file),
            log_level=log_level,
            console=False  # Trade logs only go to file
        )
        
        # Add console logger for critical events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only WARNING and above
        formatter = logging.Formatter(
            '%(asctime)s - TRADE - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_signal(self, signal_type: str, price: float, details: dict = None):
        """
        Log a trading signal
        
        Args:
            signal_type: Type of signal
            price: Current price
            details: Additional signal details
        """
        details_str = ""
        if details:
            details_str = " - " + " - ".join([f"{k}: {v}" for k, v in details.items()])
            
        self.logger.info(f"SIGNAL - {signal_type} - Price: {price:.2f}{details_str}")
    
    def log_order(self, order_type: str, side: str, quantity: int, price: float, 
                 order_id: str = None, status: str = "CREATED"):
        """
        Log an order
        
        Args:
            order_type: Type of order (MARKET, LIMIT, etc.)
            side: Order side (BUY, SELL)
            quantity: Order quantity
            price: Order price
            order_id: Order ID
            status: Order status
        """
        order_id_str = f"- Order ID: {order_id}" if order_id else ""
        self.logger.info(
            f"ORDER - {status} - {order_type} {side} {quantity} @ {price:.2f} {order_id_str}"
        )
    
    def log_trade(self, side: str, quantity: int, price: float, 
                 pnl: float = None, pnl_pct: float = None):
        """
        Log a trade execution
        
        Args:
            side: Trade side (BUY, SELL)
            quantity: Trade quantity
            price: Execution price
            pnl: Profit/loss
            pnl_pct: Profit/loss percentage
        """
        pnl_str = ""
        if pnl is not None:
            pnl_str = f" - P&L: ${pnl:.2f}"
            if pnl_pct is not None:
                pnl_str += f" ({pnl_pct:.2f}%)"
                
        self.logger.info(f"TRADE - {side} {quantity} @ {price:.2f}{pnl_str}")
    
    def log_position(self, status: str, quantity: int, entry_price: float, 
                    current_price: float = None, pnl: float = None):
        """
        Log position status
        
        Args:
            status: Position status (OPENED, CLOSED, UPDATED)
            quantity: Position quantity
            entry_price: Position entry price
            current_price: Current market price
            pnl: Current P&L
        """
        current_price_str = ""
        if current_price is not None:
            current_price_str = f" - Current: {current_price:.2f}"
            
        pnl_str = ""
        if pnl is not None:
            pnl_str = f" - P&L: ${pnl:.2f} ({pnl/entry_price*100:.2f}%)"
            
        self.logger.info(
            f"POSITION - {status} - Qty: {quantity} - Entry: {entry_price:.2f}{current_price_str}{pnl_str}"
        )
    
    def log_error(self, error_type: str, message: str):
        """
        Log a trading error
        
        Args:
            error_type: Type of error
            message: Error message
        """
        self.logger.error(f"ERROR - {error_type} - {message}")


# Example usage when run directly
if __name__ == "__main__":
    # Test basic logger
    logger = setup_logger(
        name="test_logger",
        log_file="logs/test.log",
        log_level="DEBUG"
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test trade logger
    trade_logger = TradeLogger(
        strategy_name="GannEquity",
        symbol="RELIANCE"
    )
    
    trade_logger.log_signal("LONG", 2500.0, {"buy_above": 2480.0, "targets": [2550.0, 2600.0]})
    trade_logger.log_order("MARKET", "BUY", 10, 2500.0, "123456")
    trade_logger.log_trade("BUY", 10, 2500.0)
    trade_logger.log_position("OPENED", 10, 2500.0)
    
    print("Logging tests completed. Check logs directory for output files.")