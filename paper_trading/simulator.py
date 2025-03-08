# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
Paper Trading Simulator for Gann Trading System
Simulates live trading without real money
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from core.gann.square_of_9 import GannSquareOf9
from core.strategy.base_strategy import BaseStrategy, TimeFrame, SignalType
from api.factory import get_client, get_websocket
from utils.logger import TradeLogger


class PaperTradingSimulator:
    """
    Paper trading simulator for Gann Trading System
    Connects to real market data but simulates order execution
    """
    
    def __init__(self, 
                strategy: BaseStrategy,
                initial_capital: float = 100000.0,
                commission_rate: float = 0.05,
                slippage: float = 0.01,
                use_real_time_data: bool = True):
        """
        Initialize paper trading simulator
        
        Args:
            strategy: Trading strategy to use
            initial_capital: Initial capital
            commission_rate: Commission rate percentage
            slippage: Slippage rate percentage
            use_real_time_data: Whether to use real-time data from API
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.use_real_time_data = use_real_time_data
        
        # Setup logging
        self.logger = TradeLogger(
            strategy_name=f"paper_{strategy.__class__.__name__}",
            symbol=strategy.symbol
        )
        
        # Trading state
        self.positions = {}  # Symbol -> Position details
        self.pending_orders = {}  # Order ID -> Order details
        self.filled_orders = {}  # Order ID -> Order details
        self.trade_history = []  # List of closed trades
        self.equity_curve = []  # List of equity points
        
        # Market data
        self.price_data = pd.DataFrame()  # Historical price data
        self.latest_prices = {}  # Symbol -> Latest price
        self.option_chains = {}  # Date -> Option chain
        
        # API clients
        self.api_client = None
        self.websocket = None
        if use_real_time_data:
            self.api_client = get_client()
            self.websocket = get_websocket()
            
        # Running state
        self.is_running = False
        self.start_time = None
        self.last_process_time = None
        self.processing_interval = 60  # Process market data every 60 seconds
        
    def initialize(self) -> bool:
        """
        Initialize the simulator
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.logger.info("Initializing paper trading simulator")
        
        # Connect to API if using real-time data
        if self.use_real_time_data:
            if not self.api_client:
                self.logger.logger.error("API client not initialized")
                return False
                
            # Connect to API
            if not self.api_client.connect():
                self.logger.logger.error("Failed to connect to API")
                return False
                
            # Register tick callback
            if self.websocket:
                self.websocket.register_tick_callback(self._on_tick)
                if not self.websocket.connect():
                    self.logger.logger.error("Failed to connect to WebSocket")
                    return False
                    
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.pending_orders = {}
        self.filled_orders = {}
        self.trade_history = []
        self.equity_curve = []
        self.latest_prices = {}
        
        # Add initial equity point
        self._update_equity()
        
        return True
        
    def load_historical_data(self, data: pd.DataFrame) -> None:
        """
        Load historical price data
        
        Args:
            data: DataFrame with price data
        """
        self.price_data = data.copy()
        self.logger.logger.info(f"Loaded {len(data)} bars of historical data")
        
    def start(self) -> None:
        """Start the paper trading simulator"""
        if self.is_running:
            self.logger.logger.warning("Simulator is already running")
            return
            
        # Initialize simulator
        if not self.initialize():
            self.logger.logger.error("Failed to initialize simulator")
            return
            
        self.is_running = True
        self.start_time = datetime.now()
        self.last_process_time = self.start_time
        
        self.logger.logger.info(f"Starting paper trading at {self.start_time}")
        
        if self.use_real_time_data:
            # Subscribe to market data
            if self.websocket:
                self.websocket.subscribe_ticks([self.strategy.symbol])
                self.logger.logger.info(f"Subscribed to real-time data for {self.strategy.symbol}")
        else:
            # Use historical data for simulation
            self._simulate_with_historical_data()
            
    def stop(self) -> None:
        """Stop the paper trading simulator"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Close all positions
        self._close_all_positions()
        
        # Unsubscribe from market data
        if self.use_real_time_data and self.websocket:
            self.websocket.unsubscribe_ticks([self.strategy.symbol])
            self.websocket.disconnect()
            
        self.logger.logger.info(f"Stopped paper trading at {datetime.now()}")
        
    def process(self) -> None:
        """Process market data and check for signals"""
        if not self.is_running:
            return
            
        current_time = datetime.now()
        
        # Check if it's time to process
        if (current_time - self.last_process_time).total_seconds() < self.processing_interval:
            return
            
        self.last_process_time = current_time
        
        # Get latest price
        latest_price = self._get_latest_price(self.strategy.symbol)
        if latest_price is None:
            self.logger.logger.warning(f"No price data available for {self.strategy.symbol}")
            return
            
        # Update prices
        self.latest_prices[self.strategy.symbol] = latest_price
        
        # Update equity
        self._update_equity()
        
        # Process pending orders
        self._process_pending_orders()
        
        # Get historical data for strategy
        historical_data = self._get_historical_data_for_strategy()
        
        # Check for signals
        signal = self.strategy.process_market_data(historical_data)
        
        # Handle signal
        if signal != SignalType.NO_SIGNAL:
            self._handle_signal(signal, latest_price, current_time)
            
    def _simulate_with_historical_data(self) -> None:
        """Simulate trading with historical data"""
        if self.price_data.empty:
            self.logger.logger.error("No historical data loaded")
            return
            
        self.logger.logger.info("Starting simulation with historical data")
        
        # Sort by datetime
        self.price_data = self.price_data.sort_values('datetime')
        
        # Set index to datetime
        if 'datetime' in self.price_data.columns:
            self.price_data = self.price_data.set_index('datetime')
            
        # Track the current time in simulation
        current_idx = 1  # Start from second bar to have at least one previous bar
        
        while current_idx < len(self.price_data) and self.is_running:
            # Get current and previous bars
            current_bar = self.price_data.iloc[current_idx]
            prev_bars = self.price_data.iloc[:current_idx]
            
            # Update latest prices
            self.latest_prices[self.strategy.symbol] = current_bar['close']
            
            # Update equity
            self._update_equity()
            
            # Process pending orders
            self._process_pending_orders()
            
            # Check for signals
            signal = self.strategy.process_market_data(prev_bars)
            
            # Handle signal
            if signal != SignalType.NO_SIGNAL:
                self._handle_signal(signal, current_bar['close'], prev_bars.index[-1])
                
            # Move to next bar
            current_idx += 1
            
            # Simulate delay between bars
            time.sleep(0.01)
            
        # Close all positions at the end
        self._close_all_positions()
        
        self.logger.logger.info("Simulation completed")
        
    def _on_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Callback for tick data
        
        Args:
            tick_data: Tick data from WebSocket
        """
        if not self.is_running:
            return
            
        symbol = tick_data.get('symbol')
        price = tick_data.get('ltp')  # Last traded price
        
        if symbol and price:
            self.latest_prices[symbol] = price
            
    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Latest price or None if not available
        """
        # Check if price is in latest_prices
        if symbol in self.latest_prices:
            return self.latest_prices[symbol]
            
        # If using real-time data, fetch from API
        if self.use_real_time_data and self.api_client:
            try:
                quote = self.api_client.get_quote(symbol)
                if quote:
                    price = quote.get('last_price')
                    if price:
                        self.latest_prices[symbol] = price
                        return price
            except Exception as e:
                self.logger.logger.error(f"Error fetching quote: {e}")
                
        # If using historical data, use the last price
        if not self.price_data.empty:
            last_price = self.price_data.iloc[-1]['close']
            self.latest_prices[symbol] = last_price
            return last_price
            
        return None
        
    def _get_historical_data_for_strategy(self) -> pd.DataFrame:
        """
        Get historical data for strategy
        
        Returns:
            DataFrame with historical data
        """
        if self.use_real_time_data and self.api_client:
            # Get data from API
            try:
                # Calculate timeframe in minutes
                timeframe_minutes = {
                    '1minute': 1,
                    '5minute': 5,
                    '15minute': 15,
                    '30minute': 30,
                    '1hour': 60,
                    '1day': 1440
                }
                
                minutes = timeframe_minutes.get(self.strategy.timeframe.value, 1)
                
                # Calculate dates
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # Get 30 days of data
                
                # Get historical data
                data = self.api_client.get_historical_data(
                    stock_code=self.strategy.symbol,
                    exchange_code='NSE',
                    from_date=start_date.strftime("%d-%m-%Y %H:%M:%S"),
                    to_date=end_date.strftime("%d-%m-%Y %H:%M:%S"),
                    interval=self.strategy.timeframe.value
                )
                
                if not data.empty:
                    # Update price data
                    self.price_data = data
                    return data
            except Exception as e:
                self.logger.logger.error(f"Error fetching historical data: {e}")
                
        # Use cached historical data
        return self.price_data
    
    def _get_option_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get option chain data
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            DataFrame with option chain data or None if not available
        """
        current_date = datetime.now().date()
        
        # Check if we have cached data for today
        if current_date in self.option_chains:
            return self.option_chains[current_date]
            
        # If using real-time data, fetch from API
        if self.use_real_time_data and self.api_client:
            try:
                # Get option chain
                option_chain = self.api_client.get_option_chain(symbol)
                
                if not option_chain.empty:
                    # Cache option chain
                    self.option_chains[current_date] = option_chain
                    return option_chain
            except Exception as e:
                self.logger.logger.error(f"Error fetching option chain: {e}")
                
        return None
        
    def _update_equity(self) -> None:
        """Update equity curve"""
        # Calculate position value
        position_value = 0
        for symbol, position in self.positions.items():
            if position['is_option']:
                # For options, use current P&L
                position_value += position['pnl']
            else:
                # For equity, use latest price
                latest_price = self._get_latest_price(symbol)
                if latest_price:
                    position_value += position['quantity'] * latest_price
                    
        # Calculate equity
        equity = self.capital + position_value
        
        # Add to equity curve
        self.equity_curve.append({
            'time': datetime.now(),
            'equity': equity,
            'capital': self.capital,
            'position_value': position_value
        })
        
    def _process_pending_orders(self) -> None:
        """Process pending orders"""
        for order_id, order in list(self.pending_orders.items()):
            # Check if order is still valid
            if order['status'] != 'PENDING':
                continue
                
            # Get latest price
            latest_price = self._get_latest_price(order['symbol'])
            if not latest_price:
                continue
                
            # Check if order can be filled
            filled = False
            
            if order['order_type'] == 'MARKET':
                # Market orders are filled immediately
                filled = True
                fill_price = latest_price
                
            elif order['order_type'] == 'LIMIT':
                # Limit orders are filled if price crosses limit price
                if (order['side'] == 'BUY' and latest_price <= order['price']) or \
                   (order['side'] == 'SELL' and latest_price >= order['price']):
                    filled = True
                    fill_price = order['price']
                    
            elif order['order_type'] == 'STOP':
                # Stop orders are filled if price crosses stop price
                if (order['side'] == 'BUY' and latest_price >= order['stop_price']) or \
                   (order['side'] == 'SELL' and latest_price <= order['stop_price']):
                    filled = True
                    fill_price = latest_price
                    
            # Fill order if conditions met
            if filled:
                self._fill_order(order_id, fill_price)
                
    def _fill_order(self, order_id: str, fill_price: float) -> None:
        """
        Fill an order
        
        Args:
            order_id: Order ID
            fill_price: Price to fill at
        """
        if order_id not in self.pending_orders:
            return
            
        order = self.pending_orders[order_id]
        
        # Apply slippage
        if order['side