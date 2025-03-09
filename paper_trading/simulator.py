# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

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
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from core.gann.square_of_9 import GannSquareOf9
from core.strategy.base_strategy import BaseStrategy, TimeFrame, SignalType
from api.factory import get_client, get_websocket
from utils.logger import TradeLogger
from paper_trading.virtual_broker import VirtualBroker, OrderSide, OrderType, OrderStatus, ProductType

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
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.use_real_time_data = use_real_time_data
        
        # Setup logging
        self.logger = TradeLogger(
            strategy_name=f"paper_{strategy.__class__.__name__}",
            symbol=strategy.symbol
        )
        
        # Initialize virtual broker
        self.broker = VirtualBroker(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage
        )
        
        # Register broker callbacks
        self.broker.on_order_update = self._on_order_update
        self.broker.on_trade_update = self._on_trade_update
        self.broker.on_position_update = self._on_position_update
        
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
        self.simulation_speed = 1.0  # Simulation speed multiplier (for backtesting mode)
        
        # Thread for backtesting simulation
        self.simulation_thread = None
        self.stop_simulation = threading.Event()
        
        # Position management
        self.position_details = {}  # Current position details
        self.active_signal = SignalType.NO_SIGNAL
        
    def initialize(self) -> bool:
        """
        Initialize the simulator
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.logger.info("Initializing paper trading simulator")
        
        # Reset broker
        self.broker.reset()
        
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
        self.position_details = {}
        self.active_signal = SignalType.NO_SIGNAL
        self.latest_prices = {}
        
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
            self.stop_simulation.clear()
            self.simulation_thread = threading.Thread(target=self._simulate_with_historical_data)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def stop(self) -> None:
        """Stop the paper trading simulator"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop simulation thread
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.stop_simulation.set()
            self.simulation_thread.join(timeout=5.0)
            
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
        
        # Process pending orders
        self.broker.execute_all_market_orders(self.strategy.symbol, latest_price)
        
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
            data = self.price_data.set_index('datetime')
        else:
            data = self.price_data.copy()
            
        # Track the current time in simulation
        current_idx = 1  # Start from second bar to have at least one previous bar
        
        while current_idx < len(data) and self.is_running and not self.stop_simulation.is_set():
            # Get current and previous bars
            current_bar = data.iloc[current_idx]
            prev_bars = data.iloc[:current_idx]
            
            # Update latest prices
            current_price = current_bar['close']
            self.latest_prices[self.strategy.symbol] = current_price
            
            # Simulate market data update
            self.broker.process_market_data(self.strategy.symbol, {
                'ltp': current_price,
                'high': current_bar['high'],
                'low': current_bar['low'],
                'open': current_bar['open'],
                'close': current_bar['close'],
                'volume': current_bar.get('volume', 0)
            })
            
            # Check for signals
            signal = self.strategy.process_market_data(prev_bars)
            
            # Handle signal
            if signal != SignalType.NO_SIGNAL:
                simulated_time = prev_bars.index[-1]
                if isinstance(simulated_time, pd.Timestamp):
                    simulated_time = simulated_time.to_pydatetime()
                self._handle_signal(signal, current_price, simulated_time)
                
            # Move to next bar
            current_idx += 1
            
            # Simulate delay between bars
            delay = 0.1 / self.simulation_speed  # Adjust based on simulation speed
            time.sleep(delay)
            
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
            
            # Process market data
            self.broker.process_market_data(symbol, tick_data)
            
            # Check if it's time to process strategy
            current_time = datetime.now()
            if (current_time - self.last_process_time).total_seconds() >= self.processing_interval:
                self.process()
            
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
            last_row = self.price_data.iloc[-1]
            last_price = last_row['close']
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
                    TimeFrame.MINUTE_1: 1,
                    TimeFrame.MINUTE_3: 3,
                    TimeFrame.MINUTE_5: 5,
                    TimeFrame.MINUTE_15: 15,
                    TimeFrame.MINUTE_30: 30,
                    TimeFrame.HOUR_1: 60,
                    TimeFrame.HOUR_2: 120,
                    TimeFrame.HOUR_4: 240,
                    TimeFrame.DAY_1: 1440,
                    TimeFrame.WEEK_1: 10080
                }
                
                minutes = timeframe_minutes.get(self.strategy.timeframe, 1)
                
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
        
    def _handle_signal(self, 
                      signal: SignalType, 
                      current_price: float, 
                      timestamp: datetime) -> None:
        """
        Handle strategy signal
        
        Args:
            signal: Signal type
            current_price: Current market price
            timestamp: Signal timestamp
        """
        self.logger.log_signal(signal.name, current_price)
        
        # Update active signal
        if signal != SignalType.NO_SIGNAL:
            self.active_signal = signal
        
        # Get option chain if needed
        option_chain = self._get_option_chain(self.strategy.symbol)
        
        if signal == SignalType.LONG:
            # Execute long strategy
            execution = self.strategy.execute_long_strategy(
                current_price=current_price,
                capital=self.broker.capital,
                option_chain=option_chain
            )
            
            if "error" in execution:
                self.logger.log_error("STRATEGY", f"Error executing long strategy: {execution['error']}")
                return
                
            # Store position details
            self.position_details = execution
                
            # Place orders
            for order_details in execution.get("orders", []):
                self._place_order_from_details(order_details)
                
        elif signal == SignalType.SHORT:
            # Execute short strategy
            execution = self.strategy.execute_short_strategy(
                current_price=current_price,
                capital=self.broker.capital,
                option_chain=option_chain
            )
            
            if "error" in execution:
                self.logger.log_error("STRATEGY", f"Error executing short strategy: {execution['error']}")
                return
                
            # Store position details
            self.position_details = execution
                
            # Place orders
            for order_details in execution.get("orders", []):
                self._place_order_from_details(order_details)
                
        elif signal == SignalType.EXIT_LONG:
            # Execute exit long strategy
            execution = self.strategy.execute_exit_long(
                current_price=current_price,
                position_details=self.position_details
            )
            
            if "error" in execution:
                self.logger.log_error("STRATEGY", f"Error executing exit long strategy: {execution['error']}")
                return
                
            # Place orders
            for order_details in execution.get("orders", []):
                self._place_order_from_details(order_details)
                
            # Reset position details
            self.position_details = {}
            self.active_signal = SignalType.NO_SIGNAL
                
        elif signal == SignalType.EXIT_SHORT:
            # Execute exit short strategy
            execution = self.strategy.execute_exit_short(
                current_price=current_price,
                position_details=self.position_details
            )
            
            if "error" in execution:
                self.logger.log_error("STRATEGY", f"Error executing exit short strategy: {execution['error']}")
                return
                
            # Place orders
            for order_details in execution.get("orders", []):
                self._place_order_from_details(order_details)
                
            # Reset position details
            self.position_details = {}
            self.active_signal = SignalType.NO_SIGNAL
    
    def _place_order_from_details(self, order_details: Dict[str, Any]) -> str:
        """
        Place order from strategy order details
        
        Args:
            order_details: Order details from strategy execution
            
        Returns:
            Order ID
        """
        # Extract order details
        symbol = order_details.get("symbol")
        transaction_type = order_details.get("transaction_type")
        quantity = order_details.get("quantity")
        price = order_details.get("price")
        order_type = order_details.get("order_type", "MARKET")
        product_type = order_details.get("product_type", "CNC")
        
        # Map to broker enums
        if transaction_type == "BUY":
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
            
        # Map order type
        if order_type == "MARKET":
            broker_order_type = OrderType.MARKET
        elif order_type == "LIMIT":
            broker_order_type = OrderType.LIMIT
        elif order_type == "STOP":
            broker_order_type = OrderType.STOP
        elif order_type == "STOP_LIMIT":
            broker_order_type = OrderType.STOP_LIMIT
        else:
            broker_order_type = OrderType.MARKET
            
        # Map product type
        if product_type == "CNC":
            broker_product_type = ProductType.CNC
        elif product_type == "MIS":
            broker_product_type = ProductType.MIS
        else:
            broker_product_type = ProductType.NRML
            
        # Place order
        result = self.broker.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=broker_order_type,
            price=price,
            product_type=broker_product_type,
            tag=f"Signal: {self.active_signal.name}"
        )
        
        if result.get("status") == "SUCCESS":
            order_id = result.get("order_id")
            self.logger.log_order(
                order_type=broker_order_type.name,
                side=side.name,
                quantity=quantity,
                price=price,
                order_id=order_id,
                status="PLACED"
            )
            return order_id
        else:
            self.logger.log_error("ORDER", f"Failed to place order: {result.get('reason')}")
            return ""
        
    def _on_order_update(self, order: Dict[str, Any]) -> None:
        """
        Callback for order updates
        
        Args:
            order: Updated order details
        """
        self.logger.log_order(
            order_type=order["order_type"].name,
            side=order["side"].name,
            quantity=order["quantity"],
            price=order["price"] if order["price"] else 0.0,
            order_id=order["order_id"],
            status=order["status"].name
        )
        
    def _on_trade_update(self, trade: Dict[str, Any]) -> None:
        """
        Callback for trade updates
        
        Args:
            trade: Trade details
        """
        self.logger.log_trade(
            side=trade["side"].name,
            quantity=trade["quantity"],
            price=trade["price"],
            pnl=None  # P&L calculated at position level
        )
        
    def _on_position_update(self, position: Dict[str, Any]) -> None:
        """
        Callback for position updates
        
        Args:
            position: Updated position details
        """
        self.logger.log_position(
            status="UPDATED",
            quantity=position["quantity"],
            entry_price=position["average_price"],
            current_price=self._get_latest_price(position["symbol"]),
            pnl=position["unrealized_pnl"]
        )
        
    def _close_all_positions(self) -> None:
        """Close all open positions"""
        # Get all active positions
        positions = self.broker.get_positions(active_only=True)
        
        for position in positions:
            symbol = position["symbol"]
            quantity = abs(position["quantity"])
            
            if quantity == 0:
                continue
                
            current_price = self._get_latest_price(symbol)
            if not current_price:
                self.logger.log_error("CLOSE", f"Cannot close position for {symbol}: No price available")
                continue
                
            # Place order to close position
            if position["quantity"] > 0:
                # Close long position
                self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    tag="Close All Positions"
                )
            else:
                # Close short position
                self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    tag="Close All Positions"
                )
                
        # Reset position details
        self.position_details = {}
        self.active_signal = SignalType.NO_SIGNAL
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        portfolio_value = self.broker.get_portfolio_value()
        equity_curve = self.broker.get_equity_curve()
        trades = self.broker.get_trades()
        
        # Calculate metrics
        total_trades = len(trades)
        
        if total_trades == 0:
            return {
                "initial_capital": self.initial_capital,
                "current_equity": portfolio_value["total_equity"],
                "return_pct": portfolio_value["returns_pct"],
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "max_drawdown_pct": 0
            }
            
        # Calculate winning/losing trades
        winning_trades = []
        losing_trades = []
        
        for trade in trades:
            # Calculate P&L
            if trade["side"] == OrderSide.BUY:
                # Calculate when closing a long position
                for order in self.broker.orders.values():
                    if (order["symbol"] == trade["symbol"] and 
                        order["side"] == OrderSide.SELL and 
                        order["status"] == OrderStatus.FILLED):
                        # Found matching sell order
                        pnl = (order["average_price"] - trade["price"]) * trade["quantity"]
                        if pnl > 0:
                            winning_trades.append(pnl)
                        else:
                            losing_trades.append(pnl)
                        break
            else:
                # Calculate when closing a short position
                for order in self.broker.orders.values():
                    if (order["symbol"] == trade["symbol"] and 
                        order["side"] == OrderSide.BUY and 
                        order["status"] == OrderStatus.FILLED):
                        # Found matching buy order
                        pnl = (trade["price"] - order["average_price"]) * trade["quantity"]
                        if pnl > 0:
                            winning_trades.append(pnl)
                        else:
                            losing_trades.append(pnl)
                        break
                        
        # Calculate metrics
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        win_rate = num_winners / total_trades if total_trades > 0 else 0
        
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = sum(winning_trades) / num_winners if num_winners > 0 else 0
        average_loss = sum(losing_trades) / num_losers if num_losers > 0 else 0
        
        # Calculate drawdown
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = self.initial_capital
        
        for point in equity_curve:
            equity = point["total_equity"]
            if equity > peak:
                peak = equity
            
            drawdown = peak - equity
            drawdown_pct = drawdown / peak * 100
            
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
                
        return {
            "initial_capital": self.initial_capital,
            "current_equity": portfolio_value["total_equity"],
            "return_pct": portfolio_value["returns_pct"],
            "total_trades": total_trades,
            "winning_trades": num_winners,
            "losing_trades": num_losers,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown": max_drawdown
        }
        
    def set_simulation_speed(self, speed: float) -> None:
        """
        Set simulation speed for backtesting mode
        
        Args:
            speed: Speed multiplier (1.0 = realtime, >1.0 = faster, <1.0 = slower)
        """
        if speed <= 0:
            self.logger.logger.warning("Simulation speed must be positive, setting to 1.0")
            self.simulation_speed = 1.0
        else:
            self.simulation_speed = speed
            self.logger.logger.info(f"Set simulation speed to {speed}x")
            
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account and trading summary
        
        Returns:
            Dictionary with account summary
        """
        portfolio_value = self.broker.get_portfolio_value()
        positions = self.broker.get_positions(active_only=True)
        trades = self.broker.get_trades()
        orders = self.broker.get_orders()
        
        # Organize by symbol
        positions_by_symbol = {}
        for position in positions:
            symbol = position["symbol"]
            positions_by_symbol[symbol] = position
            
        # Count orders by status
        order_stats = {}
        for order in orders:
            status = order["status"].name
            if status not in order_stats:
                order_stats[status] = 0
            order_stats[status] += 1
            
        return {
            "capital": portfolio_value["capital"],
            "position_value": portfolio_value["position_value"],
            "total_equity": portfolio_value["total_equity"],
            "returns_pct": portfolio_value["returns_pct"],
            "active_positions": len(positions),
            "positions_by_symbol": positions_by_symbol,
            "total_trades": len(trades),
            "orders_by_status": order_stats,
            "active_signal": self.active_signal.name
        }