# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
Backtesting engine for Gann Trading System
Implements event-driven backtesting for strategy evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import logging
from enum import Enum

from core.strategy.base_strategy import BaseStrategy, TimeFrame, SignalType
from core.strategy.equity_strategy import EquityStrategy
from core.strategy.index_strategy import IndexStrategy


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order sides for backtesting"""
    BUY = "BUY"
    SELL = "SELL"


class Order:
    """Order representation for backtesting"""
    
    def __init__(self, 
                symbol: str, 
                side: OrderSide, 
                quantity: int, 
                order_type: OrderType = OrderType.MARKET,
                limit_price: Optional[float] = None,
                stop_price: Optional[float] = None,
                order_id: Optional[str] = None,
                product_type: str = "CNC"):
        """
        Initialize order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            limit_price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            order_id: Optional order ID (generated if not provided)
            product_type: Product type (CNC, MIS)
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.order_id = order_id or f"order_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.product_type = product_type
        
        self.created_time = datetime.now()
        self.executed_time = None
        self.executed_price = None
        self.status = "PENDING"
        
    def execute(self, price: float, time: datetime) -> None:
        """
        Execute the order at given price and time
        
        Args:
            price: Execution price
            time: Execution time
        """
        self.executed_price = price
        self.executed_time = time
        self.status = "FILLED"
        
    def __str__(self) -> str:
        """String representation of the order"""
        return (f"Order(id={self.order_id}, symbol={self.symbol}, "
                f"side={self.side.value}, qty={self.quantity}, "
                f"type={self.order_type.value}, status={self.status})")


class Position:
    """Position representation for backtesting"""
    
    def __init__(self, symbol: str, side: OrderSide, quantity: int, entry_price: float, entry_time: datetime, 
                product_type: str = "CNC", is_option: bool = False):
        """
        Initialize position
        
        Args:
            symbol: Trading symbol
            side: Long or short
            quantity: Position quantity
            entry_price: Average entry price
            entry_time: Entry time
            product_type: Product type (CNC, MIS)
            is_option: Whether this is an option position
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.product_type = product_type
        self.is_option = is_option
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.is_open = True
        self.partial_exits = []
        
    def update_pnl(self, current_price: float) -> float:
        """
        Update position P&L
        
        Args:
            current_price: Current market price
            
        Returns:
            Current P&L
        """
        if self.side == OrderSide.BUY:
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
            
        return self.pnl
        
    def close(self, exit_price: float, exit_time: datetime) -> float:
        """
        Close the position
        
        Args:
            exit_price: Exit price
            exit_time: Exit time
            
        Returns:
            Realized P&L
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.is_open = False
        
        if self.side == OrderSide.BUY:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
        return self.pnl
        
    def partial_exit(self, quantity: int, exit_price: float, exit_time: datetime) -> float:
        """
        Partially close the position
        
        Args:
            quantity: Quantity to close
            exit_price: Exit price
            exit_time: Exit time
            
        Returns:
            Realized P&L for the partial exit
        """
        if quantity > self.quantity:
            quantity = self.quantity
            
        # Calculate P&L for the exited portion
        if self.side == OrderSide.BUY:
            partial_pnl = (exit_price - self.entry_price) * quantity
        else:
            partial_pnl = (self.entry_price - exit_price) * quantity
            
        # Record the partial exit
        self.partial_exits.append({
            "quantity": quantity,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "pnl": partial_pnl
        })
        
        # Update remaining quantity
        self.quantity -= quantity
        
        # If no quantity left, mark as closed
        if self.quantity <= 0:
            self.is_open = False
            self.exit_price = exit_price
            self.exit_time = exit_time
            
        return partial_pnl
        
    def __str__(self) -> str:
        """String representation of the position"""
        status = "OPEN" if self.is_open else "CLOSED"
        exit_info = f", exit_price={self.exit_price}, exit_time={self.exit_time}" if not self.is_open else ""
        return (f"Position(symbol={self.symbol}, side={self.side.value}, "
                f"qty={self.quantity}, entry_price={self.entry_price}, "
                f"entry_time={self.entry_time}{exit_info}, "
                f"pnl={self.pnl:.2f}, type={'OPTION' if self.is_option else 'EQUITY'}, status={status})")


class BacktestEngine:
    """
    Engine for backtesting trading strategies
    """
    
    def __init__(self, 
                initial_capital: float = 100000.0,
                commission_rate: float = 0.0,
                option_chains: Optional[Dict[datetime, pd.DataFrame]] = None):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate as percentage
            option_chains: Dictionary of option chain data keyed by date
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.option_chains = option_chains
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.orders = []
        self.positions = []
        self.active_positions = {}  # Symbol -> Position
        self.active_option_positions = {}  # Symbol -> Position
        
        # Position details for strategies
        self.position_details = {}
        
        # Logging
        self.logger = logging.getLogger("backtest")
        
    def reset(self) -> None:
        """Reset the backtesting engine to initial state"""
        self.capital = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.orders = []
        self.positions = []
        self.active_positions = {}
        self.active_option_positions = {}
        self.position_details = {}
        self.logger.info(f"Backtest engine reset. Initial capital: {self.initial_capital}")
        
    def run_backtest(self, 
                    strategy: BaseStrategy, 
                    data: pd.DataFrame, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest with given strategy and data
        
        Args:
            strategy: Strategy to test
            data: Historical data with OHLCV and datetime index
            start_date: Start date for backtest (format: YYYY-MM-DD)
            end_date: End date for backtest (format: YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data.set_index('datetime', inplace=True)
            else:
                raise ValueError("Data must have datetime index or 'datetime' column")
                
        # Filter data by date range if provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if data.empty:
            self.logger.error("No data available for the specified date range")
            return {"error": "No data available for the specified date range"}
            
        # Convert data index to datetime if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        self.logger.info(f"Starting backtest with {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Run strategy on each bar
        for i in range(1, len(data)):
            bar = data.iloc[i]
            prev_bars = data.iloc[:i]
            
            # Update equity curve
            self._update_equity_curve(bar.name, bar['close'])
            
            # Process pending orders
            self._process_orders(bar)
            
            # Update open positions
            self._update_positions(bar)
            
            # Check for strategy signals
            signal = strategy.process_market_data(prev_bars)
            
            # Handle signals
            if signal != SignalType.NO_SIGNAL:
                self._handle_signal(strategy, signal, bar, prev_bars)
            
        # Close all open positions at the end of backtest
        self._close_all_positions(data.iloc[-1])
        
        # Calculate and return performance metrics
        return self._calculate_performance()
        
    def _handle_signal(self, 
                      strategy: BaseStrategy, 
                      signal: SignalType, 
                      bar: pd.Series, 
                      prev_bars: pd.DataFrame) -> None:
        """
        Handle strategy signals
        
        Args:
            strategy: Strategy instance
            signal: Signal type
            bar: Current price bar
            prev_bars: Previous price bars
        """
        current_price = bar['close']
        current_time = bar.name
        
        # Get option chain if available
        option_chain = None
        if self.option_chains is not None and current_time in self.option_chains:
            option_chain = self.option_chains[current_time]
        
        if signal == SignalType.LONG:
            if isinstance(strategy, EquityStrategy):
                # Execute equity long strategy
                execution = strategy.execute_long_strategy(
                    current_price=current_price,
                    capital=self.capital,
                    option_chain=option_chain
                )
            elif isinstance(strategy, IndexStrategy):
                # Execute index long strategy
                execution = strategy.execute_long_strategy(
                    current_price=current_price,
                    capital=self.capital,
                    option_chain=option_chain
                )
            else:
                self.logger.error(f"Unsupported strategy type: {type(strategy)}")
                return
                
            if "error" in execution:
                self.logger.error(f"Error executing long strategy: {execution['error']}")
                return
                
            # Store position details
            self.position_details = execution
                
            # Process orders from execution
            for order_details in execution.get("orders", []):
                self._create_order_from_details(order_details, current_time)
                
            self.logger.info(f"LONG signal executed at {current_price:.2f}")
                
        elif signal == SignalType.SHORT:
            if isinstance(strategy, EquityStrategy):
                # Execute equity short strategy
                execution = strategy.execute_short_strategy(
                    current_price=current_price,
                    capital=self.capital,
                    option_chain=option_chain
                )
            elif isinstance(strategy, IndexStrategy):
                # Execute index short strategy
                execution = strategy.execute_short_strategy(
                    current_price=current_price,
                    capital=self.capital,
                    option_chain=option_chain
                )
            else:
                self.logger.error(f"Unsupported strategy type: {type(strategy)}")
                return
                
            if "error" in execution:
                self.logger.error(f"Error executing short strategy: {execution['error']}")
                return
                
            # Store position details
            self.position_details = execution
                
            # Process orders from execution
            for order_details in execution.get("orders", []):
                self._create_order_from_details(order_details, current_time)
                
            self.logger.info(f"SHORT signal executed at {current_price:.2f}")
                
        elif signal == SignalType.EXIT_LONG:
            if isinstance(strategy, EquityStrategy):
                # Execute equity exit long strategy
                execution = strategy.execute_exit_long(
                    current_price=current_price,
                    position_details=self.position_details
                )
            elif isinstance(strategy, IndexStrategy):
                # Execute index exit long strategy
                execution = strategy.execute_exit_long(
                    current_price=current_price,
                    position_details=self.position_details
                )
            else:
                self.logger.error(f"Unsupported strategy type: {type(strategy)}")
                return
                
            if "error" in execution:
                self.logger.error(f"Error executing exit long strategy: {execution['error']}")
                return
                
            # Process orders from execution
            for order_details in execution.get("orders", []):
                self._create_order_from_details(order_details, current_time)
                
            self.logger.info(f"EXIT_LONG signal executed at {current_price:.2f}")
                
        elif signal == SignalType.EXIT_SHORT:
            if isinstance(strategy, EquityStrategy):
                # Execute equity exit short strategy
                execution = strategy.execute_exit_short(
                    current_price=current_price,
                    position_details=self.position_details
                )
            elif isinstance(strategy, IndexStrategy):
                # Execute index exit short strategy
                execution = strategy.execute_exit_short(
                    current_price=current_price,
                    position_details=self.position_details
                )
            else:
                self.logger.error(f"Unsupported strategy type: {type(strategy)}")
                return
                
            if "error" in execution:
                self.logger.error(f"Error executing exit short strategy: {execution['error']}")
                return
                
            # Process orders from execution
            for order_details in execution.get("orders", []):
                self._create_order_from_details(order_details, current_time)
                
            self.logger.info(f"EXIT_SHORT signal executed at {current_price:.2f}")
    
    def _create_order_from_details(self, 
                                     order_details: Dict[str, Any], 
                                     current_time: datetime) -> Order:
            """
            Create order from details
            
            Args:
                order_details: Order details dictionary
                current_time: Current bar time
                
            Returns:
                Created order
            """
            # Extract order details
            symbol = order_details.get("symbol")
            transaction_type = order_details.get("transaction_type")
            quantity = order_details.get("quantity")
            price = order_details.get("price")
            order_type = order_details.get("order_type", "MARKET")
            product_type = order_details.get("product_type", "CNC")
            limit_price = order_details.get("limit_price")
            stop_price = order_details.get("stop_price")
            
            # Convert to enum types
            side = OrderSide.BUY if transaction_type == "BUY" else OrderSide.SELL
            order_type_enum = OrderType[order_type]
            
            # Create and execute order
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type_enum,
                limit_price=limit_price,
                stop_price=stop_price,
                product_type=product_type
            )
            
            # Add to orders list
            self.orders.append(order)
            
            # For market orders, execute immediately
            if order_type_enum == OrderType.MARKET:
                self._execute_order(order, price, current_time)
                
            return order
        
    def _execute_order(self, order: Order, price: float, time: datetime) -> None:
        """
        Execute an order
        
        Args:
            order: Order to execute
            price: Execution price
            time: Execution time
        """
        # Update order status
        order.execute(price, time)
        
        # Calculate commission
        commission = price * order.quantity * self.commission_rate / 100
        
        # Check if this is an option order
        is_option = "CE" in order.symbol or "PE" in order.symbol or order.product_type == "MIS"
        
        # Update capital
        if order.side == OrderSide.BUY:
            cost = price * order.quantity + commission
            self.capital -= cost
            
            # Create or update position
            position_dict = self.active_option_positions if is_option else self.active_positions
            if order.symbol in position_dict:
                position = position_dict[order.symbol]
                # If position is on opposite side, close it first
                if position.side == OrderSide.SELL:
                    pnl = position.close(price, time)
                    self.capital += pnl
                    self.positions.append(position)
                    # Create new position
                    position = Position(
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        quantity=order.quantity,
                        entry_price=price,
                        entry_time=time,
                        product_type=order.product_type,
                        is_option=is_option
                    )
                    position_dict[order.symbol] = position
                else:
                    # Average in to existing position
                    total_quantity = position.quantity + order.quantity
                    avg_price = (position.entry_price * position.quantity + price * order.quantity) / total_quantity
                    position.quantity = total_quantity
                    position.entry_price = avg_price
            else:
                # Create new position
                position = Position(
                    symbol=order.symbol,
                    side=OrderSide.BUY,
                    quantity=order.quantity,
                    entry_price=price,
                    entry_time=time,
                    product_type=order.product_type,
                    is_option=is_option
                )
                position_dict[order.symbol] = position
                
        elif order.side == OrderSide.SELL:
            proceeds = price * order.quantity - commission
            self.capital += proceeds
            
            # Update position
            position_dict = self.active_option_positions if is_option else self.active_positions
            if order.symbol in position_dict:
                position = position_dict[order.symbol]
                # If position is on same side, add to it (for shorts)
                if position.side == OrderSide.SELL:
                    total_quantity = position.quantity + order.quantity
                    avg_price = (position.entry_price * position.quantity + price * order.quantity) / total_quantity
                    position.quantity = total_quantity
                    position.entry_price = avg_price
                # If position is on opposite side, reduce or close it
                else:
                    if order.quantity < position.quantity:
                        # Partial exit
                        pnl = position.partial_exit(order.quantity, price, time)
                        # Record the trade
                        self.trades.append({
                            "symbol": position.symbol,
                            "entry_time": position.entry_time,
                            "exit_time": time,
                            "entry_price": position.entry_price,
                            "exit_price": price,
                            "quantity": order.quantity,
                            "side": "LONG",
                            "type": "OPTION" if position.is_option else "EQUITY",
                            "product_type": position.product_type,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (position.entry_price * order.quantity) * 100) if position.entry_price > 0 else 0,
                            "status": "PARTIAL"
                        })
                    else:
                        # Full close
                        pnl = position.close(price, time)
                        # Remove from active positions
                        del position_dict[order.symbol]
                        # Add to closed positions
                        self.positions.append(position)
                        # Record the trade
                        self.trades.append({
                            "symbol": position.symbol,
                            "entry_time": position.entry_time,
                            "exit_time": time,
                            "entry_price": position.entry_price,
                            "exit_price": price,
                            "quantity": position.quantity,
                            "side": "LONG",
                            "type": "OPTION" if position.is_option else "EQUITY",
                            "product_type": position.product_type,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (position.entry_price * position.quantity) * 100) if position.entry_price > 0 else 0,
                            "status": "CLOSED"
                        })
            else:
                # Create new short position
                position = Position(
                    symbol=order.symbol,
                    side=OrderSide.SELL,
                    quantity=order.quantity,
                    entry_price=price,
                    entry_time=time,
                    product_type=order.product_type,
                    is_option=is_option
                )
                position_dict[order.symbol] = position
                
        # Log execution
        self.logger.info(f"Executed: {order} @ {price} - Capital: {self.capital:.2f}")
        
    def _process_orders(self, bar: pd.Series) -> None:
        """
        Process all pending orders
        
        Args:
            bar: Current price bar
        """
        for order in [o for o in self.orders if o.status == "PENDING"]:
            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and bar['low'] <= order.limit_price) or \
                   (order.side == OrderSide.SELL and bar['high'] >= order.limit_price):
                    self._execute_order(order, order.limit_price, bar.name)
                    
            elif order.order_type == OrderType.STOP:
                if (order.side == OrderSide.BUY and bar['high'] >= order.stop_price) or \
                   (order.side == OrderSide.SELL and bar['low'] <= order.stop_price):
                    execution_price = order.stop_price
                    self._execute_order(order, execution_price, bar.name)
                    
            elif order.order_type == OrderType.STOP_LIMIT:
                # Check if stop price is hit
                if (order.side == OrderSide.BUY and bar['high'] >= order.stop_price) or \
                   (order.side == OrderSide.SELL and bar['low'] <= order.stop_price):
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
                    
    def _update_positions(self, bar: pd.Series) -> None:
        """
        Update all open positions P&L
        
        Args:
            bar: Current price bar
        """
        for symbol, position in list(self.active_positions.items()):
            position.update_pnl(bar['close'])
            
        for symbol, position in list(self.active_option_positions.items()):
            # For options, we need option prices, but we use a simplified model here
            # where option price moves proportionally with the underlying
            position.update_pnl(bar['close'])
            
    def _update_equity_curve(self, time: datetime, price: float) -> None:
        """
        Update equity curve
        
        Args:
            time: Current time
            price: Current price
        """
        # Calculate position value
        position_value = 0
        for symbol, position in self.active_positions.items():
            if position.side == OrderSide.BUY:
                position_value += position.quantity * price
            else:
                # For short positions, subtract value (simplified)
                position_value -= position.quantity * price
                
        # For options, use current P&L
        option_value = 0
        for symbol, position in self.active_option_positions.items():
            option_value += position.pnl
                
        # Calculate equity
        equity = self.capital + position_value + option_value
        
        # Add to equity curve
        self.equity_curve.append({
            "time": time,
            "equity": equity,
            "capital": self.capital,
            "position_value": position_value,
            "option_value": option_value
        })
        
    def _close_all_positions(self, bar: pd.Series) -> None:
        """
        Close all open positions at the end of backtest
        
        Args:
            bar: Last price bar
        """
        # Close equity positions
        for symbol, position in list(self.active_positions.items()):
            if position.side == OrderSide.BUY:
                self._create_order_from_details({
                    "symbol": symbol,
                    "transaction_type": "SELL",
                    "quantity": position.quantity,
                    "price": bar['close'],
                    "product_type": position.product_type
                }, bar.name)
            else:
                self._create_order_from_details({
                    "symbol": symbol,
                    "transaction_type": "BUY",
                    "quantity": position.quantity,
                    "price": bar['close'],
                    "product_type": position.product_type
                }, bar.name)
        
        # Close option positions
        for symbol, position in list(self.active_option_positions.items()):
            if position.side == OrderSide.BUY:
                self._create_order_from_details({
                    "symbol": symbol,
                    "transaction_type": "SELL",
                    "quantity": position.quantity,
                    "price": bar['close'] * 0.1,  # Simplified option pricing
                    "product_type": position.product_type
                }, bar.name)
            else:
                self._create_order_from_details({
                    "symbol": symbol,
                    "transaction_type": "BUY",
                    "quantity": position.quantity,
                    "price": bar['close'] * 0.1,  # Simplified option pricing
                    "product_type": position.product_type
                }, bar.name)
                
    def _calculate_performance(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('time', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate cumulative returns
        equity_df['cum_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate drawdowns
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Calculate key metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        max_drawdown = equity_df['drawdown'].min()
        win_trades = len([t for t in self.trades if t['pnl'] > 0])
        lose_trades = len([t for t in self.trades if t['pnl'] <= 0])
        total_trades = len(self.trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate avg trade
        avg_trade = sum([t['pnl'] for t in self.trades]) / total_trades if total_trades > 0 else 0
        avg_win = sum([t['pnl'] for t in self.trades if t['pnl'] > 0]) / win_trades if win_trades > 0 else 0
        avg_loss = sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]) / lose_trades if lose_trades > 0 else 0
        
        # Calculate Sharpe ratio (assuming 252 trading days per year)
        if len(equity_df) > 1:
            daily_returns = equity_df['returns'].dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Return metrics
        return {
            "initial_capital": self.initial_capital,
            "final_capital": equity_df['equity'].iloc[-1],
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": total_trades,
            "win_trades": win_trades,
            "lose_trades": lose_trades,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": equity_df,
            "trades": self.trades
        }