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
Virtual Broker for Paper Trading
Simulates order execution and position management without real money
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto

class OrderStatus(Enum):
    """Order status enum"""
    PENDING = auto()
    OPEN = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()

class OrderType(Enum):
    """Order type enum"""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()

class OrderSide(Enum):
    """Order side enum"""
    BUY = auto()
    SELL = auto()

class ProductType(Enum):
    """Product type enum"""
    CNC = auto()  # Cash and Carry (Delivery)
    MIS = auto()  # Margin Intraday Square-off
    NRML = auto()  # Normal (F&O)

class PositionType(Enum):
    """Position type enum"""
    LONG = auto()
    SHORT = auto()

class VirtualBroker:
    """
    Virtual broker for paper trading
    Simulates order execution and position management
    """
    
    def __init__(self, 
                initial_capital: float = 100000.0, 
                commission_rate: float = 0.05,
                slippage: float = 0.01,
                log_level: str = 'INFO'):
        """
        Initialize virtual broker
        
        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate (percentage)
            slippage: Slippage rate (percentage)
            log_level: Logging level
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Setup logging
        self.logger = logging.getLogger("virtual_broker")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Add handlers only if none exist
        if not self.logger.handlers:
            # Create directory for logs if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / "virtual_broker.log")
            file_handler.setLevel(getattr(logging, log_level))
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # Trading state
        self.orders = {}  # order_id -> order details
        self.positions = {}  # symbol -> position details
        self.trades = []  # list of executed trades
        self.equity_curve = []  # list of equity points over time
        self.daily_pnl = {}  # date -> daily P&L
        
        # Broker state
        self.is_market_open = True
        self.market_open_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        self.market_close_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Callbacks
        self.on_order_update = None
        self.on_trade_update = None
        self.on_position_update = None
        
        # Add initial equity point
        self._update_equity_curve()
        
        self.logger.info(f"Virtual broker initialized with {initial_capital:.2f} capital")
        
    def reset(self) -> None:
        """Reset broker state"""
        self.capital = self.initial_capital
        self.orders = {}
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        
        # Add initial equity point
        self._update_equity_curve()
        
        self.logger.info("Virtual broker reset")
        
    def place_order(self, 
                  symbol: str, 
                  side: Union[OrderSide, str], 
                  quantity: int, 
                  order_type: Union[OrderType, str] = OrderType.MARKET,
                  price: Optional[float] = None,
                  stop_price: Optional[float] = None,
                  product_type: Union[ProductType, str] = ProductType.CNC,
                  validity: str = 'DAY',
                  tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type (MARKET/LIMIT/STOP/STOP_LIMIT)
            price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            product_type: Product type (CNC/MIS/NRML)
            validity: Order validity (DAY/IOC)
            tag: Optional tag for identifying order purpose
            
        Returns:
            Order details dictionary
        """
        # Convert string enums to Enum types if needed
        if isinstance(side, str):
            side = OrderSide[side]
        if isinstance(order_type, str):
            order_type = OrderType[order_type]
        if isinstance(product_type, str):
            product_type = ProductType[product_type]
            
        # Validate order parameters
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            error_msg = "Limit price is required for LIMIT and STOP_LIMIT orders"
            self.logger.error(error_msg)
            return {"status": "REJECTED", "reason": error_msg}
            
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            error_msg = "Stop price is required for STOP and STOP_LIMIT orders"
            self.logger.error(error_msg)
            return {"status": "REJECTED", "reason": error_msg}
            
        # Check if market is open for MIS orders
        if product_type == ProductType.MIS and not self.is_market_open:
            current_time = datetime.now().time()
            market_open = self.market_open_time.time()
            market_close = self.market_close_time.time()
            
            if current_time < market_open or current_time > market_close:
                error_msg = "MIS orders can only be placed during market hours"
                self.logger.error(error_msg)
                return {"status": "REJECTED", "reason": error_msg}
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order object
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "stop_price": stop_price,
            "product_type": product_type,
            "validity": validity,
            "status": OrderStatus.PENDING,
            "filled_quantity": 0,
            "average_price": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "trades": [],
            "tag": tag
        }
        
        # Store the order
        self.orders[order_id] = order
        
        # Call order update callback if set
        if self.on_order_update:
            self.on_order_update(order)
            
        self.logger.info(f"Order placed: {order_id} {side.name} {quantity} {symbol}")
        
        return {
            "status": "SUCCESS",
            "order_id": order_id,
            "order": order
        }
        
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Result dictionary
        """
        if order_id not in self.orders:
            error_msg = f"Order {order_id} not found"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        order = self.orders[order_id]
        
        # Check if order can be cancelled
        if order["status"] in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            error_msg = f"Cannot cancel order {order_id} with status {order['status']}"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        # Update order status
        order["status"] = OrderStatus.CANCELLED
        order["updated_at"] = datetime.now()
        
        # Call order update callback if set
        if self.on_order_update:
            self.on_order_update(order)
            
        self.logger.info(f"Order cancelled: {order_id}")
        
        return {"status": "SUCCESS", "order_id": order_id}
        
    def modify_order(self, 
                   order_id: str, 
                   quantity: Optional[int] = None,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            stop_price: New stop price
            
        Returns:
            Result dictionary
        """
        if order_id not in self.orders:
            error_msg = f"Order {order_id} not found"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        order = self.orders[order_id]
        
        # Check if order can be modified
        if order["status"] in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            error_msg = f"Cannot modify order {order_id} with status {order['status']}"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        # Update order details
        if quantity is not None:
            order["quantity"] = quantity
            
        if price is not None:
            order["price"] = price
            
        if stop_price is not None:
            order["stop_price"] = stop_price
            
        order["updated_at"] = datetime.now()
        
        # Call order update callback if set
        if self.on_order_update:
            self.on_order_update(order)
            
        self.logger.info(f"Order modified: {order_id}")
        
        return {"status": "SUCCESS", "order_id": order_id, "order": order}
        
    def process_market_data(self, symbol: str, price_data: Dict[str, float]) -> None:
        """
        Process market data update
        
        Args:
            symbol: Symbol being updated
            price_data: Dictionary with OHLCV data
        """
        # Extract price info
        current_price = price_data.get('ltp', 
                         price_data.get('close', 
                           price_data.get('last_price')))
                           
        if current_price is None:
            self.logger.warning(f"No price found in update for {symbol}")
            return
            
        # Get high and low prices if available
        high_price = price_data.get('high', current_price)
        low_price = price_data.get('low', current_price)
        
        # Process pending orders
        self._process_orders(symbol, current_price, high_price, low_price)
        
        # Update positions
        self._update_positions(symbol, current_price)
        
        # Update equity curve periodically
        self._update_equity_curve()
        
    def simulate_execution(self, 
                         symbol: str, 
                         order_id: str, 
                         execution_price: float, 
                         quantity: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate order execution (manual fill)
        
        Args:
            symbol: Symbol being traded
            order_id: Order ID to execute
            execution_price: Price to execute at
            quantity: Quantity to execute (defaults to full order quantity)
            
        Returns:
            Execution result
        """
        if order_id not in self.orders:
            error_msg = f"Order {order_id} not found"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        order = self.orders[order_id]
        
        # Check if order can be executed
        if order["status"] in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            error_msg = f"Cannot execute order {order_id} with status {order['status']}"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        # Check symbol match
        if order["symbol"] != symbol:
            error_msg = f"Symbol mismatch: order is for {order['symbol']}, got {symbol}"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        # Determine quantity to execute
        if quantity is None:
            quantity = order["quantity"] - order["filled_quantity"]
        else:
            quantity = min(quantity, order["quantity"] - order["filled_quantity"])
            
        if quantity <= 0:
            error_msg = f"Invalid execution quantity: {quantity}"
            self.logger.error(error_msg)
            return {"status": "FAILED", "reason": error_msg}
            
        # Execute the order
        return self._execute_order(order, execution_price, quantity, datetime.now())
    
    def execute_all_market_orders(self, symbol: str, current_price: float) -> None:
        """
        Execute all market orders for a symbol
        
        Args:
            symbol: Symbol to execute orders for
            current_price: Current market price
        """
        for order_id, order in list(self.orders.items()):
            if (order["symbol"] == symbol and 
                order["status"] == OrderStatus.PENDING and 
                order["order_type"] == OrderType.MARKET):
                self._execute_order(order, current_price, order["quantity"], datetime.now())
                
    def _execute_order(self, 
                     order: Dict[str, Any], 
                     execution_price: float, 
                     quantity: int, 
                     execution_time: datetime) -> Dict[str, Any]:
        """
        Execute an order (internal method)
        
        Args:
            order: Order to execute
            execution_price: Price to execute at
            quantity: Quantity to execute
            execution_time: Execution timestamp
            
        Returns:
            Execution result
        """
        # Apply slippage to execution price
        if order["side"] == OrderSide.BUY:
            adjusted_price = execution_price * (1 + self.slippage/100)
        else:
            adjusted_price = execution_price * (1 - self.slippage/100)
            
        adjusted_price = round(adjusted_price, 2)
        
        # Calculate commission
        commission = round(adjusted_price * quantity * self.commission_rate / 100, 2)
        
        # Create trade record
        trade = {
            "order_id": order["order_id"],
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": quantity,
            "price": adjusted_price,
            "commission": commission,
            "execution_time": execution_time,
            "trade_id": str(uuid.uuid4())
        }
        
        # Calculate total cost/proceeds
        if order["side"] == OrderSide.BUY:
            trade_amount = -(adjusted_price * quantity + commission)
        else:
            trade_amount = adjusted_price * quantity - commission
            
        # Update capital
        self.capital += trade_amount
        
        # Update order status
        order["trades"].append(trade)
        order["filled_quantity"] += quantity
        total_executed = sum(t["quantity"] for t in order["trades"])
        total_value = sum(t["quantity"] * t["price"] for t in order["trades"])
        order["average_price"] = total_value / total_executed if total_executed > 0 else 0
        
        if order["filled_quantity"] >= order["quantity"]:
            order["status"] = OrderStatus.FILLED
        else:
            order["status"] = OrderStatus.PARTIALLY_FILLED
            
        order["updated_at"] = execution_time
        
        # Call order update callback if set
        if self.on_order_update:
            self.on_order_update(order)
            
        # Add trade to list
        self.trades.append(trade)
        
        # Call trade update callback if set
        if self.on_trade_update:
            self.on_trade_update(trade)
            
        # Update position
        self._update_position_after_trade(trade)
        
        self.logger.info(f"Executed {quantity} of {order['order_id']} at {adjusted_price}")
        
        return {
            "status": "SUCCESS",
            "order_id": order["order_id"],
            "trade_id": trade["trade_id"],
            "filled_quantity": quantity,
            "remaining_quantity": order["quantity"] - order["filled_quantity"],
            "execution_price": adjusted_price,
            "commission": commission,
            "trade_amount": trade_amount
        }
        
    def _update_position_after_trade(self, trade: Dict[str, Any]) -> None:
        """
        Update position after a trade
        
        Args:
            trade: Trade details
        """
        symbol = trade["symbol"]
        side = trade["side"]
        quantity = trade["quantity"]
        price = trade["price"]
        
        # Get existing position or create new one
        if symbol in self.positions:
            position = self.positions[symbol]
        else:
            position = {
                "symbol": symbol,
                "quantity": 0,
                "average_price": 0,
                "side": None,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "trades": []
            }
            self.positions[symbol] = position
            
        # Update position based on trade side
        if side == OrderSide.BUY:
            if position["quantity"] < 0:
                # Reducing short position
                # Calculate realized P&L
                realized_pnl = (position["average_price"] - price) * min(quantity, abs(position["quantity"]))
                position["realized_pnl"] += realized_pnl
                
                # Update position
                if quantity < abs(position["quantity"]):
                    # Partial cover
                    position["quantity"] += quantity
                else:
                    # Full cover with potential reversal
                    excess = quantity - abs(position["quantity"])
                    position["quantity"] = excess
                    position["average_price"] = price if excess > 0 else 0
                    position["side"] = OrderSide.BUY if excess > 0 else None
            else:
                # Adding to existing long or new position
                total_quantity = position["quantity"] + quantity
                total_value = position["quantity"] * position["average_price"] + quantity * price
                position["average_price"] = total_value / total_quantity if total_quantity > 0 else 0
                position["quantity"] = total_quantity
                position["side"] = OrderSide.BUY
        else:  # SELL
            if position["quantity"] > 0:
                # Reducing long position
                # Calculate realized P&L
                realized_pnl = (price - position["average_price"]) * min(quantity, position["quantity"])
                position["realized_pnl"] += realized_pnl
                
                # Update position
                if quantity < position["quantity"]:
                    # Partial sell
                    position["quantity"] -= quantity
                else:
                    # Full sell with potential reversal
                    excess = quantity - position["quantity"]
                    position["quantity"] = -excess
                    position["average_price"] = price if excess > 0 else 0
                    position["side"] = OrderSide.SELL if excess > 0 else None
            else:
                # Adding to existing short or new position
                total_quantity = position["quantity"] - quantity
                total_value = abs(position["quantity"]) * position["average_price"] + quantity * price
                position["average_price"] = total_value / abs(total_quantity) if total_quantity != 0 else 0
                position["quantity"] = total_quantity
                position["side"] = OrderSide.SELL
                
        # Add trade reference
        position["trades"].append(trade["trade_id"])
        
        # Call position update callback if set
        if self.on_position_update:
            self.on_position_update(position)
            
    def _process_orders(self, 
                      symbol: str, 
                      current_price: float, 
                      high_price: float, 
                      low_price: float) -> None:
        """
        Process orders based on new market data
        
        Args:
            symbol: Symbol to process orders for
            current_price: Current market price
            high_price: High price of current bar
            low_price: Low price of current bar
        """
        # Process all pending orders for this symbol
        for order_id, order in list(self.orders.items()):
            if order["symbol"] != symbol or order["status"] != OrderStatus.PENDING:
                continue
                
            # Check order triggers based on type
            order_type = order["order_type"]
            side = order["side"]
            
            if order_type == OrderType.MARKET:
                # Market orders execute immediately
                self._execute_order(order, current_price, order["quantity"], datetime.now())
                
            elif order_type == OrderType.LIMIT:
                # Limit orders
                if (side == OrderSide.BUY and low_price <= order["price"]) or \
                   (side == OrderSide.SELL and high_price >= order["price"]):
                    # Order triggered
                    self._execute_order(order, order["price"], order["quantity"], datetime.now())
                    
            elif order_type == OrderType.STOP:
                # Stop orders
                if (side == OrderSide.BUY and high_price >= order["stop_price"]) or \
                   (side == OrderSide.SELL and low_price <= order["stop_price"]):
                    # Order triggered
                    self._execute_order(order, current_price, order["quantity"], datetime.now())
                    
            elif order_type == OrderType.STOP_LIMIT:
                # Stop-limit orders
                if (side == OrderSide.BUY and high_price >= order["stop_price"]) or \
                   (side == OrderSide.SELL and low_price <= order["stop_price"]):
                    # Stop triggered, convert to limit order
                    order["order_type"] = OrderType.LIMIT
                    order["updated_at"] = datetime.now()
                    
                    # Call order update callback if set
                    if self.on_order_update:
                        self.on_order_update(order)
                        
    def _update_positions(self, symbol: str, current_price: float) -> None:
        """
        Update position valuations based on current price
        
        Args:
            symbol: Symbol to update positions for
            current_price: Current market price
        """
        # Update only the specific symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Skip empty positions
            if position["quantity"] == 0:
                return
                
            # Calculate unrealized P&L
            if position["quantity"] > 0:
                # Long position
                position["unrealized_pnl"] = (current_price - position["average_price"]) * position["quantity"]
            else:
                # Short position
                position["unrealized_pnl"] = (position["average_price"] - current_price) * abs(position["quantity"])
                
            # Call position update callback if set
            if self.on_position_update:
                self.on_position_update(position)
                
    def _update_equity_curve(self) -> None:
        """Update equity curve with current portfolio value"""
        # Calculate current time
        current_time = datetime.now()
        
        # Check if we already have an update for the current minute
        if self.equity_curve:
            last_update = self.equity_curve[-1]["time"]
            if (current_time - last_update).total_seconds() < 60:
                # Skip update if less than a minute has passed
                return
                
        # Calculate position value
        position_value = 0
        for symbol, position in self.positions.items():
            position_value += position["unrealized_pnl"]
            
        # Calculate total equity
        total_equity = self.capital + position_value
        
        # Calculate daily P&L
        today = current_time.date()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = {
                "starting_equity": total_equity,
                "current_equity": total_equity,
                "realized_pnl": 0,
                "unrealized_pnl": position_value
            }
        else:
            self.daily_pnl[today]["current_equity"] = total_equity
            self.daily_pnl[today]["unrealized_pnl"] = position_value
            self.daily_pnl[today]["realized_pnl"] = sum(
                position["realized_pnl"] for position in self.positions.values()
            )
            
        # Add to equity curve
        self.equity_curve.append({
            "time": current_time,
            "capital": self.capital,
            "position_value": position_value,
            "total_equity": total_equity
        })
        
    def get_orders(self, 
                 symbol: Optional[str] = None, 
                 status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """
        Get orders matching filter criteria
        
        Args:
            symbol: Filter by symbol
            status: Filter by order status
            
        Returns:
            List of matching orders
        """
        result = []
        
        for order in self.orders.values():
            # Apply filters
            if symbol and order["symbol"] != symbol:
                continue
                
            if status and order["status"] != status:
                continue
                
            result.append(order)
            
        return result
        
    def get_positions(self, 
                    symbol: Optional[str] = None, 
                    active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get positions matching filter criteria
        
        Args:
            symbol: Filter by symbol
            active_only: Only return positions with non-zero quantity
            
        Returns:
            List of matching positions
        """
        result = []
        
        for position in self.positions.values():
            # Apply filters
            if symbol and position["symbol"] != symbol:
                continue
                
            if active_only and position["quantity"] == 0:
                continue
                
            result.append(position)
            
        return result
        
    def get_trades(self, 
                 symbol: Optional[str] = None, 
                 from_date: Optional[datetime] = None,
                 to_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get trades matching filter criteria
        
        Args:
            symbol: Filter by symbol
            from_date: Filter trades after this date
            to_date: Filter trades before this date
            
        Returns:
            List of matching trades
        """
        result = []
        
        for trade in self.trades:
            # Apply filters
            if symbol and trade["symbol"] != symbol:
                continue
                
            if from_date and trade["execution_time"] < from_date:
                continue
                
            if to_date and trade["execution_time"] > to_date:
                continue
                
            result.append(trade)
            
        return result
        
    def get_portfolio_value(self) -> Dict[str, float]:
        """
        Get current portfolio value
        
        Returns:
            Dictionary with portfolio value breakdown
        """
        # Calculate position value
        position_value = 0
        for symbol, position in self.positions.items():
            position_value += position["unrealized_pnl"]
            
        # Calculate total equity
        total_equity = self.capital + position_value
        
        return {
            "capital": self.capital,
            "position_value": position_value,
            "total_equity": total_equity,
            "returns_pct": ((total_equity / self.initial_capital) - 1) * 100
        }
        
    def get_equity_curve(self, 
                       from_date: Optional[datetime] = None,
                       to_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get equity curve data
        
        Args:
            from_date: Filter data after this date
            to_date: Filter data before this date
            
        Returns:
            List of equity curve points
        """
        result = []
        
        for point in self.equity_curve:
            # Apply filters
            if from_date and point["time"] < from_date:
                continue
                
            if to_date and point["time"] > to_date:
                continue
                
            result.append(point)
            
        return result
        
    def set_market_hours(self, market_open: datetime, market_close: datetime) -> None:
        """
        Set market trading hours
        
        Args:
            market_open: Market open time
            market_close: Market close time
        """
        self.market_open_time = market_open
        self.market_close_time = market_close
        
    def set_market_status(self, is_open: bool) -> None:
        """
        Set market open/closed status
        
        Args:
            is_open: True if market is open
        """
        self.is_market_open = is_open
        
        # Cancel all day orders if market closes
        if not is_open:
            self._expire_day_orders()
            
    def _expire_day_orders(self) -> None:
        """Expire all day orders when market closes"""
        for order_id, order in list(self.orders.items()):
            if (order["status"] == OrderStatus.PENDING and 
                order["validity"] == 'DAY'):
                order["status"] = OrderStatus.EXPIRED
                order["updated_at"] = datetime.now()
                
                # Call order update callback if set
                if self.on_order_update:
                    self.on_order_update(order)
                    
                self.logger.info(f"Order expired: {order_id}")