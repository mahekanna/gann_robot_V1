# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
WebSocket client for ICICI Direct Breeze API
Handles real-time data streaming and subscription management
"""

import logging
import threading
import time
import json
import queue
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from .auth import BreezeAuth

class WebSocketClient:
    """
    WebSocket client for ICICI Direct Breeze API
    """
    
    def __init__(self, auth: Optional[BreezeAuth] = None, log_level: int = logging.INFO):
        """
        Initialize WebSocket client
        
        Args:
            auth: Authentication handler, if None a new one will be created
            log_level: Logging level
        """
        self.auth = auth if auth else BreezeAuth(log_level=log_level)
        self.breeze = None
        self.ws_connected = False
        
        # Setup logging
        self.logger = logging.getLogger("breeze_websocket")
        self.logger.setLevel(log_level)
        
        # Add handlers only if none exist
        if not self.logger.handlers:
            # Create directory for logs if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / "breeze_websocket.log")
            file_handler.setLevel(log_level)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        # Callbacks
        self.tick_callbacks = []
        self.index_callbacks = []
        self.order_update_callbacks = []
        self.error_callbacks = []
        self.connection_callbacks = []
        
        # Subscription tracking
        self.subscribed_symbols = set()
        self.subscribed_indices = set()
        self.subscribed_orders = False
        
        # Heartbeat and reconnection
        self.heartbeat_interval = 30  # seconds
        self.reconnect_attempts = 3
        self.heartbeat_thread = None
        self.stop_heartbeat = threading.Event()
        
    def connect(self) -> bool:
        """
        Connect to WebSocket
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.ws_connected:
            self.logger.info("WebSocket already connected")
            return True
            
        # Reset stop event if it was set previously
        self.stop_heartbeat.clear()
            
        try:
            # Get authenticated client
            self.breeze = self.auth.get_breeze_client()
            if not self.breeze:
                self.logger.error("Failed to get authenticated client")
                return False
                
            # Setup callbacks
            self._setup_callbacks()
            
            # Connect to WebSocket
            self.breeze.ws_connect()
            self.ws_connected = True
            self.logger.info("Connected to WebSocket")
            
            # Start heartbeat thread
            self._start_heartbeat()
            
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to WebSocket: {e}")
            return False
            
    def disconnect(self) -> bool:
        """
        Disconnect from WebSocket
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.info("WebSocket not connected")
            return True
            
        # Stop heartbeat thread
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.stop_heartbeat.set()
            self.heartbeat_thread.join(timeout=1.0)
            
        try:
            # Disconnect from WebSocket
            self.breeze.ws_disconnect()
            self.ws_connected = False
            self.logger.info("Disconnected from WebSocket")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from WebSocket: {e}")
            return False
    
    def reconnect(self) -> bool:
        """
        Reconnect to WebSocket
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.logger.info("Attempting to reconnect to WebSocket...")
        
        # Disconnect first
        self.disconnect()
        
        # Try to reconnect with multiple attempts
        for attempt in range(1, self.reconnect_attempts + 1):
            self.logger.info(f"Reconnection attempt {attempt}/{self.reconnect_attempts}")
            if self.connect():
                # Resubscribe to previous subscriptions
                self._resubscribe()
                return True
            
            # Wait before next attempt
            if attempt < self.reconnect_attempts:
                time.sleep(2 * attempt)  # Exponential backoff
                
        self.logger.error("Failed to reconnect to WebSocket after multiple attempts")
        return False
    
    def _resubscribe(self) -> None:
        """Resubscribe to previously subscribed tickers and indices"""
        # Resubscribe to symbols
        if self.subscribed_symbols:
            symbols_by_exchange = {}
            for token in self.subscribed_symbols:
                if '|' in token:
                    exchange, symbol = token.split('|', 1)
                    if exchange not in symbols_by_exchange:
                        symbols_by_exchange[exchange] = []
                    symbols_by_exchange[exchange].append(symbol)
            
            for exchange, symbols in symbols_by_exchange.items():
                self.subscribe_ticks(symbols, exchange)
                
        # Resubscribe to indices
        if self.subscribed_indices:
            indices_by_exchange = {}
            for token in self.subscribed_indices:
                if '|' in token:
                    exchange, index = token.split('|', 1)
                    if exchange not in indices_by_exchange:
                        indices_by_exchange[exchange] = []
                    indices_by_exchange[exchange].append(index)
            
            for exchange, indices in indices_by_exchange.items():
                self.subscribe_indices(indices, exchange)
                
        # Resubscribe to order updates
        if self.subscribed_orders:
            self.subscribe_orders()
            
    def _start_heartbeat(self) -> None:
        """Start heartbeat thread to monitor and maintain connection"""
        self.stop_heartbeat.clear()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop to maintain connection and reconnect if needed"""
        while not self.stop_heartbeat.is_set():
            # Sleep for a while
            for _ in range(self.heartbeat_interval * 10):  # Check the flag more frequently
                if self.stop_heartbeat.is_set():
                    return
                time.sleep(0.1)
                
            # Check connection status
            if not self._check_connection():
                # Try to reconnect
                self.reconnect()
                
    def _check_connection(self) -> bool:
        """
        Check if the WebSocket connection is still alive
        
        Returns:
            True if connected, False otherwise
        """
        if not self.ws_connected:
            return False
            
        # Try to ping websocket or get a simple API response
        try:
            # Simple test API call if possible
            if hasattr(self.breeze, 'get_limits'):
                self.breeze.get_limits()
            return True
        except Exception as e:
            self.logger.warning(f"Connection check failed: {e}")
            return False
            
    def subscribe_ticks(self, symbols: List[str], exchange_code: str = 'NSE') -> bool:
        """
        Subscribe to real-time ticks for symbols
        
        Args:
            symbols: List of symbols
            exchange_code: Exchange code (NSE, BSE, NFO)
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        try:
            # Create stock tokens
            stock_tokens = [f"{exchange_code}|{symbol}" for symbol in symbols]
            
            # Subscribe to feeds
            self.breeze.subscribe_feeds(stock_tokens)
            
            # Add to subscribed symbols
            for token in stock_tokens:
                self.subscribed_symbols.add(token)
                
            self.logger.info(f"Subscribed to {len(stock_tokens)} symbols: {', '.join(symbols)}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to ticks: {e}")
            return False
            
    def unsubscribe_ticks(self, symbols: List[str], exchange_code: str = 'NSE') -> bool:
        """
        Unsubscribe from real-time ticks for symbols
        
        Args:
            symbols: List of symbols
            exchange_code: Exchange code (NSE, BSE, NFO)
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        try:
            # Create stock tokens
            stock_tokens = [f"{exchange_code}|{symbol}" for symbol in symbols]
            
            # Unsubscribe from feeds
            self.breeze.unsubscribe_feeds(stock_tokens)
            
            # Remove from subscribed symbols
            for token in stock_tokens:
                self.subscribed_symbols.discard(token)
                
            self.logger.info(f"Unsubscribed from {len(stock_tokens)} symbols")
            return True
        except Exception as e:
            self.logger.error(f"Error unsubscribing from ticks: {e}")
            return False
    
    def subscribe_indices(self, indices: List[str], exchange_code: str = 'NSE') -> bool:
        """
        Subscribe to real-time index data
        
        Args:
            indices: List of index symbols
            exchange_code: Exchange code (NSE, BSE)
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        try:
            # Create index tokens
            index_tokens = [f"{exchange_code}|{index}" for index in indices]
            
            # Subscribe to index feeds
            self.breeze.subscribe_feeds(index_tokens, feed_type='indices')
            
            # Add to subscribed indices
            for token in index_tokens:
                self.subscribed_indices.add(token)
                
            self.logger.info(f"Subscribed to {len(index_tokens)} indices: {', '.join(indices)}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to indices: {e}")
            return False
            
    def unsubscribe_indices(self, indices: List[str], exchange_code: str = 'NSE') -> bool:
        """
        Unsubscribe from real-time index data
        
        Args:
            indices: List of index symbols
            exchange_code: Exchange code (NSE, BSE)
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        try:
            # Create index tokens
            index_tokens = [f"{exchange_code}|{index}" for index in indices]
            
            # Unsubscribe from index feeds
            self.breeze.unsubscribe_feeds(index_tokens, feed_type='indices')
            
            # Remove from subscribed indices
            for token in index_tokens:
                self.subscribed_indices.discard(token)
                
            self.logger.info(f"Unsubscribed from {len(index_tokens)} indices")
            return True
        except Exception as e:
            self.logger.error(f"Error unsubscribing from indices: {e}")
            return False
            
    def subscribe_orders(self) -> bool:
        """
        Subscribe to order updates
        
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        if self.subscribed_orders:
            self.logger.info("Already subscribed to order updates")
            return True
            
        try:
            # Subscribe to order updates
            self.breeze.subscribe_orderupdate()
            self.subscribed_orders = True
            self.logger.info("Subscribed to order updates")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to order updates: {e}")
            return False
            
    def unsubscribe_orders(self) -> bool:
        """
        Unsubscribe from order updates
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.ws_connected:
            self.logger.error("WebSocket not connected")
            return False
            
        if not self.subscribed_orders:
            self.logger.info("Not subscribed to order updates")
            return True
            
        try:
            # Unsubscribe from order updates
            self.breeze.unsubscribe_orderupdate()
            self.subscribed_orders = False
            self.logger.info("Unsubscribed from order updates")
            return True
        except Exception as e:
            self.logger.error(f"Error unsubscribing from order updates: {e}")
            return False
            
    def register_tick_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Register callback for tick data
        
        Args:
            callback: Callback function that receives tick data
        """
        if callback not in self.tick_callbacks:
            self.tick_callbacks.append(callback)
            self.logger.info(f"Registered tick callback: {callback.__name__}")
            
    def register_index_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Register callback for index data
        
        Args:
            callback: Callback function that receives index data
        """
        if callback not in self.index_callbacks:
            self.index_callbacks.append(callback)
            self.logger.info(f"Registered index callback: {callback.__name__}")
            
    def register_order_update_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Register callback for order updates
        
        Args:
            callback: Callback function that receives order update data
        """
        if callback not in self.order_update_callbacks:
            self.order_update_callbacks.append(callback)
            self.logger.info(f"Registered order update callback: {callback.__name__}")
            
    def register_error_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register callback for error notifications
        
        Args:
            callback: Callback function that receives error messages
        """
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            self.logger.info(f"Registered error callback: {callback.__name__}")
            
    def register_connection_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Register callback for connection status changes
        
        Args:
            callback: Callback function that receives connection status (bool)
        """
        if callback not in self.connection_callbacks:
            self.connection_callbacks.append(callback)
            self.logger.info(f"Registered connection callback: {callback.__name__}")
            
    def _setup_callbacks(self) -> None:
        """Setup callbacks for the WebSocket connection"""
        if not self.breeze:
            return
            
        def on_ticks(ticks):
            """Callback for tick data"""
            for tick in ticks:
                # Check if it's an index update
                is_index = False
                if 'symbol' in tick:
                    token = tick.get('instrument_token', '')
                    is_index = token.startswith('NSE_INDEX') or token.startswith('BSE_INDEX')
                
                # Route to appropriate callbacks
                if is_index:
                    for callback in self.index_callbacks:
                        try:
                            callback(tick)
                        except Exception as e:
                            self.logger.error(f"Error in index callback: {e}")
                else:
                    for callback in self.tick_callbacks:
                        try:
                            callback(tick)
                        except Exception as e:
                            self.logger.error(f"Error in tick callback: {e}")
                        
        def on_connect():
            """Callback for connection"""
            self.ws_connected = True
            self.logger.info("WebSocket connected")
            for callback in self.connection_callbacks:
                try:
                    callback(True)
                except Exception as e:
                    self.logger.error(f"Error in connection callback: {e}")
                    
        def on_close():
            """Callback for disconnection"""
            self.ws_connected = False
            self.logger.info("WebSocket disconnected")
            for callback in self.connection_callbacks:
                try:
                    callback(False)
                except Exception as e:
                    self.logger.error(f"Error in connection callback: {e}")
                    
        def on_error(error):
            """Callback for error"""
            self.logger.error(f"WebSocket error: {error}")
            for callback in self.error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")
                    
        def on_orderupdate(order_update):
            """Callback for order update"""
            for callback in self.order_update_callbacks:
                try:
                    callback(order_update)
                except Exception as e:
                    self.logger.error(f"Error in order update callback: {e}")
                    
        # Set callbacks
        self.breeze.on_ticks = on_ticks
        self.breeze.on_connect = on_connect
        self.breeze.on_close = on_close
        self.breeze.on_error = on_error
        self.breeze.on_orderupdate = on_orderupdate