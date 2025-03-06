# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:37:03 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Demo script for ICICI Direct WebSocket integration
"""

import argparse
import logging
import time
import json
import signal
import sys
from typing import Dict, List, Any

from api.factory import get_websocket_instance

# Global flag for signal handling
running = True

def handle_signal(sig, frame):
    """Handle SIGINT (Ctrl+C)"""
    global running
    print("\nShutting down...")
    running = False

def on_tick(tick_data: Dict[str, Any]) -> None:
    """
    Callback for tick data
    
    Args:
        tick_data: Tick data
    """
    try:
        if 'symbol' in tick_data and 'ltp' in tick_data:
            print(f"[TICK] {tick_data['symbol']}: {tick_data['ltp']} ({tick_data.get('datetime', 'N/A')})")
    except Exception as e:
        print(f"Error processing tick: {e}")

def on_index(index_data: Dict[str, Any]) -> None:
    """
    Callback for index data
    
    Args:
        index_data: Index data
    """
    try:
        if 'symbol' in index_data and 'ltp' in index_data:
            print(f"[INDEX] {index_data['symbol']}: {index_data['ltp']} ({index_data.get('datetime', 'N/A')})")
    except Exception as e:
        print(f"Error processing index: {e}")

def on_order_update(order_data: Dict[str, Any]) -> None:
    """
    Callback for order updates
    
    Args:
        order_data: Order update data
    """
    try:
        print(f"[ORDER UPDATE] {json.dumps(order_data, indent=2)}")
    except Exception as e:
        print(f"Error processing order update: {e}")

def on_error(error_msg: str) -> None:
    """
    Callback for error messages
    
    Args:
        error_msg: Error message
    """
    print(f"[ERROR] {error_msg}")

def on_connection(status: bool) -> None:
    """
    Callback for connection status changes
    
    Args:
        status: Connection status
    """
    print(f"[CONNECTION] {'Connected' if status else 'Disconnected'}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='ICICI Direct WebSocket Demo')
    parser.add_argument('--symbols', type=str, nargs='+', default=['RELIANCE', 'INFY', 'TCS'], 
                      help='Symbols to subscribe to')
    parser.add_argument('--indices', type=str, nargs='+', default=['NIFTY', 'BANKNIFTY'], 
                      help='Indices to subscribe to')
    parser.add_argument('--exchange', type=str, default='NSE', help='Exchange (NSE, BSE, NFO)')
    parser.add_argument('--orders', action='store_true', help='Subscribe to order updates')
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, handle_signal)
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get WebSocket client
    ws = get_websocket_instance(log_level=log_level)
    
    # Register callbacks
    ws.register_tick_callback(on_tick)
    ws.register_index_callback(on_index)
    ws.register_order_update_callback(on_order_update)
    ws.register_error_callback(on_error)
    ws.register_connection_callback(on_connection)
    
    # Connect to WebSocket
    if not ws.connect():
        print("Failed to connect to WebSocket. Exiting.")
        return
    
    # Subscribe to feeds
    if args.symbols:
        print(f"Subscribing to symbols: {args.symbols}")
        ws.subscribe_ticks(args.symbols, args.exchange)
    
    if args.indices:
        print(f"Subscribing to indices: {args.indices}")
        ws.subscribe_indices(args.indices, args.exchange)
    
    if args.orders:
        print("Subscribing to order updates")
        ws.subscribe_orders()
    
    # Main loop
    try:
        print("Streaming data... Press Ctrl+C to exit")
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Unsubscribe and disconnect
        if args.symbols:
            ws.unsubscribe_ticks(args.symbols, args.exchange)
        
        if args.indices:
            ws.unsubscribe_indices(args.indices, args.exchange)
        
        if args.orders:
            ws.unsubscribe_orders()
        
        ws.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()