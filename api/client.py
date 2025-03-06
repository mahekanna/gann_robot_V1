# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
API client for ICICI Direct Breeze API
"""

import logging
import datetime
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

from .auth import BreezeAuth

class BreezeClient:
    """
    Client for ICICI Direct Breeze API
    """
    
    def __init__(self, auth: Optional[BreezeAuth] = None, log_level: int = logging.INFO):
        """
        Initialize the client
        
        Args:
            auth: Authentication handler, if None a new one will be created
            log_level: Logging level
        """
        self.auth = auth if auth else BreezeAuth(log_level=log_level)
        self.breeze = None
        
        # Setup logging
        self.logger = logging.getLogger("breeze_client")
        self.logger.setLevel(log_level)
        
        # Add handlers only if none exist
        if not self.logger.handlers:
            # Create directory for logs if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / "breeze_client.log")
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
            
    def connect(self, force_new_session: bool = False) -> bool:
        """
        Connect to the API and get an authenticated client
        
        Args:
            force_new_session: Force creation of a new session
            
        Returns:
            True if connection successful, False otherwise
        """
        self.logger.info("Connecting to ICICI Direct API...")
        self.breeze = self.auth.get_breeze_client()
        
        if self.breeze is None:
            self.logger.error("Failed to connect to ICICI Direct API")
            return False
            
        self.logger.info("Successfully connected to ICICI Direct API")
        return True
        
    def reconnect_if_needed(self) -> bool:
        """
        Reconnect to the API if not connected or session is invalid
        
        Returns:
            True if connected, False otherwise
        """
        if self.breeze is None:
            return self.connect()
            
        if not self.auth.check_session_validity():
            self.logger.info("Session invalid, reconnecting...")
            return self.connect(force_new_session=True)
            
        return True
        
    def get_historical_data(self, 
                           stock_code: str,
                           exchange_code: str = 'NSE',
                           from_date: Optional[Union[str, datetime.datetime]] = None,
                           to_date: Optional[Union[str, datetime.datetime]] = None, 
                           interval: str = '1minute',
                           indices: bool = False) -> pd.DataFrame:
        """
        Get historical data for a stock or index
        
        Args:
            stock_code: Stock code
            exchange_code: Exchange code (NSE, BSE, NFO)
            from_date: Start date
            to_date: End date
            interval: Candle interval (1minute, 5minute, 15minute, 30minute, 1hour, 1day)
            indices: True if stock_code is an index
            
        Returns:
            DataFrame with historical data
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for historical data fetch")
            return pd.DataFrame()
            
        # Format dates if provided
        if from_date is not None:
            if isinstance(from_date, datetime.datetime):
                from_date = from_date.strftime("%d-%m-%Y %H:%M:%S")
            elif not isinstance(from_date, str):
                self.logger.error(f"Invalid from_date type: {type(from_date)}")
                return pd.DataFrame()
                
        if to_date is not None:
            if isinstance(to_date, datetime.datetime):
                to_date = to_date.strftime("%d-%m-%Y %H:%M:%S")
            elif not isinstance(to_date, str):
                self.logger.error(f"Invalid to_date type: {type(to_date)}")
                return pd.DataFrame()
                
        try:
            # Set default dates if not provided
            if from_date is None:
                # Default to 7 days ago
                from_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%d-%m-%Y %H:%M:%S")
                
            if to_date is None:
                # Default to now
                to_date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                
            self.logger.info(f"Fetching historical data: {stock_code}, {exchange_code}, {from_date} to {to_date}, {interval}")
            
            # Get historical data
            response = self.breeze.get_historical_data(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=stock_code,
                exchange_code=exchange_code,
                indices=indices
            )
            
            # Check response
            if not response or not isinstance(response, Dict) or 'Success' not in response or not response['Success']:
                self.logger.error(f"Failed to fetch historical data: {response}")
                return pd.DataFrame()
                
            data = response['Success']
            if not data:
                self.logger.warning("No historical data returned")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                # Add symbol column
                df['symbol'] = stock_code
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
            
    def get_quote(self, 
                 stock_code: str,
                 exchange_code: str = 'NSE',
                 indices: bool = False) -> Dict:
        """
        Get current quote for a stock or index
        
        Args:
            stock_code: Stock code
            exchange_code: Exchange code (NSE, BSE, NFO)
            indices: True if stock_code is an index
            
        Returns:
            Quote data
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for quote fetch")
            return {}
            
        try:
            self.logger.info(f"Fetching quote: {stock_code}, {exchange_code}")
            
            # Get quote
            response = self.breeze.get_quotes(
                stock_code=stock_code,
                exchange_code=exchange_code,
                indices=indices
            )
            
            # Check response
            if not response or not isinstance(response, Dict) or 'Success' not in response or not response['Success']:
                self.logger.error(f"Failed to fetch quote: {response}")
                return {}
                
            return response['Success'][0] if response['Success'] else {}
            
        except Exception as e:
            self.logger.error(f"Error fetching quote: {e}")
            return {}
            
    def get_option_chain(self, 
                        stock_code: str,
                        exchange_code: str = 'NFO',
                        expiry_date: Optional[str] = None,
                        strike_price: Optional[float] = None,
                        option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get option chain data
        
        Args:
            stock_code: Stock code
            exchange_code: Exchange code (NFO for options)
            expiry_date: Expiry date (format: YYYYMMDD)
            strike_price: Strike price
            option_type: Option type (CE or PE)
            
        Returns:
            DataFrame with option chain data
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for option chain fetch")
            return pd.DataFrame()
            
        try:
            self.logger.info(f"Fetching option chain: {stock_code}")
            
            # Get option chain
            response = self.breeze.get_option_chain(
                stock_code=stock_code,
                exchange_code=exchange_code,
                expiry_date=expiry_date,
                strike_price=strike_price,
                option_type=option_type
            )
            
            # Check response
            if not response or not isinstance(response, Dict) or 'Success' not in response or not response['Success']:
                self.logger.error(f"Failed to fetch option chain: {response}")
                return pd.DataFrame()
                
            data = response['Success']
            if not data:
                self.logger.warning("No option chain data returned")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                numeric_columns = ['strike_price', 'last_price', 'change', 'open_interest']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain: {e}")
            return pd.DataFrame()
            
    def place_order(self, 
                   stock_code: str,
                   exchange_code: str = 'NSE',
                   quantity: int = 1,
                   price: float = 0,
                   product_type: str = 'MIS',
                   transaction_type: str = 'B',
                   order_type: str = 'L',
                   validity: str = 'DAY',
                   disclosed_quantity: int = 0,
                   trigger_price: float = 0) -> Dict:
        """
        Place an order
        
        Args:
            stock_code: Stock code
            exchange_code: Exchange code (NSE, BSE, NFO)
            quantity: Order quantity
            price: Order price (0 for market order)
            product_type: Product type (MIS, CNC, NRML)
            transaction_type: Transaction type (B for Buy, S for Sell)
            order_type: Order type (L for Limit, M for Market, SL for Stop Loss, SL-M for Stop Loss Market)
            validity: Order validity (DAY, IOC)
            disclosed_quantity: Disclosed quantity
            trigger_price: Trigger price for stop loss orders
            
        Returns:
            Order response
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for order placement")
            return {"Status": "Error", "Error": "Failed to reconnect"}
            
        try:
            self.logger.info(f"Placing order: {stock_code}, {exchange_code}, {transaction_type}, {quantity}, {price}")
            
            # Place order
            response = self.breeze.place_order(
                stock_code=stock_code,
                exchange_code=exchange_code,
                quantity=quantity,
                price=price,
                product_type=product_type,
                transaction_type=transaction_type,
                order_type=order_type,
                validity=validity,
                disclosed_quantity=disclosed_quantity,
                trigger_price=trigger_price
            )
            
            self.logger.info(f"Order placed response: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"Status": "Error", "Error": str(e)}
            
    def get_portfolio_holdings(self) -> pd.DataFrame:
        """
        Get portfolio holdings
        
        Returns:
            DataFrame with portfolio holdings
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for portfolio holdings fetch")
            return pd.DataFrame()
            
        try:
            self.logger.info("Fetching portfolio holdings")
            
            # Get portfolio holdings
            response = self.breeze.get_portfolio_holdings()
            
            # Check response
            if not response or not isinstance(response, Dict) or 'Success' not in response or not response['Success']:
                self.logger.error(f"Failed to fetch portfolio holdings: {response}")
                return pd.DataFrame()
                
            data = response['Success']
            if not data:
                self.logger.warning("No portfolio holdings data returned")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                numeric_columns = ['quantity', 'last_price', 'average_price', 'close_price']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching portfolio holdings: {e}")
            return pd.DataFrame()
            
    def get_orders(self) -> pd.DataFrame:
        """
        Get all orders
        
        Returns:
            DataFrame with orders
        """
        if not self.reconnect_if_needed():
            self.logger.error("Failed to reconnect for orders fetch")
            return pd.DataFrame()
            
        try:
            self.logger.info("Fetching orders")
            
            # Get orders
            response = self.breeze.get_order_list()
            
            # Check response
            if not response or not isinstance(response, Dict) or 'Success' not in response or not response['Success']:
                self.logger.error(f"Failed to fetch orders: {response}")
                return pd.DataFrame()
                
            data = response['Success']
            if not data:
                self.logger.warning("No orders data returned")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert columns to appropriate types
            if not df.empty:
                if 'order_execution_time' in df.columns:
                    df['order_execution_time'] = pd.to_datetime(df['order_execution_time'], errors='coerce')
                    
                numeric_columns = ['quantity', 'price', 'trigger_price']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching orders: {e}")
            return pd.DataFrame()