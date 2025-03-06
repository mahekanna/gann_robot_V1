# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

# File: data/market_data.py
"""
Market Data Manager for the Gann Trading System
Handles retrieving, caching, and managing market data
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from api.factory import get_client
from core.strategy.base_strategy import TimeFrame

class MarketDataManager:
    """
    Manager for retrieving and caching market data
    """
    
    def __init__(self, 
                data_dir: str = 'data',
                cache_duration: int = 24,  # Hours
                log_level: str = 'INFO'):
        """
        Initialize market data manager
        
        Args:
            data_dir: Directory for data storage
            cache_duration: Cache duration in hours
            log_level: Logging level
        """
        self.data_dir = Path(data_dir)
        self.cache_duration = cache_duration
        
        # Create directories
        self.historical_dir = self.data_dir / 'historical'
        self.cache_dir = self.data_dir / 'cache'
        self.options_dir = self.data_dir / 'options'
        
        self.historical_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.options_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logging
        self.logger = logging.getLogger("market_data")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Add handlers only if none exist
        if not self.logger.handlers:
            # Create directory for logs if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / "market_data.log")
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
        
        # Initialize API client
        self.client = None
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: TimeFrame,
                          start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None,
                          exchange: str = 'NSE',
                          force_download: bool = False,
                          use_api: bool = True,
                          indices: bool = False) -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date
            end_date: End date
            exchange: Exchange code
            force_download: Force download even if cache is valid
            use_api: Whether to use API if cache is invalid
            indices: Whether symbol is an index
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
            
        # Ensure UTC timezone
        start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(end_date).tz_localize(None)
        
        # Format dates for file naming
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Try to get from cache first
        cache_file = self.cache_dir / f"{symbol}_{exchange}_{timeframe.value}_{start_str}_{end_str}.csv"
        
        if not force_download and cache_file.exists():
            # Check if cache is still valid
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < timedelta(hours=self.cache_duration):
                self.logger.info(f"Loading cached data for {symbol} ({timeframe.value})")
                df = pd.read_csv(cache_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
        
        # If we reach here, we need to get the data
        # First, try to get from historical data directory
        historical_file = self.historical_dir / f"{symbol}_{exchange}_{timeframe.value}.csv"
        
        if historical_file.exists():
            self.logger.info(f"Loading historical data for {symbol} ({timeframe.value})")
            df = pd.read_csv(historical_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter by date range
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            
            if not df.empty:
                # Save to cache
                df.to_csv(cache_file, index=False)
                return df
        
        # If we reach here, we need to download the data
        if use_api:
            self.logger.info(f"Downloading data for {symbol} ({timeframe.value})")
            
            # Initialize API client if needed
            if self.client is None:
                self.client = get_client()
                
                # Connect to API
                if not self.client.connect():
                    self.logger.error("Failed to connect to API")
                    return pd.DataFrame()
            
            # Map TimeFrame enum to interval string
            interval_map = {
                TimeFrame.MINUTE_1: "1minute",
                TimeFrame.MINUTE_3: "3minute",
                TimeFrame.MINUTE_5: "5minute",
                TimeFrame.MINUTE_10: "10minute",
                TimeFrame.MINUTE_15: "15minute",
                TimeFrame.MINUTE_30: "30minute",
                TimeFrame.HOUR_1: "1hour",
                TimeFrame.HOUR_2: "2hour",
                TimeFrame.HOUR_4: "4hour",
                TimeFrame.DAY_1: "1day",
                TimeFrame.WEEK_1: "1week"
            }
            
            # Get data from API
            df = self.client.get_historical_data(
                stock_code=symbol,
                exchange_code=exchange,
                from_date=start_date,
                to_date=end_date,
                interval=interval_map[timeframe],
                indices=indices
            )
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol} ({timeframe.value})")
                return df
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            
            # Also save to historical directory with append
            if historical_file.exists():
                # Load existing historical data
                historical_df = pd.read_csv(historical_file)
                historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])
                
                # Combine with new data
                combined_df = pd.concat([historical_df, df])
                
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['datetime'])
                
                # Sort by datetime
                combined_df = combined_df.sort_values('datetime')
                
                # Save back to historical file
                combined_df.to_csv(historical_file, index=False)
            else:
                # Save new historical file
                df.to_csv(historical_file, index=False)
            
            return df
        
        # If we reach here, we couldn't get the data
        self.logger.warning(f"No data available for {symbol} ({timeframe.value})")
        return pd.DataFrame()
    
    def get_option_chain(self, 
                        symbol: str,
                        expiry_date: Optional[Union[str, datetime]] = None,
                        exchange: str = 'NFO') -> pd.DataFrame:
        """
        Get option chain for a symbol
        
        Args:
            symbol: Underlying symbol
            expiry_date: Option expiry date
            exchange: Exchange code
            
        Returns:
            DataFrame with option chain data
        """
        # Try to get from cache first
        expiry_str = ""
        if expiry_date:
            if isinstance(expiry_date, str):
                expiry_str = expiry_date
            else:
                expiry_str = expiry_date.strftime('%Y%m%d')
                
            cache_file = self.options_dir / f"{symbol}_{exchange}_{expiry_str}.csv"
            
            if cache_file.exists():
                # Check if cache is still valid (options data changes rapidly, use shorter cache)
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(hours=1):  # 1 hour cache for options
                    self.logger.info(f"Loading cached option chain for {symbol} (expiry: {expiry_str})")
                    df = pd.read_csv(cache_file)
                    return df
        
        # If we reach here, we need to download the data
        self.logger.info(f"Downloading option chain for {symbol}")
        
        # Initialize API client if needed
        if self.client is None:
            self.client = get_client()
            
            # Connect to API
            if not self.client.connect():
                self.logger.error("Failed to connect to API")
                return pd.DataFrame()
        
        # Get option chain from API
        df = self.client.get_option_chain(
            stock_code=symbol,
            exchange_code=exchange,
            expiry_date=expiry_str if expiry_str else None
        )
        
        if df.empty:
            self.logger.warning(f"No option chain available for {symbol}")
            return df
        
        # Save to cache if we have an expiry date
        if expiry_str:
            cache_file = self.options_dir / f"{symbol}_{exchange}_{expiry_str}.csv"
            df.to_csv(cache_file, index=False)
        
        return df
    
    def get_quote(self, 
                symbol: str,
                exchange: str = 'NSE',
                indices: bool = False) -> Dict:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            indices: Whether symbol is an index
            
        Returns:
            Quote data
        """
        # Initialize API client if needed
        if self.client is None:
            self.client = get_client()
            
            # Connect to API
            if not self.client.connect():
                self.logger.error("Failed to connect to API")
                return {}
        
        # Get quote from API
        quote = self.client.get_quote(
            stock_code=symbol,
            exchange_code=exchange,
            indices=indices
        )
        
        return quote
    
    def save_backtest_data(self, 
                         symbol: str,
                         timeframe: TimeFrame,
                         data: pd.DataFrame,
                         exchange: str = 'NSE') -> bool:
        """
        Save data for backtesting
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: DataFrame with OHLCV data
            exchange: Exchange code
            
        Returns:
            True if successful, False otherwise
        """
        # Create backtest directory if it doesn't exist
        backtest_dir = self.data_dir / 'backtest'
        backtest_dir.mkdir(exist_ok=True)
        
        # Save data
        file_path = backtest_dir / f"{symbol}_{exchange}_{timeframe.value}.csv"
        
        try:
            data.to_csv(file_path, index=False)
            self.logger.info(f"Saved backtest data to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save backtest data: {e}")
            return False
            
# File: data/market_data.py (continuation)
    def load_backtest_data(self,
                         symbol: str,
                         timeframe: TimeFrame,
                         exchange: str = 'NSE') -> pd.DataFrame:
       """
       Load data for backtesting
       
       Args:
           symbol: Trading symbol
           timeframe: Timeframe
           exchange: Exchange code
           
       Returns:
           DataFrame with OHLCV data
       """
       # Check if file exists
       backtest_dir = self.data_dir / 'backtest'
       file_path = backtest_dir / f"{symbol}_{exchange}_{timeframe.value}.csv"
       
       if not file_path.exists():
           self.logger.warning(f"No backtest data found for {symbol} ({timeframe.value})")
           return pd.DataFrame()
       
       # Load data
       try:
           df = pd.read_csv(file_path)
           df['datetime'] = pd.to_datetime(df['datetime'])
           self.logger.info(f"Loaded backtest data from {file_path}: {len(df)} rows")
           return df
       except Exception as e:
           self.logger.error(f"Failed to load backtest data: {e}")
           return pd.DataFrame()
   
    def get_expiry_dates(self, symbol: str, exchange: str = 'NFO') -> List[str]:
           """
           Get available expiry dates for options
           
           Args:
               symbol: Underlying symbol
               exchange: Exchange code
               
           Returns:
               List of expiry dates in YYYYMMDD format
           """
           # Initialize API client if needed
           if self.client is None:
               self.client = get_client()
               
               # Connect to API
               if not self.client.connect():
                   self.logger.error("Failed to connect to API")
                   return []
           
           # Get option chain with no expiry to get available expiries
           df = self.client.get_option_chain(
               stock_code=symbol,
               exchange_code=exchange
           )
           
           if df.empty:
               self.logger.warning(f"No option chain available for {symbol}")
               return []
           
           # Extract unique expiry dates
           if 'expiry_date' in df.columns:
               expiry_dates = sorted(df['expiry_date'].unique().tolist())
               return expiry_dates
           
           return []
   
    def get_atm_strike(self, symbol: str, expiry_date: str = None, exchange: str = 'NFO') -> float:
           """
           Get at-the-money strike price for options
           
           Args:
               symbol: Underlying symbol
               expiry_date: Option expiry date
               exchange: Exchange code
               
           Returns:
               ATM strike price
           """
           # Get current price of underlying
           quote = self.get_quote(symbol, 'NSE')
           if not quote or 'last_price' not in quote:
               self.logger.error(f"Failed to get quote for {symbol}")
               return 0.0
           
           current_price = quote['last_price']
           
           # Get option chain
           df = self.get_option_chain(symbol, expiry_date, exchange)
           
           if df.empty:
               self.logger.warning(f"No option chain available for {symbol}")
               return 0.0
           
           # Extract unique strike prices
           if 'strike_price' in df.columns:
               strikes = sorted(df['strike_price'].unique())
               
               # Find closest strike to current price
               atm_strike = min(strikes, key=lambda x: abs(x - current_price))
               return atm_strike
           
           return 0.0
   
    def get_option_details(self, 
                             symbol: str, 
                             expiry_date: str, 
                             strike: float, 
                             option_type: str,
                             exchange: str = 'NFO') -> Dict:
           """
           Get details for a specific option
           
           Args:
               symbol: Underlying symbol
               expiry_date: Option expiry date
               strike: Strike price
               option_type: Option type ('CE' or 'PE')
               exchange: Exchange code
               
           Returns:
               Dictionary with option details
           """
           # Get option chain
           df = self.get_option_chain(symbol, expiry_date, exchange)
           
           if df.empty:
               self.logger.warning(f"No option chain available for {symbol}")
               return {}
           
           # Filter for specific option
           option_df = df[(df['strike_price'] == strike) & (df['option_type'] == option_type)]
           
           if option_df.empty:
               self.logger.warning(f"Option not found: {symbol} {expiry_date} {strike} {option_type}")
               return {}
           
           # Return as dictionary
           return option_df.iloc[0].to_dict()
   
    def download_historical_data(self, 
                                  symbols: List[str], 
                                  timeframes: List[TimeFrame],
                                  start_date: datetime,
                                  end_date: datetime,
                                  exchange: str = 'NSE',
                                  indices: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
           """
           Download historical data for multiple symbols and timeframes
           
           Args:
               symbols: List of trading symbols
               timeframes: List of timeframes
               start_date: Start date
               end_date: End date
               exchange: Exchange code
               indices: Whether symbols are indices
               
           Returns:
               Dictionary of symbol -> timeframe -> DataFrame
           """
           result = {}
           
           for symbol in symbols:
               result[symbol] = {}
               
               for timeframe in timeframes:
                   self.logger.info(f"Downloading {symbol} {timeframe.value} data")
                   
                   df = self.get_historical_data(
                       symbol=symbol,
                       timeframe=timeframe,
                       start_date=start_date,
                       end_date=end_date,
                       exchange=exchange,
                       force_download=True,
                       use_api=True,
                       indices=indices
                   )
                   
                   result[symbol][timeframe] = df
                   
                   # Sleep to avoid hitting API rate limits
                   import time
                   time.sleep(1)
           
           return result
   
    def convert_timeframe(self, df: pd.DataFrame, target_timeframe: TimeFrame) -> pd.DataFrame:
           """
           Convert data from one timeframe to another
           
           Args:
               df: DataFrame with OHLCV data
               target_timeframe: Target timeframe
               
           Returns:
               DataFrame with converted timeframe
           """
           # Ensure datetime index
           if 'datetime' in df.columns:
               df = df.set_index('datetime')
           
           # Map timeframe to pandas resample rule
           resample_map = {
               TimeFrame.MINUTE_1: '1min',
               TimeFrame.MINUTE_3: '3min',
               TimeFrame.MINUTE_5: '5min',
               TimeFrame.MINUTE_10: '10min',
               TimeFrame.MINUTE_15: '15min',
               TimeFrame.MINUTE_30: '30min',
               TimeFrame.HOUR_1: '1H',
               TimeFrame.HOUR_2: '2H',
               TimeFrame.HOUR_4: '4H',
               TimeFrame.DAY_1: '1D',
               TimeFrame.WEEK_1: '1W'
           }
           
           # Get resample rule
           rule = resample_map[target_timeframe]
           
           # Resample data
           resampled = df.resample(rule).agg({
               'open': 'first',
               'high': 'max',
               'low': 'min',
               'close': 'last',
               'volume': 'sum'
           })
           
           # Reset index
           resampled = resampled.reset_index()
           
           return resampled
   
    def merge_data(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
           """
           Merge multiple dataframes with OHLCV data
           
           Args:
               dfs: List of DataFrames with OHLCV data
               
           Returns:
               Merged DataFrame
           """
           if not dfs:
               return pd.DataFrame()
           
           # Ensure datetime columns
           for i, df in enumerate(dfs):
               if 'datetime' not in df.columns and df.index.name != 'datetime':
                   if df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
                       df = df.reset_index()
                   else:
                       self.logger.error(f"DataFrame {i} does not have datetime column or index")
                       return pd.DataFrame()
           
           # Concatenate dataframes
           merged = pd.concat(dfs)
           
           # Remove duplicates
           merged = merged.drop_duplicates(subset=['datetime'])
           
           # Sort by datetime
           merged = merged.sort_values('datetime')
           
           return merged
       
    def get_future_contract(self, 
                             symbol: str, 
                             expiry_date: str = None,
                             exchange: str = 'NFO') -> Dict:
           """
           Get future contract details
           
           Args:
               symbol: Underlying symbol
               expiry_date: Future expiry date
               exchange: Exchange code
               
           Returns:
               Dictionary with future contract details
           """
           # Initialize API client if needed
           if self.client is None:
               self.client = get_client()
               
               # Connect to API
               if not self.client.connect():
                   self.logger.error("Failed to connect to API")
                   return {}
           
           try:
               # Use the security master to get future contract details
               from api.factory import get_security_master
               security_master = get_security_master()
               
               # Get future contracts for the symbol
               if expiry_date:
                   # Search for specific expiry
                   contracts = security_master.search_securities(
                       f"{symbol} FUT {expiry_date}", exchange=exchange
                   )
               else:
                   # Search for all futures
                   contracts = security_master.search_securities(
                       f"{symbol} FUT", exchange=exchange
                   )
               
               if not contracts:
                   self.logger.warning(f"No future contracts found for {symbol}")
                   return {}
               
               # If expiry date is provided, find the matching contract
               if expiry_date:
                   for contract in contracts:
                       if 'expiry_date' in contract and contract['expiry_date'] == expiry_date:
                           return contract
               
               # Otherwise, return the nearest expiry
               if 'expiry_date' in contracts[0]:
                   # Sort by expiry date
                   contracts.sort(key=lambda x: x['expiry_date'])
                   return contracts[0]
               
               return contracts[0]
           except Exception as e:
               self.logger.error(f"Failed to get future contract: {e}")
               return {}
   
    def generate_synthetic_data(self,
                                 symbol: str,
                                 timeframe: TimeFrame,
                                 days: int = 60,
                                 start_price: float = 1000.0,
                                 volatility: float = 0.015,
                                 trend: float = 0.0002) -> pd.DataFrame:
           """
           Generate synthetic OHLCV data for testing
           
           Args:
               symbol: Symbol name
               timeframe: Data timeframe
               days: Number of days of data
               start_price: Starting price
               volatility: Daily volatility
               trend: Daily trend factor (positive=uptrend, negative=downtrend)
               
           Returns:
               DataFrame with synthetic OHLCV data
           """
           self.logger.info(f"Generating synthetic data for {symbol} ({timeframe.value})")
           
           import numpy as np
           
           # Set random seed for reproducibility
           np.random.seed(42)
           
           # Generate dates
           end_date = datetime.now().date()
           start_date = end_date - timedelta(days=days)
           
           # Generate timestamps based on timeframe
           timestamps = []
           
           # Map timeframe to timedelta
           timeframe_map = {
               TimeFrame.MINUTE_1: timedelta(minutes=1),
               TimeFrame.MINUTE_3: timedelta(minutes=3),
               TimeFrame.MINUTE_5: timedelta(minutes=5),
               TimeFrame.MINUTE_10: timedelta(minutes=10),
               TimeFrame.MINUTE_15: timedelta(minutes=15),
               TimeFrame.MINUTE_30: timedelta(minutes=30),
               TimeFrame.HOUR_1: timedelta(hours=1),
               TimeFrame.HOUR_2: timedelta(hours=2),
               TimeFrame.HOUR_4: timedelta(hours=4),
               TimeFrame.DAY_1: timedelta(days=1),
               TimeFrame.WEEK_1: timedelta(days=7)
           }
           
           td = timeframe_map[timeframe]
           
           if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1]:
               # Simple date range for daily/weekly data
               current_date = start_date
               while current_date <= end_date:
                   if timeframe == TimeFrame.WEEK_1:
                       # Ensure weeks start on Monday
                       weekday = current_date.weekday()
                       if weekday == 0:  # Monday
                           timestamps.append(datetime.combine(current_date, datetime.min.time()))
                   else:
                       # Skip weekends for daily data
                       if current_date.weekday() < 5:  # 0-4 = Monday-Friday
                           timestamps.append(datetime.combine(current_date, datetime.min.time()))
                   current_date += td
           else:
               # Intraday data - use market hours
               market_open = datetime.strptime("09:15", "%H:%M").time()
               market_close = datetime.strptime("15:30", "%H:%M").time()
               
               current_date = start_date
               while current_date <= end_date:
                   # Skip weekends
                   if current_date.weekday() < 5:  # 0-4 = Monday-Friday
                       # Start at market open
                       current_time = datetime.combine(current_date, market_open)
                       end_time = datetime.combine(current_date, market_close)
                       
                       while current_time <= end_time:
                           timestamps.append(current_time)
                           current_time += td
                   
                   current_date += timedelta(days=1)
           
           # Generate prices
           prices = [start_price]
           for i in range(1, len(timestamps)):
               # Check if this is a new day
               is_new_day = timestamps[i].date() != timestamps[i-1].date()
               
               if is_new_day:
                   # More volatility between days
                   daily_return = np.random.normal(trend, volatility)
               else:
                   # Less volatility intraday
                   daily_return = np.random.normal(trend/5, volatility/3)
                   
               prices.append(prices[-1] * (1 + daily_return))
           
           # Generate OHLCV data
           data = []
           for i, timestamp in enumerate(timestamps):
               # Determine if this is the first candle of the day
               if i == 0 or timestamps[i].date() != timestamps[i-1].date():
                   # First candle of the day
                   prev_close = prices[i-1] if i > 0 else start_price * 0.99
                   
                   # Generate gap from previous close
                   gap = np.random.normal(0, volatility/2)
                   open_price = prev_close * (1 + gap)
                   
                   close = prices[i]
                   high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility/10)))
                   low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility/10)))
               else:
                   # Regular candle
                   open_price = prices[i-1]  # Open at previous close
                   close = prices[i]
                   
                   # High and low with some randomness
                   high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility/10)))
                   low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility/10)))
               
               # Volume varies with price movement
               price_change = abs(close - open_price) / open_price
               volume = int(np.random.uniform(50000, 200000) * (1 + 5 * price_change))
               
               data.append({
                   'datetime': timestamp,
                   'open': round(open_price, 2),
                   'high': round(high, 2),
                   'low': round(low, 2),
                   'close': round(close, 2),
                   'volume': volume,
                   'symbol': symbol
               })
           
           # Create DataFrame
           df = pd.DataFrame(data)
           
           # Save to backtest directory for future use
           backtest_dir = self.data_dir / 'backtest'
           backtest_dir.mkdir(exist_ok=True)
           
           file_path = backtest_dir / f"{symbol}_SYNTHETIC_{timeframe.value}.csv"
           df.to_csv(file_path, index=False)
           
           self.logger.info(f"Generated {len(df)} rows of synthetic data for {symbol}, saved to {file_path}")
           
           return df


# Example usage when run directly
if __name__ == "__main__":
   # Setup basic logging configuration
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   
   # Create data manager
   data_manager = MarketDataManager()
   
   # Test generating synthetic data
   df = data_manager.generate_synthetic_data(
       symbol="SAMPLESTOCK",
       timeframe=TimeFrame.DAY_1,
       days=60,
       start_price=1000.0,
       volatility=0.015,
       trend=0.0002
   )
   
   print(f"Generated {len(df)} rows of data")
   print(df.head())
   
   # Test different timeframe
   df_hourly = data_manager.generate_synthetic_data(
       symbol="SAMPLESTOCK",
       timeframe=TimeFrame.HOUR_1,
       days=10
   )
   
   print(f"Generated {len(df_hourly)} rows of hourly data")
   print(df_hourly.head())
