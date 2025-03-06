# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

# File: util/timeframe.py
"""
Timeframe utilities for the Gann Trading System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pytz
from core.strategy.base_strategy import TimeFrame


class TimeframeUtils:
    """
    Utility class for handling timeframes and data conversions
    """
    
    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, 
                      timeframe: TimeFrame, 
                      datetime_column: str = 'datetime',
                      price_columns: Dict[str, str] = None) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe
            datetime_column: Name of the datetime column
            price_columns: Dictionary mapping standard column names to actual column names
            
        Returns:
            Resampled DataFrame
        """
        # Define default column mappings if not provided
        if price_columns is None:
            price_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
        # Ensure datetime index
        if datetime_column in df.columns:
            df = df.set_index(datetime_column)
        
        # Ensure datetime index is timezone-aware
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
            
        # Map pandas resample rule from TimeFrame enum
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
        rule = resample_map[timeframe]
        
        # Define aggregation functions
        agg_dict = {
            price_columns['open']: 'first',
            price_columns['high']: 'max',
            price_columns['low']: 'min',
            price_columns['close']: 'last'
        }
        
        # Add volume if present
        if price_columns['volume'] in df.columns:
            agg_dict[price_columns['volume']] = 'sum'
            
        # Resample data
        resampled = df.resample(rule).agg(agg_dict)
        
        # Reset index if needed
        if datetime_column != 'datetime':
            resampled = resampled.reset_index()
            resampled = resampled.rename(columns={'index': datetime_column})
        else:
            resampled = resampled.reset_index()
            
        return resampled
    
    @staticmethod
    def align_to_timeframe(timestamp: datetime, 
                         timeframe: TimeFrame, 
                         timezone: str = 'UTC') -> datetime:
        """
        Align a timestamp to the start of a timeframe
        
        Args:
            timestamp: Timestamp to align
            timeframe: Timeframe to align to
            timezone: Timezone for alignment
            
        Returns:
            Aligned timestamp
        """
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = pytz.timezone(timezone).localize(timestamp)
            
        # Align timestamp based on timeframe
        if timeframe == TimeFrame.MINUTE_1:
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_3:
            minute = (timestamp.minute // 3) * 3
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_10:
            minute = (timestamp.minute // 10) * 10
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_15:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_30:
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.HOUR_1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
# File: util/timeframe.py (continuation)
        elif timeframe == TimeFrame.HOUR_2:
           hour = (timestamp.hour // 2) * 2
           return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.HOUR_4:
           hour = (timestamp.hour // 4) * 4
           return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.DAY_1:
           return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.WEEK_1:
           # Calculate days to subtract to get to Monday
           days_to_monday = timestamp.weekday()
           monday = timestamp - timedelta(days=days_to_monday)
           return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
           raise ValueError(f"Unsupported timeframe: {timeframe}")
   
    @staticmethod
    def get_previous_candle_time(timestamp: datetime, 
                               timeframe: TimeFrame, 
                               timezone: str = 'UTC') -> datetime:
       """
       Get the start time of the previous candle
       
       Args:
           timestamp: Current timestamp
           timeframe: Timeframe
           timezone: Timezone
           
       Returns:
           Start time of the previous candle
       """
       # First, align to current timeframe
       aligned = TimeframeUtils.align_to_timeframe(timestamp, timeframe, timezone)
       
       # If timestamp is already aligned (exactly at candle start),
       # we need to go back one full candle
       if aligned == timestamp:
           # Go back one timeframe
           if timeframe == TimeFrame.MINUTE_1:
               return aligned - timedelta(minutes=1)
           elif timeframe == TimeFrame.MINUTE_3:
               return aligned - timedelta(minutes=3)
           elif timeframe == TimeFrame.MINUTE_5:
               return aligned - timedelta(minutes=5)
           elif timeframe == TimeFrame.MINUTE_10:
               return aligned - timedelta(minutes=10)
           elif timeframe == TimeFrame.MINUTE_15:
               return aligned - timedelta(minutes=15)
           elif timeframe == TimeFrame.MINUTE_30:
               return aligned - timedelta(minutes=30)
           elif timeframe == TimeFrame.HOUR_1:
               return aligned - timedelta(hours=1)
           elif timeframe == TimeFrame.HOUR_2:
               return aligned - timedelta(hours=2)
           elif timeframe == TimeFrame.HOUR_4:
               return aligned - timedelta(hours=4)
           elif timeframe == TimeFrame.DAY_1:
               return aligned - timedelta(days=1)
           elif timeframe == TimeFrame.WEEK_1:
               return aligned - timedelta(weeks=1)
           else:
               raise ValueError(f"Unsupported timeframe: {timeframe}")
       else:
           # If we're in the middle of a candle, return the start of the current candle
           return aligned
   
    @staticmethod
    def get_next_candle_time(timestamp: datetime, 
                          timeframe: TimeFrame, 
                          timezone: str = 'UTC') -> datetime:
       """
       Get the start time of the next candle
       
       Args:
           timestamp: Current timestamp
           timeframe: Timeframe
           timezone: Timezone
           
       Returns:
           Start time of the next candle
       """
       # First, align to current timeframe
       aligned = TimeframeUtils.align_to_timeframe(timestamp, timeframe, timezone)
       
       # If timestamp is already aligned (exactly at candle start),
       # the next candle is one timeframe ahead
       if aligned == timestamp:
           if timeframe == TimeFrame.MINUTE_1:
               return aligned + timedelta(minutes=1)
           elif timeframe == TimeFrame.MINUTE_3:
               return aligned + timedelta(minutes=3)
           elif timeframe == TimeFrame.MINUTE_5:
               return aligned + timedelta(minutes=5)
           elif timeframe == TimeFrame.MINUTE_10:
               return aligned + timedelta(minutes=10)
           elif timeframe == TimeFrame.MINUTE_15:
               return aligned + timedelta(minutes=15)
           elif timeframe == TimeFrame.MINUTE_30:
               return aligned + timedelta(minutes=30)
           elif timeframe == TimeFrame.HOUR_1:
               return aligned + timedelta(hours=1)
           elif timeframe == TimeFrame.HOUR_2:
               return aligned + timedelta(hours=2)
           elif timeframe == TimeFrame.HOUR_4:
               return aligned + timedelta(hours=4)
           elif timeframe == TimeFrame.DAY_1:
               return aligned + timedelta(days=1)
           elif timeframe == TimeFrame.WEEK_1:
               return aligned + timedelta(weeks=1)
           else:
               raise ValueError(f"Unsupported timeframe: {timeframe}")
       else:
           # If we're in the middle of a candle, return the start of the next candle
           if timeframe == TimeFrame.MINUTE_1:
               return aligned + timedelta(minutes=1)
           elif timeframe == TimeFrame.MINUTE_3:
               return aligned + timedelta(minutes=3)
           elif timeframe == TimeFrame.MINUTE_5:
               return aligned + timedelta(minutes=5)
           elif timeframe == TimeFrame.MINUTE_10:
               return aligned + timedelta(minutes=10)
           elif timeframe == TimeFrame.MINUTE_15:
               return aligned + timedelta(minutes=15)
           elif timeframe == TimeFrame.MINUTE_30:
               return aligned + timedelta(minutes=30)
           elif timeframe == TimeFrame.HOUR_1:
               return aligned + timedelta(hours=1)
           elif timeframe == TimeFrame.HOUR_2:
               return aligned + timedelta(hours=2)
           elif timeframe == TimeFrame.HOUR_4:
               return aligned + timedelta(hours=4)
           elif timeframe == TimeFrame.DAY_1:
               return aligned + timedelta(days=1)
           elif timeframe == TimeFrame.WEEK_1:
               return aligned + timedelta(weeks=1)
           else:
               raise ValueError(f"Unsupported timeframe: {timeframe}")
   
    @staticmethod
    def get_timeframe_delta(timeframe: TimeFrame) -> timedelta:
       """
       Get the timedelta for a timeframe
       
       Args:
           timeframe: Timeframe
           
       Returns:
           Timedelta for the timeframe
       """
       if timeframe == TimeFrame.MINUTE_1:
           return timedelta(minutes=1)
       elif timeframe == TimeFrame.MINUTE_3:
           return timedelta(minutes=3)
       elif timeframe == TimeFrame.MINUTE_5:
           return timedelta(minutes=5)
       elif timeframe == TimeFrame.MINUTE_10:
           return timedelta(minutes=10)
       elif timeframe == TimeFrame.MINUTE_15:
           return timedelta(minutes=15)
       elif timeframe == TimeFrame.MINUTE_30:
           return timedelta(minutes=30)
       elif timeframe == TimeFrame.HOUR_1:
           return timedelta(hours=1)
       elif timeframe == TimeFrame.HOUR_2:
           return timedelta(hours=2)
       elif timeframe == TimeFrame.HOUR_4:
           return timedelta(hours=4)
       elif timeframe == TimeFrame.DAY_1:
           return timedelta(days=1)
       elif timeframe == TimeFrame.WEEK_1:
           return timedelta(weeks=1)
       else:
           raise ValueError(f"Unsupported timeframe: {timeframe}")
   
    @staticmethod
    def get_market_hours(exchange: str = 'NSE') -> Tuple[datetime.time, datetime.time]:
       """
       Get market hours for an exchange
       
       Args:
           exchange: Exchange code
           
       Returns:
           Tuple of (market_open, market_close) times
       """
       if exchange in ['NSE', 'BSE', 'NFO']:
           # Indian markets: 9:15 AM to 3:30 PM
           market_open = datetime.strptime("09:15", "%H:%M").time()
           market_close = datetime.strptime("15:30", "%H:%M").time()
       elif exchange in ['NYSE', 'NASDAQ']:
           # US markets: 9:30 AM to 4:00 PM ET
           market_open = datetime.strptime("09:30", "%H:%M").time()
           market_close = datetime.strptime("16:00", "%H:%M").time()
       else:
           # Default to NSE hours
           market_open = datetime.strptime("09:15", "%H:%M").time()
           market_close = datetime.strptime("15:30", "%H:%M").time()
           
       return market_open, market_close
   
    @staticmethod
    def is_market_open(timestamp: datetime, exchange: str = 'NSE') -> bool:
       """
       Check if market is open at a given time
       
       Args:
           timestamp: Timestamp to check
           exchange: Exchange code
           
       Returns:
           True if market is open, False otherwise
       """
       # Get weekday (0 = Monday, 6 = Sunday)
       weekday = timestamp.weekday()
       
       # Check if weekend
       if weekday >= 5:  # Saturday or Sunday
           return False
           
       # Get market hours
       market_open, market_close = TimeframeUtils.get_market_hours(exchange)
       
       # Get time of day
       time_of_day = timestamp.time()
       
       # Check if within market hours
       return market_open <= time_of_day <= market_close


# Example usage when run directly
if __name__ == "__main__":
   # Test timeframe alignment
   now = datetime.now()
   print(f"Current time: {now}")
   
   for tf in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, 
             TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.DAY_1]:
       aligned = TimeframeUtils.align_to_timeframe(now, tf)
       next_candle = TimeframeUtils.get_next_candle_time(now, tf)
       prev_candle = TimeframeUtils.get_previous_candle_time(now, tf)
       
       print(f"{tf.value}:")
       print(f"  Aligned: {aligned}")
       print(f"  Previous: {prev_candle}")
       print(f"  Next: {next_candle}")
   
   # Test market hours
   market_open, market_close = TimeframeUtils.get_market_hours('NSE')
   print(f"\nNSE Market Hours: {market_open} to {market_close}")
   
   # Test market open check
   test_times = [
       datetime.now().replace(hour=9, minute=0),  # Before market open
       datetime.now().replace(hour=9, minute=30),  # After market open
       datetime.now().replace(hour=12, minute=0),  # Middle of trading day
       datetime.now().replace(hour=15, minute=45),  # After market close
   ]
   
   for test_time in test_times:
       is_open = TimeframeUtils.is_market_open(test_time)
       print(f"Market is {'open' if is_open else 'closed'} at {test_time.time()}")