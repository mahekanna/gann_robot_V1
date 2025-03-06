# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

#!/usr/bin/env python3
"""
Test Data Generator for Backtesting
This script generates synthetic OHLCV data for testing the Gann trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import argparse
from typing import List, Dict, Optional, Union, Tuple


def generate_equity_data(symbol: str, 
                         days: int = 60, 
                         start_price: float = 1000.0,
                         volatility: float = 0.015,
                         trend: float = 0.0002,
                         filename: Optional[str] = None,
                         market_hours: bool = True) -> pd.DataFrame:
    """
    Generate synthetic equity price data
    
    Args:
        symbol: Stock symbol
        days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility
        trend: Daily trend factor (positive=uptrend, negative=downtrend)
        filename: Output filename (if None, won't save to file)
        market_hours: If True, only generate data during market hours
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    end_date = datetime.now().date()
    dates = []
    current_date = end_date - timedelta(days=days)
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0, Sunday=6
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Generate minute-level timestamps if using market hours
    if market_hours:
        # Indian market hours: 9:15 AM to 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        minute_timestamps = []
        for date in dates:
            current_time = datetime.combine(date, market_open)
            end_time = datetime.combine(date, market_close)
            
            while current_time <= end_time:
                minute_timestamps.append(current_time)
                current_time += timedelta(minutes=1)
        
        timestamps = minute_timestamps
    else:
        # Use daily timestamps
        timestamps = [datetime.combine(date, time(0, 0)) for date in dates]
    
    # Generate price series with random walk
    prices = [start_price]
    for i in range(1, len(timestamps)):
        # If this is a new day, add more randomness
        if timestamps[i].date() != timestamps[i-1].date():
            # Random daily return with trend bias
            daily_factor = np.random.normal(trend, volatility)
        else:
            # Intraday moves are smaller
            daily_factor = np.random.normal(trend/10, volatility/5)
            
        prices.append(prices[-1] * (1 + daily_factor))
    
    # Generate OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps):
        if i == 0 or timestamps[i].date() != timestamps[i-1].date():
            # First minute of the day
            prev_close = prices[i-1] if i > 0 else start_price * 0.99
            # Open with a gap from previous close
            gap = np.random.normal(0, volatility/2)
            open_price = prev_close * (1 + gap)
            high = max(open_price, prices[i])
            low = min(open_price, prices[i])
            close = prices[i]
        else:
            # Intraday minute
            open_price = prices[i-1]  # Open at previous minute's close
            close = prices[i]
            high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility/10)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility/10)))
        
        # Volume varies with price movement and time of day
        hour = timestamp.hour
        minute = timestamp.minute
        time_factor = 1.0
        
        # More volume at open and close
        if (hour == 9 and minute < 30) or (hour == 15 and minute > 0):
            time_factor = 2.0
        
        # More volume on price moves
        price_change = abs(close - open_price) / open_price
        volume = int(np.random.uniform(50000, 200000) * (1 + 10 * price_change) * time_factor)
        
        data.append({
            'datetime': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'symbol': symbol
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to file if specified
    if filename:
        os.makedirs('data/generated', exist_ok=True)
        filepath = os.path.join('data/generated', filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
    
    return df


def generate_index_data(symbol: str, 
                       days: int = 60, 
                       start_price: float = 20000.0,
                       volatility: float = 0.01,
                       trend: float = 0.0002,
                       filename: Optional[str] = None) -> pd.DataFrame:
    """
    Generate synthetic index price data
    
    Args:
        symbol: Index symbol (e.g., "NIFTY", "BANKNIFTY")
        days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility
        trend: Daily trend factor
        filename: Output filename
        
    Returns:
        DataFrame with OHLCV data
    """
    # Indices are less volatile than individual stocks
    return generate_equity_data(
        symbol=symbol,
        days=days,
        start_price=start_price,
        volatility=volatility,
        trend=trend,
        filename=filename,
        market_hours=True  # Always use market hours for indices
    )


def generate_option_chain(underlying_price: float, 
                         symbol: str,
                         expiry_date: datetime,
                         strikes: Optional[List[float]] = None,
                         num_strikes: int = 10,
                         strike_gap: Optional[float] = None) -> pd.DataFrame:
    """
    Generate synthetic option chain data
    
    Args:
        underlying_price: Current price of the underlying
        symbol: Underlying symbol
        expiry_date: Expiry date for the options
        strikes: List of strike prices (if None, will generate around ATM)
        num_strikes: Number of strikes to generate on each side of ATM
        strike_gap: Gap between strikes (if None, will use 0.5% of price)
        
    Returns:
        DataFrame with option chain data
    """
    # If strikes not provided, generate around ATM
    if strikes is None:
        if strike_gap is None:
            # Use 0.5% of price as default strike gap
            strike_gap = round(underlying_price * 0.005, 1)
            
            # Round to nearest standard strike value
            if underlying_price > 10000:
                strike_gap = round(strike_gap / 50) * 50  # Round to nearest 50
            elif underlying_price > 1000:
                strike_gap = round(strike_gap / 10) * 10  # Round to nearest 10
            elif underlying_price > 100:
                strike_gap = round(strike_gap / 5) * 5    # Round to nearest 5
            else:
                strike_gap = round(strike_gap)            # Round to nearest 1
        
        # Find ATM strike (nearest to current price)
        atm_strike = round(underlying_price / strike_gap) * strike_gap
        
        # Generate strikes around ATM
        strikes = [atm_strike + (i - num_strikes) * strike_gap for i in range(num_strikes * 2 + 1)]
    
    # Generate option data
    today = datetime.now()
    days_to_expiry = (expiry_date - today).days
    
    # Simple Black-Scholes approximation for IV and prices
    data = []
    for strike in strikes:
        # Call option
        moneyness = underlying_price / strike
        iv_call = 0.2 + 0.1 * (1 - moneyness)  # Higher IV for OTM calls
        iv_call = max(0.1, min(0.8, iv_call))
        
        # Very basic price approximation
        if strike <= underlying_price:  # ITM call
            intrinsic = underlying_price - strike
            time_value = underlying_price * iv_call * np.sqrt(days_to_expiry/365)
            call_price = intrinsic + time_value
        else:  # OTM call
            call_price = underlying_price * iv_call * np.sqrt(days_to_expiry/365) * np.exp(-0.5 * ((strike/underlying_price)-1)**2)
        
        # Put option
        moneyness_put = strike / underlying_price
        iv_put = 0.2 + 0.1 * (moneyness_put - 1)  # Higher IV for OTM puts
        iv_put = max(0.1, min(0.8, iv_put))
        
        # Basic price approximation
        if strike >= underlying_price:  # ITM put
            intrinsic = strike - underlying_price
            time_value = underlying_price * iv_put * np.sqrt(days_to_expiry/365)
            put_price = intrinsic + time_value
        else:  # OTM put
            put_price = underlying_price * iv_put * np.sqrt(days_to_expiry/365) * np.exp(-0.5 * ((underlying_price/strike)-1)**2)
        
        # Call option data
        data.append({
            'symbol': f"{symbol}{expiry_date.strftime('%d%b%y').upper()}C{int(strike)}",
            'underlying': symbol,
            'expiry_date': expiry_date,
            'strike': strike,
            'type': 'CE',
            'last_price': round(call_price, 2),
            'iv': round(iv_call * 100, 2),
            'delta': round(0.5 + 0.5 * (1 if underlying_price > strike else -1) * (1 - np.exp(-0.5 * ((underlying_price/strike)-1)**2)), 2),
            'open_interest': int(np.random.uniform(100, 1000) * np.exp(-0.5 * ((strike/underlying_price)-1)**2)),
            'underlying_price': underlying_price
        })
        
        # Put option data
        data.append({
            'symbol': f"{symbol}{expiry_date.strftime('%d%b%y').upper()}P{int(strike)}",
            'underlying': symbol,
            'expiry_date': expiry_date,
            'strike': strike,
            'type': 'PE',
            'last_price': round(put_price, 2),
            'iv': round(iv_put * 100, 2),
            'delta': round(-0.5 - 0.5 * (1 if underlying_price < strike else -1) * (1 - np.exp(-0.5 * ((underlying_price/strike)-1)**2)), 2),
            'open_interest': int(np.random.uniform(100, 1000) * np.exp(-0.5 * ((underlying_price/strike)-1)**2)),
            'underlying_price': underlying_price
        })
    
    return pd.DataFrame(data)


def main(args: List[str] = None) -> None:
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Generate synthetic market data for backtesting')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Symbol to generate data for')
    parser.add_argument('--type', choices=['equity', 'index', 'option'], default='equity', help='Type of data to generate')
    parser.add_argument('--days', type=int, default=60, help='Number of days of data')
    parser.add_argument('--price', type=float, default=None, help='Starting price (default: 1000 for equity, 20000 for index)')
    parser.add_argument('--volatility', type=float, default=None, help='Volatility factor (default: 0.015 for equity, 0.01 for index)')
    parser.add_argument('--trend', type=float, default=0.0002, help='Trend factor (positive=uptrend, negative=downtrend)')
    parser.add_argument('--filename', type=str, default=None, help='Output filename')
    parser.add_argument('--intraday', action='store_true', help='Generate intraday (minute) data instead of daily')
    
    parsed_args = parser.parse_args(args)
    
    # Set default filename if not provided
    if parsed_args.filename is None:
        timeframe = "intraday" if parsed_args.intraday else "daily"
        parsed_args.filename = f"{parsed_args.symbol}_{parsed_args.type}_{timeframe}.csv"
    
    # Set default prices based on data type
    if parsed_args.price is None:
        if parsed_args.type == 'equity':
            parsed_args.price = 1000.0
        elif parsed_args.type == 'index':
            parsed_args.price = 20000.0
    
    # Set default volatility based on data type
    if parsed_args.volatility is None:
        if parsed_args.type == 'equity':
            parsed_args.volatility = 0.015
        elif parsed_args.type == 'index':
            parsed_args.volatility = 0.01
    
    # Generate data based on type
    if parsed_args.type == 'equity':
        generate_equity_data(
            symbol=parsed_args.symbol,
            days=parsed_args.days,
            start_price=parsed_args.price,
            volatility=parsed_args.volatility,
            trend=parsed_args.trend,
            filename=parsed_args.filename,
            market_hours=parsed_args.intraday
        )
    elif parsed_args.type == 'index':
        generate_index_data(
            symbol=parsed_args.symbol,
            days=parsed_args.days,
            start_price=parsed_args.price,
            volatility=parsed_args.volatility,
            trend=parsed_args.trend,
            filename=parsed_args.filename
        )
    elif parsed_args.type == 'option':
        print("Option chain generation requires an underlying price and expiry date.")
        print("Use the generate_option_chain function directly for more control.")
        
        # Example usage:
        # expiry_date = datetime.now() + timedelta(days=30)
        # option_chain = generate_option_chain(
        #     underlying_price=parsed_args.price,
        #     symbol=parsed_args.symbol,
        #     expiry_date=expiry_date
        # )
        # option_chain.to_csv(f"data/generated/{parsed_args.filename}", index=False)


if __name__ == "__main__":
    main()