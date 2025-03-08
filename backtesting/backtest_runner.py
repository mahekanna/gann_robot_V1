# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 00:16:02 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Backtest Runner for Gann Trading System
This script runs backtests for various strategies using historical data.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from core.gann.square_of_9 import GannSquareOf9
from core.strategy.base_strategy import TimeFrame, SignalType
from core.strategy.equity_strategy import EquityStrategy
from core.strategy.index_strategy import IndexStrategy
from backtesting.engine import BacktestEngine
from utils.logger import setup_logger
from main import generate_equity_data, generate_index_data, generate_option_chain


def load_data(symbol: str, 
             timeframe: TimeFrame, 
             start_date: str, 
             end_date: str, 
             data_dir: str = 'data/historical',
             data_type: str = 'equity') -> pd.DataFrame:
    """
    Load historical data for backtesting
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for the data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing data files
        data_type: Type of data ('equity' or 'index')
        
    Returns:
        DataFrame with OHLCV data
    """
    # Create path for data file
    file_path = Path(data_dir) / f"{symbol}_{timeframe.value}.csv"
    
    # Check if file exists
    if not file_path.exists():
        # Try to find any file with the symbol
        potential_files = list(Path(data_dir).glob(f"{symbol}_*.csv"))
        if potential_files:
            file_path = potential_files[0]
            logging.info(f"Using available data file: {file_path}")
        else:
            # If no data file exists, use the synthetic data generator
            logging.info(f"No data file found for {symbol}. Generating synthetic data.")
            
            # Parse dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days
            
            # Generate data based on type
            if data_type.lower() == 'equity':
                df = generate_equity_data(
                    symbol=symbol,
                    days=days + 10,  # Add some extra days for safety
                    start_price=1000.0,  # Default starting price
                    volatility=0.015,
                    trend=0.0002  # Slight upward bias
                )
            elif data_type.lower() == 'index':
                df = generate_index_data(
                    symbol=symbol,
                    days=days + 10,
                    start_price=20000.0 if symbol in ['NIFTY', 'BANKNIFTY'] else 1000.0,
                    volatility=0.01,
                    trend=0.0001  # Slight upward bias
                )
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Ensure data directory exists
            os.makedirs(data_dir, exist_ok=True)
            
            # Save generated data
            file_path = Path(data_dir) / f"{symbol}_{data_type}_{timeframe.value}.csv"
            df.to_csv(file_path, index=False)
            
            # Filter by date range
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
            
            return df
    
    # Load data from file
    df = pd.read_csv(file_path)
    
    # Convert datetime to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter by date range if provided
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    return df


def generate_option_data(underlying_df: pd.DataFrame, symbol: str, expiry_days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate option chain data for backtesting
    
    Args:
        underlying_df: DataFrame with underlying price data
        symbol: Underlying symbol
        expiry_days: Days to expiry
        
    Returns:
        Dictionary with date -> option chain mapping
    """
    option_chains = {}
    
    # Generate option chain for each day in the data
    for i, row in underlying_df.iterrows():
        current_date = row['datetime']
        current_price = row['close']
        
        # Calculate expiry date
        expiry_date = current_date + pd.Timedelta(days=expiry_days)
        
        # Generate option chain
        option_chain = generate_option_chain(
            underlying_price=current_price,
            symbol=symbol,
            expiry_date=expiry_date,
            num_strikes=10,  # 10 strikes above and below ATM
            strike_gap=None  # Auto-calculate based on price
        )
        
        # Store in dictionary
        option_chains[current_date] = option_chain
    
    return option_chains


def run_backtest(args):
    """
    Run backtest with specified parameters
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(
        f"backtest_{args.symbol}_{args.timeframe}",
        log_file=log_dir / f"backtest_{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_level=args.log_level
    )
    
    logger.info(f"Starting backtest for {args.symbol} on {args.timeframe} timeframe")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Load historical data
    timeframe = TimeFrame(args.timeframe)
    data = load_data(
        symbol=args.symbol,
        timeframe=timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        data_type=args.strategy
    )
    
    if data.empty:
        logger.error("No data available for the specified parameters")
        return
    
    logger.info(f"Loaded {len(data)} bars of historical data")
    
    # Generate option data if needed
    option_chains = None
    if args.use_options:
        logger.info("Generating option chain data for backtesting")
        option_chains = generate_option_data(
            underlying_df=data,
            symbol=args.symbol,
            expiry_days=args.expiry_days
        )
        logger.info(f"Generated option chains for {len(option_chains)} days")
    
    # Create strategy
    if args.strategy == "equity":
        strategy = EquityStrategy(
            symbol=args.symbol,
            timeframe=timeframe,
            num_targets=args.targets,
            buffer_percentage=args.buffer,
            risk_percentage=args.risk,
            trailing_trigger_percentage=args.trailing_trigger,
            option_lot_size=args.lot_size,
            use_options=args.use_options
        )
    elif args.strategy == "index":
        strategy = IndexStrategy(
            symbol=args.symbol,
            timeframe=timeframe,
            num_targets=args.targets,
            buffer_percentage=args.buffer,
            risk_percentage=args.risk,
            trailing_trigger_percentage=args.trailing_trigger,
            option_lot_size=args.lot_size
        )
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Initialize backtesting engine
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_rate=args.commission,
        option_chains=option_chains
    )
    
    # Run backtest
    results = engine.run_backtest(
        strategy=strategy,
        data=data,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Check for errors
    if "error" in results:
        logger.error(f"Backtest failed: {results['error']}")
        return
    
    # Print results summary
    print("\n=== Backtest Results ===")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Save results to file
    output_dir = Path("output/backtests")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save equity curve
    equity_curve = results['equity_curve']
    equity_curve.to_csv(output_dir / f"{args.symbol}_{args.timeframe}_equity_curve.csv")
    
    # Save trades
    trades_df = pd.DataFrame(results['trades'])
    if not trades_df.empty:
        trades_df.to_csv(output_dir / f"{args.symbol}_{args.timeframe}_trades.csv", index=False)
    
    # Plot equity curve
    if args.plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve['equity'])
        plt.title(f"Equity Curve - {args.symbol} ({args.timeframe})")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(equity_curve.index, equity_curve['drawdown'] * 100)
        plt.title("Drawdown")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{args.symbol}_{args.timeframe}_equity_curve.png")
        
        if not args.no_show:
            plt.show()
        
    logger.info("Backtest completed successfully")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Gann Trading System Backtest Runner')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol')
    
    # Strategy parameters
    parser.add_argument('--strategy', type=str, default='equity', choices=['equity', 'index'], 
                      help='Strategy type')
    parser.add_argument('--timeframe', type=str, default='1day', 
                      choices=['1minute', '5minute', '15minute', '30minute', '1hour', '1day'],
                      help='Trading timeframe')
    parser.add_argument('--targets', type=int, default=3, help='Number of target levels')
    parser.add_argument('--buffer', type=float, default=0.002, help='Buffer percentage for stoploss')
    parser.add_argument('--risk', type=float, default=1.0, help='Risk percentage per trade')
    parser.add_argument('--trailing-trigger', type=float, default=0.5, 
                      help='Percentage of target to trigger trailing stop')
    parser.add_argument('--use-options', action='store_true', help='Use options for hedging/trading')
    parser.add_argument('--expiry-days', type=int, default=30, help='Days to expiry for options')
    
    # Backtest parameters
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.05, help='Commission rate (percentage)')
    parser.add_argument('--lot-size', type=int, default=1, help='Option lot size')
    
    # Data and output parameters
    parser.add_argument('--data-dir', type=str, default='data/historical', help='Data directory')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    # Logging parameters
    parser.add_argument('--log-level', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Run backtest
    run_backtest(args)


if __name__ == "__main__":
    main()