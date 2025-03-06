# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:36:56 2025

@author: mahes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Stock Mapping Generator

This script focuses specifically on NSE equity stocks and options,
generating optimized mapping files for the Gann trading system.
"""

import os
import sys
import json
import argparse
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

# Setup logging
def setup_logging(level=logging.INFO):
    """Configure logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/stock_mappings.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("stock_mappings")

class StockMappingGenerator:
    """
    Generates mappings specifically for NSE equity and options
    """
    
    def __init__(self, cache_dir: str = 'cache', output_dir: str = 'data/mappings'):
        """
        Initialize the mapping generator
        
        Args:
            cache_dir: Directory containing security master CSV files
            output_dir: Directory to save the generated JSON files
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.logger = setup_logging()
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize mapping dictionaries
        self.nse_stocks = {}  # ShortName -> Token
        self.nse_options = {}  # InstrumentName -> Token
        self.token_details = {}  # Token -> Details
        self.expiry_dates = {}  # Underlying -> List[ExpiryDate]
        self.option_chains = {}  # Underlying -> ExpiryDate -> OptionType -> Strike -> Token
        
        # Important symbols to process option chains for
        self.major_underlyings = [
            'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCAP', 'RELIANCE', 'TCS', 
            'INFY', 'HDFCBANK', 'SBIN', 'TATASTEEL'
        ]
    
    def process_nse_stocks(self) -> bool:
        """
        Process NSE equity stocks from NSEScripMaster.csv
        
        Returns:
            True if successful, False otherwise
        """
        file_path = self.cache_dir / 'NSEScripMaster.csv'
        
        if not file_path.exists():
            self.logger.error(f"NSE stock master file not found: {file_path}")
            return False
            
        try:
            self.logger.info(f"Processing NSE stocks from {file_path}")
            df = pd.read_csv(file_path)
            
            # Log column names
            self.logger.info(f"Columns in NSEScripMaster.csv: {', '.join(df.columns)}")
            
            # Check for required columns
            required_columns = ['Token', 'ShortName']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                return False
            
            # Get unique series values if available
            if 'Series' in df.columns:
                series_values = df['Series'].dropna().unique()
                self.logger.info(f"Series values in NSE stocks: {', '.join([str(x) for x in series_values])}")
            
            # Process each stock
            count = 0
            for _, row in df.iterrows():
                try:
                    # Skip rows without token or shortname
                    if pd.isna(row['Token']) or pd.isna(row['ShortName']):
                        continue
                    
                    token = str(row['Token']).strip()
                    symbol = str(row['ShortName']).strip()
                    
                    # Skip non-equity series if filter needed (uncomment to enable)
                    # if 'Series' in df.columns and str(row['Series']) not in ['EQ', 'BE']:
                    #     continue
                    
                    # Skip if not permitted to trade (if that column exists)
                    if 'PermittedToTrade' in df.columns and not pd.isna(row['PermittedToTrade']):
                        if str(row['PermittedToTrade']) != 'Y':
                            continue
                    
                    # Get company name
                    company_name = symbol
                    if 'CompanyName' in df.columns and not pd.isna(row['CompanyName']):
                        company_name = str(row['CompanyName']).strip()
                    
                    # Store the stock mapping
                    self.nse_stocks[symbol] = token
                    
                    # Store token details
                    self.token_details[token] = {
                        'symbol': symbol,
                        'name': company_name,
                        'exchange': 'NSE',
                        'type': 'EQUITY',
                        'series': str(row['Series']) if 'Series' in df.columns and not pd.isna(row['Series']) else 'EQ'
                    }
                    
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Error processing stock row: {e}")
                    continue
            
            self.logger.info(f"Processed {count} NSE stocks")
            return True
        except Exception as e:
            self.logger.error(f"Error processing NSE stocks: {e}")
            return False
    
    def process_nse_options(self) -> bool:
        """
        Process NSE options from FONSEScripMaster.csv
        
        Returns:
            True if successful, False otherwise
        """
        file_path = self.cache_dir / 'FONSEScripMaster.csv'
        
        if not file_path.exists():
            self.logger.error(f"NSE options master file not found: {file_path}")
            return False
            
        try:
            self.logger.info(f"Processing NSE options from {file_path}")
            df = pd.read_csv(file_path)
            
            # Log column names
            self.logger.info(f"Columns in FONSEScripMaster.csv: {', '.join(df.columns)}")
            
            # Check for required columns
            required_columns = ['Token', 'InstrumentName']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                return False
            
            # Process each option/future
            count = 0
            expiry_map = {}  # underlying -> set of expiry dates
            
            # Print a few sample instrument names for analysis
            sample_instruments = df['InstrumentName'].dropna().head(10).tolist()
            self.logger.info(f"Sample instrument names: {', '.join(sample_instruments)}")
            
            for _, row in df.iterrows():
                try:
                    # Skip rows without token or instrument name
                    if pd.isna(row['Token']) or pd.isna(row['InstrumentName']):
                        continue
                    
                    token = str(row['Token']).strip()
                    instrument = str(row['InstrumentName']).strip()
                    
                    # Extract the underlying symbol
                    underlying = None
                    
                    # For futures (e.g., NIFTYFUT)
                    if 'FUT' in instrument:
                        underlying = instrument.split('FUT')[0].strip()
                    
                    # For options, check for known underlyings
                    elif 'CE' in instrument or 'PE' in instrument:
                        for candidate in self.major_underlyings:
                            if instrument.startswith(candidate):
                                underlying = candidate
                                break
                        
                        # If still not found, try other methods
                        if not underlying:
                            # Try to find month codes
                            month_codes = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                            
                            for month in month_codes:
                                if month in instrument:
                                    parts = instrument.split(month)
                                    if parts and len(parts) > 0:
                                        underlying = parts[0].strip()
                                        # Remove any trailing digits (year)
                                        underlying = ''.join([c for c in underlying if not c.isdigit()])
                                        break
                    
                    # If still not found, use AssetName if available
                    if not underlying and 'AssetName' in df.columns and not pd.isna(row['AssetName']):
                        underlying = str(row['AssetName']).strip()
                    
                    # If still not found, use ShortName as fallback
                    if not underlying and 'ShortName' in df.columns and not pd.isna(row['ShortName']):
                        underlying = str(row['ShortName']).strip()
                    
                    # Last resort: just use the instrument name
                    if not underlying:
                        underlying = instrument
                    
                    # Get option details
                    option_type = None
                    strike_price = None
                    expiry_date = None
                    
                    if 'OptionType' in df.columns and not pd.isna(row['OptionType']):
                        option_type = str(row['OptionType']).strip()
                    
                    if 'StrikePrice' in df.columns and not pd.isna(row['StrikePrice']):
                        try:
                            strike_price = float(row['StrikePrice'])
                        except:
                            pass
                    
                    if 'ExpiryDate' in df.columns and not pd.isna(row['ExpiryDate']):
                        expiry_date = str(row['ExpiryDate']).strip()
                    
                    # Store the option mapping
                    self.nse_options[instrument] = token
                    
                    # Store token details
                    details = {
                        'symbol': instrument,
                        'underlying': underlying,
                        'exchange': 'NSE',
                        'type': 'OPTION' if option_type else 'FUTURE',
                    }
                    
                    if option_type:
                        details['option_type'] = option_type
                    
                    if strike_price:
                        details['strike_price'] = strike_price
                    
                    if expiry_date:
                        details['expiry_date'] = expiry_date
                    
                    self.token_details[token] = details
                    
                    # Track expiry dates for each underlying
                    if underlying and expiry_date:
                        if underlying not in expiry_map:
                            expiry_map[underlying] = set()
                        expiry_map[underlying].add(expiry_date)
                        
                        # Store option chain data
                        if underlying in self.major_underlyings and option_type and strike_price:
                            if underlying not in self.option_chains:
                                self.option_chains[underlying] = {}
                            
                            if expiry_date not in self.option_chains[underlying]:
                                self.option_chains[underlying][expiry_date] = {
                                    'CE': {},
                                    'PE': {}
                                }
                            
                            # Store the option in the chain
                            self.option_chains[underlying][expiry_date][option_type][str(strike_price)] = token
                    
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Error processing option row: {e}")
                    continue
            
            # Convert expiry sets to sorted lists
            for underlying, dates in expiry_map.items():
                self.expiry_dates[underlying] = sorted(list(dates))
            
            self.logger.info(f"Processed {count} NSE options/futures")
            self.logger.info(f"Found expiry dates for {len(expiry_map)} underlyings")
            self.logger.info(f"Created option chains for {len(self.option_chains)} underlyings")
            
            return True
        except Exception as e:
            self.logger.error(f"Error processing NSE options: {e}")
            return False
    
    def save_mappings(self) -> bool:
        """
        Save mappings to JSON files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define output files
            files = {
                'nse_stocks': self.output_dir / 'nse_stocks.json',
                'nse_options': self.output_dir / 'nse_options.json',
                'token_details': self.output_dir / 'token_details.json',
                'expiry_dates': self.output_dir / 'expiry_dates.json',
                'option_chains': self.output_dir / 'option_chains.json'
            }
            
            # Save each mapping
            with open(files['nse_stocks'], 'w') as f:
                json.dump(self.nse_stocks, f, indent=2)
                self.logger.info(f"Saved {len(self.nse_stocks)} NSE stocks to {files['nse_stocks']}")
            
            with open(files['nse_options'], 'w') as f:
                json.dump(self.nse_options, f, indent=2)
                self.logger.info(f"Saved {len(self.nse_options)} NSE options to {files['nse_options']}")
            
            with open(files['token_details'], 'w') as f:
                json.dump(self.token_details, f, indent=2)
                self.logger.info(f"Saved {len(self.token_details)} token details to {files['token_details']}")
            
            with open(files['expiry_dates'], 'w') as f:
                json.dump(self.expiry_dates, f, indent=2)
                self.logger.info(f"Saved expiry dates for {len(self.expiry_dates)} underlyings to {files['expiry_dates']}")
            
            with open(files['option_chains'], 'w') as f:
                json.dump(self.option_chains, f, indent=2)
                self.logger.info(f"Saved option chains for {len(self.option_chains)} underlyings to {files['option_chains']}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving mappings: {e}")
            return False
    
    def generate_mappings(self) -> bool:
        """
        Generate all mappings
        
        Returns:
            True if successful, False otherwise
        """
        # Process NSE stocks
        stocks_success = self.process_nse_stocks()
        
        # Process NSE options
        options_success = self.process_nse_options()
        
        # Save mappings
        if stocks_success or options_success:
            return self.save_mappings()
        else:
            self.logger.error("Both stock and option processing failed")
            return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate NSE stock and option mappings')
    parser.add_argument('--cache-dir', type=str, default='cache',
                      help='Directory containing security master CSV files')
    parser.add_argument('--output-dir', type=str, default='data/mappings',
                      help='Directory to save the generated JSON files')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize generator
    generator = StockMappingGenerator(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )
    
    # Generate mappings
    success = generator.generate_mappings()
    
    print("\n" + "="*80)
    print("MAPPING GENERATION SUMMARY")
    print("="*80)
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"NSE Stocks: {len(generator.nse_stocks)}")
    print(f"NSE Options: {len(generator.nse_options)}")
    print(f"Token Details: {len(generator.token_details)}")
    print(f"Expiry Dates: {len(generator.expiry_dates)}")
    print(f"Option Chains: {len(generator.option_chains)}")
    print("="*80)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())