# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 08:58:07 2025

@author: mahes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Master Mapping Generator

This script processes the security master files and generates optimized
mapping files for quick lookups during trading.

It creates the following JSON files:
1. nse_equity_symbols.json - NSE equity symbol to token mappings
2. nse_options_symbols.json - NSE options instrument to token mappings
3. bse_equity_symbols.json - BSE equity symbol to token mappings
4. bse_options_symbols.json - BSE options instrument to token mappings
5. token_details.json - Token to details mappings
6. expiry_dates.json - Available expiry dates for derivatives
7. option_chains.json - Pre-computed option chains for major underlyings
"""

import os
import sys
import json
import argparse  # Add this import
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
            logging.FileHandler("logs/generate_mappings.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("security_mappings")

class SecurityMappingGenerator:
    """
    Generates optimized security mappings from security master files
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
        self.nse_equity_symbols = {}
        self.nse_options_symbols = {}
        self.bse_equity_symbols = {}
        self.bse_options_symbols = {}
        self.token_details = {}
        self.expiry_dates = {}
        self.option_chains = {}
        
        # Important symbols to process option chains for
        self.major_underlyings = [
            'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'RELIANCE', 'TCS', 
            'INFY', 'HDFCBANK', 'SBIN', 'TATASTEEL'
        ]
        
    def load_security_master_files(self) -> bool:
        """
        Load all security master CSV files from cache directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Loading security master files from cache...")
            
            # Define expected files
            files = {
                'nse_equity': self.cache_dir / 'NSEScripMaster.csv',
                'bse_equity': self.cache_dir / 'BSEScripMaster.csv',
                'nse_options': self.cache_dir / 'FONSEScripMaster.csv',
                'bse_options': self.cache_dir / 'FOBSEScripMaster.csv'
            }
            
            # Check file existence
            missing_files = []
            for name, path in files.items():
                if not path.exists():
                    missing_files.append(name)
            
            if missing_files:
                self.logger.error(f"Missing security master files: {', '.join(missing_files)}")
                return False
            
            # Process each file
            self.process_nse_equity(files['nse_equity'])
            self.process_bse_equity(files['bse_equity'])
            self.process_nse_options(files['nse_options'])
            self.process_bse_options(files['bse_options'])
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading security master files: {e}")
            return False
    
    def process_nse_equity(self, file_path: Path) -> None:
        """
        Process NSE equity security master file
        
        Args:
            file_path: Path to NSE equity CSV file
        """
        try:
            self.logger.info(f"Processing NSE equity file: {file_path}")
            df = pd.read_csv(file_path)
            
            # For NSE equity: "Token","ShortName","Series","CompanyName","ExchangeCode"
            count = 0
            
            for _, row in df.iterrows():
                try:
                    # Skip non-equity series
                    if 'Series' in df.columns and str(row['Series']) not in ['EQ', 'BE', 'N']:
                        continue
                    
                    token = str(row['Token']).strip()
                    
                    # Use ShortName as symbol
                    if 'ShortName' in df.columns and not pd.isna(row['ShortName']):
                        symbol = str(row['ShortName']).strip()
                    else:
                        continue
                    
                    # Store in mappings
                    self.nse_equity_symbols[symbol] = token
                    
                    # Store details
                    company_name = str(row['CompanyName']).strip() if 'CompanyName' in df.columns and not pd.isna(row['CompanyName']) else symbol
                    
                    self.token_details[token] = {
                        'symbol': symbol,
                        'name': company_name,
                        'exchange': 'NSE',
                        'type': 'EQUITY',
                        'series': str(row['Series']) if 'Series' in df.columns and not pd.isna(row['Series']) else None
                    }
                    
                    count += 1
                except Exception as e:
                    continue
            
            self.logger.info(f"Processed {count} NSE equity symbols")
        except Exception as e:
            self.logger.error(f"Error processing NSE equity file: {e}")
    
    def process_bse_equity(self, file_path: Path) -> None:
        """
        Process BSE equity security master file
        
        Args:
            file_path: Path to BSE equity CSV file
        """
        try:
            self.logger.info(f"Processing BSE equity file: {file_path}")
            df = pd.read_csv(file_path)
            
            # For BSE equity: "Token","ShortName","Series","CompanyName","ExchangeCode"
            count = 0
            
            for _, row in df.iterrows():
                try:
                    token = str(row['Token']).strip()
                    
                    # Use ShortName or ScripName as symbol
                    if 'ShortName' in df.columns and not pd.isna(row['ShortName']):
                        symbol = str(row['ShortName']).strip()
                    elif 'ScripName' in df.columns and not pd.isna(row['ScripName']):
                        symbol = str(row['ScripName']).strip()
                    else:
                        continue
                    
                    # Store in mappings
                    self.bse_equity_symbols[symbol] = token
                    
                    # Store details
                    company_name = str(row['CompanyName']).strip() if 'CompanyName' in df.columns and not pd.isna(row['CompanyName']) else symbol
                    
                    self.token_details[token] = {
                        'symbol': symbol,
                        'name': company_name,
                        'exchange': 'BSE',
                        'type': 'EQUITY',
                        'series': str(row['Series']) if 'Series' in df.columns and not pd.isna(row['Series']) else None
                    }
                    
                    count += 1
                except Exception as e:
                    continue
            
            self.logger.info(f"Processed {count} BSE equity symbols")
        except Exception as e:
            self.logger.error(f"Error processing BSE equity file: {e}")
    
    def process_nse_options(self, file_path: Path) -> None:
        """
        Process NSE futures & options security master file
        
        Args:
            file_path: Path to NSE F&O CSV file
        """
        try:
            self.logger.info(f"Processing NSE options file: {file_path}")
            df = pd.read_csv(file_path)
            
            # For NSE options: "Token","InstrumentName","ShortName","Series","ExpiryDate","StrikePrice","OptionType"
            count = 0
            expiry_map = {}  # Store expiry dates for each underlying
            
            for _, row in df.iterrows():
                try:
                    token = str(row['Token']).strip()
                    
                    # Get instrument name and underlying
                    if 'InstrumentName' in df.columns and not pd.isna(row['InstrumentName']):
                        instrument = str(row['InstrumentName']).strip()
                    else:
                        continue
                    
                    # Extract the underlying symbol from the instrument name
                    underlying = instrument
                    if underlying.endswith('FUT'):
                        underlying = underlying[:-3]
                    
                    # Get option details if available
                    option_type = None
                    strike_price = None
                    expiry_date = None
                    
                    if 'OptionType' in df.columns and not pd.isna(row['OptionType']):
                        option_type = str(row['OptionType']).strip()
                    
                    if 'StrikePrice' in df.columns and not pd.isna(row['StrikePrice']):
                        try:
                            strike_price = float(row['StrikePrice'])
                        except:
                            strike_price = None
                    
                    if 'ExpiryDate' in df.columns and not pd.isna(row['ExpiryDate']):
                        expiry_date = str(row['ExpiryDate']).strip()
                        
                    # For F&O storage, include expiry date and option type in key
                    key = instrument
                    if expiry_date:
                        # Store mapping
                        if underlying not in expiry_map:
                            expiry_map[underlying] = set()
                        expiry_map[underlying].add(expiry_date)
                        
                        # Create keys for specific option contract lookups
                        if option_type and strike_price:
                            specific_key = f"{underlying}|{expiry_date}|{strike_price}|{option_type}"
                            self.nse_options_symbols[specific_key] = token
                    
                    # Store general instrument mapping
                    self.nse_options_symbols[key] = token
                    
                    # Store details
                    self.token_details[token] = {
                        'symbol': instrument,
                        'underlying': underlying,
                        'exchange': 'NSE',
                        'type': 'OPTION' if option_type else 'FUTURE',
                        'option_type': option_type,
                        'strike_price': strike_price,
                        'expiry_date': expiry_date
                    }
                    
                    count += 1
                    
                    # If this is a major underlying and has option details,
                    # store it in option chains for quick lookup
                    if underlying in self.major_underlyings and option_type and strike_price and expiry_date:
                        if underlying not in self.option_chains:
                            self.option_chains[underlying] = {}
                        
                        if expiry_date not in self.option_chains[underlying]:
                            self.option_chains[underlying][expiry_date] = {
                                'CE': {},
                                'PE': {}
                            }
                        
                        self.option_chains[underlying][expiry_date][option_type][strike_price] = token
                    
                except Exception as e:
                    continue
            
            # Convert expiry sets to sorted lists and store in expiry_dates
            for underlying, dates in expiry_map.items():
                self.expiry_dates[underlying] = sorted(list(dates))
            
            self.logger.info(f"Processed {count} NSE options symbols")
            self.logger.info(f"Found expiry dates for {len(expiry_map)} underlyings")
            self.logger.info(f"Created option chains for {len(self.option_chains)} major underlyings")
        except Exception as e:
            self.logger.error(f"Error processing NSE options file: {e}")
    
    def process_bse_options(self, file_path: Path) -> None:
        """
        Process BSE futures & options security master file
        
        Args:
            file_path: Path to BSE F&O CSV file
        """
        try:
            self.logger.info(f"Processing BSE options file: {file_path}")
            df = pd.read_csv(file_path)
            
            # For BSE options: "Token","InstrumentName","ShortName","Series","ExpiryDate","StrikePrice","OptionType"
            count = 0
            
            for _, row in df.iterrows():
                try:
                    token = str(row['Token']).strip()
                    
                    # Get instrument name
                    if 'InstrumentName' in df.columns and not pd.isna(row['InstrumentName']):
                        instrument = str(row['InstrumentName']).strip()
                    else:
                        continue
                    
                    # Store in mappings
                    self.bse_options_symbols[instrument] = token
                    
                    # Extract the underlying symbol from the instrument name
                    underlying = instrument
                    if underlying.endswith('FUT'):
                        underlying = underlying[:-3]
                    
                    # Get option details if available
                    option_type = None
                    strike_price = None
                    expiry_date = None
                    
                    if 'OptionType' in df.columns and not pd.isna(row['OptionType']):
                        option_type = str(row['OptionType']).strip()
                    
                    if 'StrikePrice' in df.columns and not pd.isna(row['StrikePrice']):
                        try:
                            strike_price = float(row['StrikePrice'])
                        except:
                            strike_price = None
                    
                    if 'ExpiryDate' in df.columns and not pd.isna(row['ExpiryDate']):
                        expiry_date = str(row['ExpiryDate']).strip()
                    
                    # Store details
                    self.token_details[token] = {
                        'symbol': instrument,
                        'underlying': underlying,
                        'exchange': 'BSE',
                        'type': 'OPTION' if option_type else 'FUTURE',
                        'option_type': option_type,
                        'strike_price': strike_price,
                        'expiry_date': expiry_date
                    }
                    
                    count += 1
                except Exception as e:
                    continue
            
            self.logger.info(f"Processed {count} BSE options symbols")
        except Exception as e:
            self.logger.error(f"Error processing BSE options file: {e}")
    
    def save_mappings(self) -> bool:
        """
        Save all mappings to JSON files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Saving mappings to JSON files...")
            
            # Define files to save
            files = {
                'nse_equity': self.output_dir / 'nse_equity_symbols.json',
                'nse_options': self.output_dir / 'nse_options_symbols.json',
                'bse_equity': self.output_dir / 'bse_equity_symbols.json',
                'bse_options': self.output_dir / 'bse_options_symbols.json',
                'token_details': self.output_dir / 'token_details.json',
                'expiry_dates': self.output_dir / 'expiry_dates.json',
                'option_chains': self.output_dir / 'option_chains.json'
            }
            
            # Save each mapping
            with open(files['nse_equity'], 'w') as f:
                json.dump(self.nse_equity_symbols, f, indent=2)
                self.logger.info(f"Saved {len(self.nse_equity_symbols)} NSE equity symbols to {files['nse_equity']}")
            
            with open(files['nse_options'], 'w') as f:
                json.dump(self.nse_options_symbols, f, indent=2)
                self.logger.info(f"Saved {len(self.nse_options_symbols)} NSE options symbols to {files['nse_options']}")
            
            with open(files['bse_equity'], 'w') as f:
                json.dump(self.bse_equity_symbols, f, indent=2)
                self.logger.info(f"Saved {len(self.bse_equity_symbols)} BSE equity symbols to {files['bse_equity']}")
            
            with open(files['bse_options'], 'w') as f:
                json.dump(self.bse_options_symbols, f, indent=2)
                self.logger.info(f"Saved {len(self.bse_options_symbols)} BSE options symbols to {files['bse_options']}")
            
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
        Generate all mappings from security master files
        
        Returns:
            True if successful, False otherwise
        """
        if not self.load_security_master_files():
            return False
        
        return self.save_mappings()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate security mappings')
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
    generator = SecurityMappingGenerator(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )
    
    # Generate mappings
    success = generator.generate_mappings()
    
    print("\n" + "="*80)
    print("MAPPING GENERATION SUMMARY")
    print("="*80)
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"NSE Equity Symbols: {len(generator.nse_equity_symbols)}")
    print(f"NSE Options Symbols: {len(generator.nse_options_symbols)}")
    print(f"BSE Equity Symbols: {len(generator.bse_equity_symbols)}")
    print(f"BSE Options Symbols: {len(generator.bse_options_symbols)}")
    print(f"Token Details: {len(generator.token_details)}")
    print(f"Expiry Dates: {len(generator.expiry_dates)}")
    print(f"Option Chains: {len(generator.option_chains)}")
    print("="*80)
    
    return 0 if success else 1

if __name__ == "__main__":  # Fix this line
    sys.exit(main())