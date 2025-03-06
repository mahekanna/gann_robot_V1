# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:38:46 2025

@author: mahes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Lookup Client for Gann Trading System

Provides a simple interface to look up NSE stocks and options
based on pre-generated mapping files.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

class StockLookup:
    """
    Client for looking up NSE stocks and options
    """
    
    def __init__(self, mappings_dir: str = 'data/mappings', log_level: int = logging.INFO):
        """
        Initialize the lookup client
        
        Args:
            mappings_dir: Directory containing mapping files
            log_level: Logging level
        """
        self.mappings_dir = Path(mappings_dir)
        
        # Setup logging
        self.logger = logging.getLogger("stock_lookup")
        self.logger.setLevel(log_level)
        
        # Add handlers if none exist
        if not self.logger.handlers:
            # Create logs directory
            Path('logs').mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler("logs/stock_lookup.log")
            file_handler.setLevel(log_level)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # Initialize data
        self.nse_stocks = {}
        self.nse_options = {}
        self.token_details = {}
        self.expiry_dates = {}
        self.option_chains = {}
        
        # Load mappings
        self.load_mappings()
    
    def load_mappings(self) -> bool:
        """
        Load mapping files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define mapping files
            files = {
                'nse_stocks': self.mappings_dir / 'nse_stocks.json',
                'nse_options': self.mappings_dir / 'nse_options.json',
                'token_details': self.mappings_dir / 'token_details.json',
                'expiry_dates': self.mappings_dir / 'expiry_dates.json',
                'option_chains': self.mappings_dir / 'option_chains.json'
            }
            
            # Check if mapping files exist
            missing_files = []
            for name, path in files.items():
                if not path.exists():
                    missing_files.append(name)
            
            if missing_files:
                self.logger.error(f"Missing mapping files: {', '.join(missing_files)}")
                return False
            
            # Load each mapping file
            with open(files['nse_stocks']) as f:
                self.nse_stocks = json.load(f)
                self.logger.info(f"Loaded {len(self.nse_stocks)} NSE stocks")
            
            with open(files['nse_options']) as f:
                self.nse_options = json.load(f)
                self.logger.info(f"Loaded {len(self.nse_options)} NSE options")
            
            with open(files['token_details']) as f:
                self.token_details = json.load(f)
                self.logger.info(f"Loaded {len(self.token_details)} token details")
            
            with open(files['expiry_dates']) as f:
                self.expiry_dates = json.load(f)
                self.logger.info(f"Loaded expiry dates for {len(self.expiry_dates)} underlyings")
            
            with open(files['option_chains']) as f:
                self.option_chains = json.load(f)
                self.logger.info(f"Loaded option chains for {len(self.option_chains)} underlyings")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading mappings: {e}")
            return False
    
    def get_stock_token(self, symbol: str) -> Optional[str]:
        """
        Get token for a stock symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Token or None if not found
        """
        # Direct lookup
        if symbol in self.nse_stocks:
            return self.nse_stocks[symbol]
        
        # Case-insensitive lookup
        for s, token in self.nse_stocks.items():
            if s.upper() == symbol.upper():
                return token
        
        self.logger.warning(f"Stock token not found: {symbol}")
        return None
    
    def get_option_token(self, instrument: str) -> Optional[str]:
        """
        Get token for an option/future instrument
        
        Args:
            instrument: Option/future instrument name
            
        Returns:
            Token or None if not found
        """
        # Direct lookup
        if instrument in self.nse_options:
            return self.nse_options[instrument]
        
        # Case-insensitive lookup
        for i, token in self.nse_options.items():
            if i.upper() == instrument.upper():
                return token
        
        self.logger.warning(f"Option token not found: {instrument}")
        return None
    
    def get_token_details(self, token: str) -> Optional[Dict]:
        """
        Get details for a token
        
        Args:
            token: Token
            
        Returns:
            Details or None if not found
        """
        return self.token_details.get(token)
    
    def get_expiry_dates(self, underlying: str) -> List[str]:
        """
        Get expiry dates for an underlying
        
        Args:
            underlying: Underlying symbol
            
        Returns:
            List of expiry dates
        """
        # Direct lookup
        if underlying in self.expiry_dates:
            return self.expiry_dates[underlying]
        
        # Case-insensitive lookup
        for u, dates in self.expiry_dates.items():
            if u.upper() == underlying.upper():
                return dates
        
        return []
    
    def get_option_chain(self, underlying: str, expiry_date: str) -> Dict:
        """
        Get option chain for an underlying and expiry date
        
        Args:
            underlying: Underlying symbol
            expiry_date: Expiry date
            
        Returns:
            Option chain as {option_type: {strike: token}}
        """
        # Direct lookup
        if underlying in self.option_chains and expiry_date in self.option_chains[underlying]:
            return self.option_chains[underlying][expiry_date]
        
        # Case-insensitive lookup
        for u, expirations in self.option_chains.items():
            if u.upper() == underlying.upper():
                for e, chain in expirations.items():
                    if e == expiry_date:
                        return chain
        
        return {'CE': {}, 'PE': {}}
    
    def get_option_contract(self, underlying: str, expiry_date: str, 
                          strike_price: float, option_type: str) -> Optional[str]:
        """
        Get token for a specific option contract
        
        Args:
            underlying: Underlying symbol
            expiry_date: Expiry date
            strike_price: Strike price
            option_type: 'CE' or 'PE'
            
        Returns:
            Token or None if not found
        """
        chain = self.get_option_chain(underlying, expiry_date)
        
        if option_type in chain and str(strike_price) in chain[option_type]:
            return chain[option_type][str(strike_price)]
        
        return None
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks matching a query
        
        Args:
            query: Search query
            
        Returns:
            List of matching stocks
        """
        results = []
        query_lower = query.lower()
        
        for symbol, token in self.nse_stocks.items():
            if query_lower in symbol.lower():
                if token in self.token_details:
                    results.append(self.token_details[token])
        
        return results
    
    def get_nearest_strikes(self, underlying: str, expiry_date: str, 
                          current_price: float, count: int = 5) -> Dict[str, List[float]]:
        """
        Get nearest strike prices around current price
        
        Args:
            underlying: Underlying symbol
            expiry_date: Expiry date
            current_price: Current price
            count: Number of strikes above and below
            
        Returns:
            Dict with 'above' and 'below' strikes
        """
        chain = self.get_option_chain(underlying, expiry_date)
        
        # Collect all strikes from CE and PE
        all_strikes = set()
        for option_type in ['CE', 'PE']:
            if option_type in chain:
                for strike_str in chain[option_type].keys():
                    try:
                        all_strikes.add(float(strike_str))
                    except:
                        pass
        
        # Sort strikes
        strikes = sorted(all_strikes)
        
        if not strikes:
            return {'above': [], 'below': []}
        
        # Find strikes above and below current price
        above = [strike for strike in strikes if strike > current_price]
        below = [strike for strike in strikes if strike <= current_price]
        
        # Sort and limit
        above = sorted(above)[:count]
        below = sorted(below, reverse=True)[:count]
        
        return {'above': above, 'below': below}
    
    def get_atm_strike(self, underlying: str, expiry_date: str, 
                     current_price: float) -> Optional[float]:
        """
        Get at-the-money strike price
        
        Args:
            underlying: Underlying symbol
            expiry_date: Expiry date
            current_price: Current price
            
        Returns:
            ATM strike or None if not found
        """
        chain = self.get_option_chain(underlying, expiry_date)
        
        # Collect all strikes
        all_strikes = set()
        for option_type in ['CE', 'PE']:
            if option_type in chain:
                for strike_str in chain[option_type].keys():
                    try:
                        all_strikes.add(float(strike_str))
                    except:
                        pass
        
        if not all_strikes:
            return None
        
        # Find closest strike to current price
        return min(all_strikes, key=lambda x: abs(x - current_price))