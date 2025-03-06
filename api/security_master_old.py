# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:34:26 2025

@author: mahes
"""

# File: api/security_master.py
"""
Security Master module for ICICI Direct Breeze API
Handles downloading, parsing, and caching the security master file
"""

import os
import logging
import pandas as pd
import requests
import zipfile
import io
import json
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
from datetime import datetime, timedelta

class SecurityMaster:
    """
    Security Master manager for ICICI Direct Breeze API
    Downloads and manages the security master file
    """
    
    # Security master URL (this should match what's in breeze_patch.py)
    SECURITY_MASTER_URL = "https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip"
    
    def __init__(self, 
                 cache_dir: str = 'cache',
                 cache_file: str = 'security_master.csv',
                 max_age_days: int = 1,
                 log_level: int = logging.INFO):
        """
        Initialize Security Master manager
        
        Args:
            cache_dir: Directory to store cached files
            cache_file: Filename for cached security master
            max_age_days: Maximum age of cache in days
            log_level: Logging level
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / cache_file
        self.max_age_days = max_age_days
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = logging.getLogger("security_master")
        self.logger.setLevel(log_level)
        
        # Add handlers only if none exist
        if not self.logger.handlers:
            # Create directory for logs if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / "security_master.log")
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
        
        # Security master data
        self.security_master_df = None
        self.token_symbol_map = {}
        self.symbol_token_map = {}
        
    def download_security_master(self, max_retries: int = 3) -> bool:
        """
        Download security master file, extract, and convert TXT to CSV
        
        Args:
            max_retries: Maximum number of download attempts
        
        Returns:
            True if download successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Configure requests with user agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Download the file
                response = requests.get(
                    self.SECURITY_MASTER_URL, 
                    headers=headers, 
                    verify=False,  # Disable SSL verification
                    timeout=60  # Longer timeout
                )
                
                # Check if response is successful
                if response.status_code != 200:
                    self.logger.error(f"HTTP request failed with status code {response.status_code}")
                    continue
                
                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Try to open the ZIP file
                try:
                    z = zipfile.ZipFile(io.BytesIO(response.content))
                    
                    # List all files in the ZIP
                    all_files = z.namelist()
                    self.logger.info(f"ZIP contents: {all_files}")
                    
                    # Find TXT files (since you mentioned they're .txt)
                    txt_files = [f for f in all_files if f.lower().endswith('.txt')]
                    
                    if not txt_files:
                        self.logger.error("No TXT files found in security master ZIP")
                        continue
                    
                    # Extract and convert TXT files to CSV
                    csv_files = []
                    for txt_file in txt_files:
                        # Extract the TXT file
                        z.extract(txt_file, self.cache_dir)
                        extracted_path = self.cache_dir / txt_file
                        
                        # Convert TXT to CSV
                        csv_path = extracted_path.with_suffix('.csv')
                        self._convert_txt_to_csv(extracted_path, csv_path)
                        
                        # Add to CSV files list
                        csv_files.append(csv_path)
                        
                        # Log conversion details
                        self.logger.info(f"Converted {txt_file} to {csv_path}")
                    
                    # Use the first CSV as the main security master
                    if csv_files:
                        main_csv = csv_files[0]
                        
                        # Copy to specific cache file if needed
                        if str(main_csv) != str(self.cache_file):
                            import shutil
                            shutil.copy(main_csv, self.cache_file)
                        
                        self.logger.info(f"Security master downloaded and converted successfully to {self.cache_file}")
                        return True
                    else:
                        self.logger.error("No CSV files created after conversion")
                        continue
                
                except zipfile.BadZipFile as ze:
                    self.logger.error(f"Bad ZIP file: {ze}")
                    continue
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error downloading security master: {e}")
                if attempt == max_retries - 1:
                    return False
            except Exception as e:
                self.logger.error(f"Unexpected error downloading security master: {e}")
                if attempt == max_retries - 1:
                    return False
            
            # Add a small delay between retries
            import time
            time.sleep(2)
        
        return False

    def _convert_txt_to_csv(self, txt_path: Path, csv_path: Path) -> bool:
        """
        Convert a TXT file to CSV format
        
        Args:
            txt_path: Path to the input TXT file
            csv_path: Path to the output CSV file
        
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Detect delimiter (could be tab or comma)
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                # Try to detect delimiter
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    # Default to tab if no clear delimiter
                    delimiter = '\t'
                
                # Reset file pointer
                f.seek(0)
                
                # Read all lines
                lines = f.readlines()
            
            # Write to CSV
            with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                for line in lines:
                    # Clean and write the line
                    cleaned_line = line.strip().replace('\r', '').replace('\n', '')
                    csvfile.write(cleaned_line + '\n')
            
            self.logger.info(f"Converted {txt_path} to {csv_path} using delimiter '{delimiter}'")
            return True
        
        except Exception as e:
            self.logger.error(f"Error converting {txt_path} to CSV: {e}")
            return False        
    def is_cache_valid(self) -> bool:
        """
        Check if cached security master is valid
        
        Returns:
            True if cache is valid, False otherwise
        """
        if not self.cache_file.exists():
            return False
            
        # Check if cache is too old
        file_time = datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        max_age = timedelta(days=self.max_age_days)
        
        return datetime.now() - file_time < max_age
            
    def load_security_master(self, force_download: bool = False) -> bool:
        """
        Load security master from cache or download if needed
        
        Args:
            force_download: Force download even if cache is valid
            
        Returns:
            True if loading successful, False otherwise
        """
        # Check if we need to download
        if force_download or not self.is_cache_valid():
            if not self.download_security_master():
                if not self.cache_file.exists():
                    self.logger.error("Failed to download security master and no cache available")
                    return False
                else:
                    self.logger.warning("Using outdated security master cache")
                    
        try:
            # Load security master from cache
            self.logger.info(f"Loading security master from {self.cache_file}")
            self.security_master_df = pd.read_csv(self.cache_file)
            
            # Check if file is empty or has no data
            if self.security_master_df.empty:
                self.logger.warning("Security master file is empty")
                # Try to find any other CSV files
                csv_files = list(self.cache_dir.glob("*.csv"))
                if csv_files:
                    self.logger.info(f"Trying alternate security master file: {csv_files[0]}")
                    self.security_master_df = pd.read_csv(csv_files[0])
            
            # Create token-symbol and symbol-token maps
            self._create_maps()
            
            self.logger.info(f"Security master loaded: {len(self.security_master_df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading security master: {e}")
            return False
            
    def _create_maps(self) -> None:
        """
        Create token-symbol and symbol-token maps with advanced filtering
        Handles different types of security master files
        """
        if self.security_master_df is None:
            return
        
        # Reset maps
        self.token_symbol_map = {}
        self.symbol_token_map = {}
        
        try:
            # Determine column names dynamically
            columns = self.security_master_df.columns.tolist()
            
            # Mapping of possible column names
            column_mappings = {
                'token': ['Token', 'token', 'SecurityId', 'ScripCode'],
                'symbol': ['ScripName', 'Symbol', 'symbol', 'ShortName'],
                'exchange': ['ExchangeCode', 'Exchange', 'exchange'],
                'series': ['Series', 'series'],
                'isin': ['ISINCode', 'ISIN']
            }
            
            # Find the actual column names
            def find_column(possible_names):
                return next((col for col in possible_names if col in columns), None)
            
            token_col = find_column(column_mappings['token'])
            symbol_col = find_column(column_mappings['symbol'])
            exchange_col = find_column(column_mappings['exchange'])
            series_col = find_column(column_mappings['series'])
            isin_col = find_column(column_mappings['isin'])
            
            # Validate required columns
            if not (token_col and symbol_col):
                self.logger.error(f"Required columns missing. Available columns: {columns}")
                return
            
            # Filter for active and tradable securities
            # Customize filtering based on your specific requirements
            active_mask = pd.Series([True] * len(self.security_master_df))
            
            # Example filtering (adjust based on your actual data)
            if series_col:
                # Keep only certain series (e.g., EQ for equity)
                active_mask &= self.security_master_df[series_col].isin(['EQ', 'N', 'BE'])
            
            # Convert to string and remove any whitespace
            filtered_df = self.security_master_df[active_mask].copy()
            filtered_df[token_col] = filtered_df[token_col].astype(str).str.strip()
            filtered_df[symbol_col] = filtered_df[symbol_col].astype(str).str.strip()
            
            # Create token-symbol map
            for _, row in filtered_df.iterrows():
                token = str(row[token_col])
                symbol = row[symbol_col]
                
                # Determine exchange (default to NSE)
                exchange = row[exchange_col] if exchange_col and pd.notna(row[exchange_col]) else 'NSE'
                
                # Additional metadata
                metadata = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'series': row[series_col] if series_col else None,
                    'isin': row[isin_col] if isin_col else None
                }
                
                # Store in token-symbol map
                self.token_symbol_map[token] = metadata
                
                # Create symbol to token mapping
                key = f"{exchange}|{symbol}"
                self.symbol_token_map[key] = token
            
            self.logger.info(f"Created maps: {len(self.token_symbol_map)} tokens, {len(self.symbol_token_map)} symbols")
        
        except Exception as e:
            self.logger.error(f"Error creating maps: {e}")

    def search_securities(self, 
                        query: str, 
                        exchange: Optional[str] = None,
                        limit: int = 10,
                        security_type: Optional[str] = None) -> List[Dict]:
        """
        Advanced search for securities with multiple filtering options
        
        Args:
            query: Search query (symbol or name)
            exchange: Exchange code to filter by
            limit: Maximum number of results
            security_type: Type of security (e.g., 'equity', 'futures', 'options')
        
        Returns:
            List of matching securities
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return []
        
        try:
            # Determine column names dynamically
            columns = self.security_master_df.columns.tolist()
            
            # Mapping of possible column names
            column_mappings = {
                'symbol': ['ScripName', 'Symbol', 'symbol', 'ShortName'],
                'exchange': ['ExchangeCode', 'Exchange', 'exchange'],
                'series': ['Series', 'series'],
                'name': ['CompanyName', 'Company Name', 'Name']
            }
            
            # Find the actual column names
            def find_column(possible_names):
                return next((col for col in possible_names if col in columns), None)
            
            symbol_col = find_column(column_mappings['symbol'])
            exchange_col = find_column(column_mappings['exchange'])
            series_col = find_column(column_mappings['series'])
            name_col = find_column(column_mappings['name'])
            
            # Validate required columns
            if not symbol_col:
                self.logger.error(f"Symbol column not found. Available columns: {columns}")
                return []
            
            # Create base mask for filtering
            query_lower = query.lower()
            mask = (
                self.security_master_df[symbol_col].str.lower().str.contains(query_lower, na=False) |
                (name_col and self.security_master_df[name_col].str.lower().str.contains(query_lower, na=False))
            )
            
            # Apply additional filters
            if exchange and exchange_col:
                mask &= self.security_master_df[exchange_col] == exchange
            
            if security_type and series_col:
                # Map security types to series
                series_map = {
                    'equity': ['EQ', 'N', 'BE'],
                    'futures': ['FUT'],
                    'options': ['OPT']
                }
                
                # Get corresponding series for the security type
                series_list = series_map.get(security_type.lower(), [])
                if series_list:
                    mask &= self.security_master_df[series_col].isin(series_list)
            
            # Apply filtering and get results
            results = self.security_master_df[mask].head(limit)
            
            # Convert to list of dictionaries
            return results.to_dict('records')
        
        except Exception as e:
            self.logger.error(f"Error searching securities: {e}")
            return []
            
    def get_security_details(self, 
                           symbol: str, 
                           exchange: str = 'NSE') -> Optional[Dict]:
        """
        Get security details for a symbol
        
        Args:
            symbol: Symbol to look up
            exchange: Exchange code
            
        Returns:
            Dictionary with security details or None if not found
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return None
            
        try:
            # Check which columns are available
            columns = self.security_master_df.columns.tolist()
            
            # Define mappings for different possible column names
            symbol_cols = ['Symbol', 'symbol', 'ticker', 'Ticker']
            exchange_cols = ['Exchange', 'exchange', 'exch', 'Exch']
            
            # Find the actual column names
            symbol_col = next((col for col in symbol_cols if col in columns), None)
            exchange_col = next((col for col in exchange_cols if col in columns), None)
            
            if not symbol_col:
                self.logger.error(f"Symbol column not found in security master. Columns: {columns}")
                return None
            
            # Look up security by symbol
            mask = (self.security_master_df[symbol_col] == symbol)
            
            # Add exchange filter if column exists
            if exchange_col:
                mask = mask & (self.security_master_df[exchange_col] == exchange)
                
            if mask.sum() == 0:
                return None
                
            # Get first matching row
            row = self.security_master_df[mask].iloc[0]
            
            # Convert to dictionary
            return row.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting security details: {e}")
            return None
            
    def get_token(self, 
                symbol: str, 
                exchange: str = 'NSE') -> Optional[str]:
        """
        Get token for a symbol
        
        Args:
            symbol: Symbol to look up
            exchange: Exchange code
            
        Returns:
            Token or None if not found
        """
        key = f"{exchange}|{symbol}"
        return self.symbol_token_map.get(key)
            
    def get_symbol(self, 
                 token: str) -> Optional[Dict]:
        """
        Get symbol for a token
        
        Args:
            token: Token to look up
            
        Returns:
            Dictionary with symbol and exchange or None if not found
        """
        return self.token_symbol_map.get(token)
            
    def search_securities(self, 
                        query: str, 
                        exchange: Optional[str] = None,
                        limit: int = 10) -> List[Dict]:
        """
        Search for securities by name or symbol
        
        Args:
            query: Search query
            exchange: Exchange code to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching securities
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return []
            
        try:
            # Create case-insensitive query
            query = query.lower()
            
            # Check which columns are available
            columns = self.security_master_df.columns.tolist()
            
            # Define mappings for different possible column names
            symbol_cols = ['Symbol', 'symbol', 'ticker', 'Ticker']
            name_cols = ['Company Name', 'company_name', 'name', 'Name', 'Issuer', 'issuer']
            exchange_cols = ['Exchange', 'exchange', 'exch', 'Exch']
            
            # Find the actual column names
            symbol_col = next((col for col in symbol_cols if col in columns), None)
            name_col = next((col for col in name_cols if col in columns), None)
            exchange_col = next((col for col in exchange_cols if col in columns), None)
            
            if not symbol_col:
                self.logger.error(f"Symbol column not found in security master. Columns: {columns}")
                return []
            
            # Create mask for symbols containing query
            mask = self.security_master_df[symbol_col].str.lower().str.contains(query, na=False)
            
            # Add mask for company name if column exists
            if name_col:
                # Handle NaN values with na=False
                name_mask = self.security_master_df[name_col].str.lower().str.contains(query, na=False)
                mask = mask | name_mask
                
            # Add exchange filter if specified
            if exchange and exchange_col:
                mask = mask & (self.security_master_df[exchange_col] == exchange)
                
            # Get matching rows
            matches = self.security_master_df[mask].head(limit)
            
            # Convert to list of dictionaries
            return matches.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error searching securities: {e}")
            return []


# Singleton instance
_security_master_instance = None

def get_security_master(force_new: bool = False, log_level: int = logging.INFO) -> SecurityMaster:
    """
    Get SecurityMaster instance (singleton)
    
    Args:
        force_new: Force creation of new instance
        log_level: Logging level
        
    Returns:
        SecurityMaster instance
    """
    global _security_master_instance
    
    if _security_master_instance is None or force_new:
        _security_master_instance = SecurityMaster(log_level=log_level)
        
    return _security_master_instance