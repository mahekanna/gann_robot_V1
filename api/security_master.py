# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 07:46:20 2025

@author: mahes
"""

# -*- coding: utf-8 -*-
"""
Security Master module for ICICI Direct Breeze API
Handles downloading, parsing, and caching multiple security master files
"""

import os
import logging
import pandas as pd
import requests
import zipfile
import io
import time
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
from datetime import datetime, timedelta

class SecurityMaster:
    """
    Security Master manager for ICICI Direct Breeze API
    Downloads and manages security master files
    """
    
    # Security master URL (ICICI Direct Security Master)
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
        Download security master file from ICICI Direct with retry mechanism
        
        Args:
            max_retries: Maximum number of download attempts
        
        Returns:
            True if download successful, False otherwise
        """
        import ssl
        import urllib3

        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Configure requests with user agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Download ZIP file
                response = requests.get(
                    self.SECURITY_MASTER_URL, 
                    headers=headers, 
                    verify=False,  # Disable SSL verification
                    timeout=60  # Longer timeout
                )
                
                # Check if response is successful
                if response.status_code != 200:
                    self.logger.error(f"HTTP request failed with status code {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    continue
                
                # Try to open the ZIP file
                try:
                    z = zipfile.ZipFile(io.BytesIO(response.content))
                    
                    # List all files in the ZIP
                    all_files = z.namelist()
                    self.logger.info(f"ZIP contents: {all_files}")
                    
                    # Find TXT files
                    txt_files = [f for f in all_files if f.lower().endswith('.txt')]
                    
                    if not txt_files:
                        self.logger.error("No TXT files found in security master ZIP")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                        continue
                    
                    # Create cache directory if it doesn't exist
                    os.makedirs(self.cache_dir, exist_ok=True)
                    
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
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                        continue
                
                except zipfile.BadZipFile as ze:
                    self.logger.error(f"Bad ZIP file: {ze}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    continue
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error downloading security master: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error downloading security master: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return False
        
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
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
                # Try to detect delimiter
                if '\t' in first_line:
                    delimiter = '\t'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    # Default to comma
                    delimiter = ','
                
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
        # Check if cache file exists
        if not self.cache_file.exists():
            self.logger.info(f"Cache file {self.cache_file} does not exist")
            return False
        
        # Check file age
        try:
            file_time = datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            max_age = timedelta(days=self.max_age_days)
            
            is_valid = datetime.now() - file_time < max_age
            
            if is_valid:
                self.logger.info(f"Cache file is valid. Created {file_time}")
            else:
                self.logger.info(f"Cache file is older than {self.max_age_days} days")
            
            return is_valid
        
        except Exception as e:
            self.logger.error(f"Error checking cache validity: {e}")
            return False
    
    def load_security_master(self, force_download: bool = False) -> bool:
        """
        Load and combine multiple security master files
        
        Args:
            force_download: Force download even if cache is valid
            
        Returns:
            True if loading successful, False otherwise
        """
        # Check if we need to download
        if force_download or not self.is_cache_valid():
            self.logger.info("Downloading security master due to force download or invalid cache")
            if not self.download_security_master():
                if not self.cache_file.exists():
                    self.logger.error("Failed to download security master and no cache available")
                    return False
                else:
                    self.logger.warning("Using outdated security master cache")
        
        try:
            # List of CSV files to combine
            csv_files = [
                self.cache_dir / 'NSEScripMaster.csv',
                self.cache_dir / 'BSEScripMaster.csv',
                self.cache_dir / 'FONSEScripMaster.csv',
                self.cache_dir / 'FOBSEScripMaster.csv'
            ]
            
            # Combine dataframes
            dataframes = []
            for file in csv_files:
                if file.exists():
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        dataframes.append(df)
                        self.logger.info(f"Loaded {file.name}: {len(df)} records")
                    except Exception as e:
                        self.logger.warning(f"Could not load {file.name}: {e}")
            
            # Combine all dataframes
            if dataframes:
                self.security_master_df = pd.concat(dataframes, ignore_index=True)
            else:
                self.logger.error("No security master files could be loaded")
                return False
            
            # Remove duplicates based on Token column if it exists
            if 'Token' in self.security_master_df.columns:
                self.security_master_df.drop_duplicates(subset='Token', keep='first', inplace=True)
            
            # Create token-symbol and symbol-token maps
            self._create_maps()
            
            self.logger.info(f"Combined security master loaded: {len(self.security_master_df)} total records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading security master: {e}")
            return False
    
    def _create_maps(self) -> None:
        """
        Create token-symbol and symbol-token maps accounting for different file formats
        """
        if self.security_master_df is None:
            return
            
        # Reset maps
        self.token_symbol_map = {}
        self.symbol_token_map = {}
        
        try:
            # Show available columns for debugging
            columns = self.security_master_df.columns.tolist()
            if len(columns) > 10:
                self.logger.info(f"Available columns for mapping: {', '.join(columns[:10])}...")
            else:
                self.logger.info(f"Available columns for mapping: {', '.join(columns)}")
            
            # Find the token column
            token_cols = ['Token', 'ScripCode', 'SecurityId', 'Security Id']
            token_col = next((col for col in token_cols if col in columns), None)
            
            if not token_col:
                self.logger.error(f"Token column not found in available columns")
                return
                
            # Process each row
            mapped_count = 0
            for _, row in self.security_master_df.iterrows():
                try:
                    # Skip rows with missing token
                    if pd.isna(row[token_col]):
                        continue
                    
                    token = str(row[token_col]).strip()
                    if not token:
                        continue
                    
                    # Determine which type of security this is based on available columns
                    is_equity = False
                    is_derivative = False
                    
                    # Check for equity
                    if 'Series' in columns and not pd.isna(row['Series']):
                        series = str(row['Series']).strip()
                        is_equity = series in ['EQ', 'BE', 'N', 'E']
                    
                    # Check for derivatives
                    if 'OptionType' in columns:
                        is_derivative = True
                    elif 'InstrumentType' in columns and not pd.isna(row['InstrumentType']):
                        instrument_type = str(row['InstrumentType']).strip()
                        is_derivative = instrument_type in ['FUT', 'OPT', 'FUTURES', 'OPTIONS']
                    
                    # Get symbol based on available columns
                    symbol = None
                    symbol_cols = ['Symbol', 'ShortName', 'ScripName', 'InstrumentName']
                    for col in symbol_cols:
                        if col in columns and not pd.isna(row[col]):
                            symbol = str(row[col]).strip()
                            if symbol:
                                break
                    
                    if not symbol:
                        continue
                    
                    # Get exchange
                    exchange = 'NSE'  # Default exchange
                    exchange_cols = ['ExchangeCode', 'Exchange']
                    for col in exchange_cols:
                        if col in columns and not pd.isna(row[col]):
                            exchange = str(row[col]).strip()
                            break
                    
                    # Create metadata
                    metadata = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'type': 'DERIVATIVE' if is_derivative else 'EQUITY'
                    }
                    
                    # Add additional metadata for derivatives
                    if is_derivative:
                        if 'OptionType' in columns and not pd.isna(row['OptionType']):
                            metadata['option_type'] = str(row['OptionType']).strip()
                        
                        if 'StrikePrice' in columns and not pd.isna(row['StrikePrice']):
                            metadata['strike_price'] = float(row['StrikePrice'])
                        
                        if 'ExpiryDate' in columns and not pd.isna(row['ExpiryDate']):
                            metadata['expiry'] = str(row['ExpiryDate']).strip()
                    
                    # Store in token map
                    self.token_symbol_map[token] = metadata
                    
                    # Create symbol to token mapping
                    key = f"{exchange}|{symbol}"
                    self.symbol_token_map[key] = token
                    
                    # Track mapping count
                    mapped_count += 1
                    
                    # Log first few for debugging
                    if mapped_count <= 5:
                        self.logger.debug(f"Mapped: {token} -> {metadata}")
                        
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            self.logger.info(f"Created maps: {len(self.token_symbol_map)} tokens, {len(self.symbol_token_map)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error creating maps: {e}")
    
    def search_securities(self, 
                        query: str, 
                        exchange: Optional[str] = None,
                        security_type: Optional[str] = None,
                        limit: int = 10) -> List[Dict]:
        """
        Search for securities matching the query
        
        Args:
            query: Symbol, name, or partial match
            exchange: Exchange code (NSE, BSE)
            security_type: Type of security (equity, futures, options)
            limit: Maximum number of results
            
        Returns:
            List of matching securities
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return []
            
        try:
            # Determine which columns to search in based on available columns
            columns = self.security_master_df.columns.tolist()
            
            # Define search columns based on security type and available columns
            search_columns = []
            
            # For equity/index
            if not security_type or security_type.lower() in ['equity', 'index']:
                if 'Symbol' in columns: search_columns.append('Symbol')
                if 'ShortName' in columns: search_columns.append('ShortName')
                if 'CompanyName' in columns: search_columns.append('CompanyName')
                if 'ScripName' in columns: search_columns.append('ScripName')
            
            # For derivatives
            if not security_type or security_type.lower() in ['futures', 'options']:
                if 'InstrumentName' in columns: search_columns.append('InstrumentName')
                if 'ShortName' in columns: search_columns.append('ShortName')
            
            if not search_columns:
                self.logger.error(f"No searchable columns found in: {columns}")
                return []
                
            # Log which columns we're searching
            self.logger.info(f"Searching in columns: {search_columns}")
            
            # Create the search mask
            query_lower = query.lower()
            mask = pd.Series(False, index=self.security_master_df.index)
            
            # Search in each column
            for col in search_columns:
                # Skip numeric columns
                if pd.api.types.is_numeric_dtype(self.security_master_df[col]):
                    continue
                    
                # Search in string columns
                try:
                    col_mask = self.security_master_df[col].astype(str).str.lower().str.contains(query_lower, na=False)
                    mask = mask | col_mask
                except Exception as e:
                    self.logger.warning(f"Error searching in column {col}: {e}")
            
            # Apply exchange filter
            if exchange:
                exchange_cols = ['ExchangeCode', 'Exchange']
                for col in exchange_cols:
                    if col in columns:
                        exchange_mask = self.security_master_df[col] == exchange
                        mask = mask & exchange_mask
                        break
            
            # Apply security type filter
            if security_type:
                security_type_lower = security_type.lower()
                
                # For equity
                if security_type_lower == 'equity':
                    if 'Series' in columns:
                        mask = mask & self.security_master_df['Series'].isin(['EQ', 'BE', 'N', 'E'])
                    elif 'InstrumentType' in columns:
                        mask = mask & self.security_master_df['InstrumentType'].isin(['EQUITY', 'CASH'])
                
                # For futures
                elif security_type_lower == 'futures':
                    if 'OptionType' in columns:
                        # In F&O data, futures don't have an option type
                        mask = mask & pd.isna(self.security_master_df['OptionType'])
                    if 'Series' in columns:
                        mask = mask & self.security_master_df['Series'].isin(['FUT'])
                    elif 'InstrumentType' in columns:
                        mask = mask & self.security_master_df['InstrumentType'].isin(['FUT', 'FUTURES'])
                
                # For options
                elif security_type_lower == 'options':
                    if 'OptionType' in columns:
                        mask = mask & self.security_master_df['OptionType'].isin(['CE', 'PE'])
                    if 'Series' in columns:
                        mask = mask & self.security_master_df['Series'].isin(['OPT'])
                    elif 'InstrumentType' in columns:
                        mask = mask & self.security_master_df['InstrumentType'].isin(['OPT', 'OPTIONS'])
                        
                # For index
                elif security_type_lower == 'index':
                    if 'Series' in columns:
                        mask = mask & self.security_master_df['Series'].isin(['IDX', 'INDEX'])
                    # Often index symbols have "NIFTY" or "SENSEX" in them
                    index_mask = False
                    for col in search_columns:
                        if pd.api.types.is_string_dtype(self.security_master_df[col]):
                            try:
                                col_mask = (
                                    self.security_master_df[col].astype(str).str.contains('NIFTY', case=False, na=False) |
                                    self.security_master_df[col].astype(str).str.contains('SENSEX', case=False, na=False) |
                                    self.security_master_df[col].astype(str).str.contains('INDEX', case=False, na=False)
                                )
                                index_mask = index_mask | col_mask
                            except:
                                pass
                    mask = mask & index_mask
            
            # Get results
            results = self.security_master_df[mask].head(limit)
            
            if len(results) > 0:
                self.logger.info(f"Found {len(results)} matches for query: {query}")
            else:
                self.logger.warning(f"No matches found for query: {query}")
                
                # If we're searching with a specific security type and found nothing,
                # try again without the security type filter
                if security_type and not exchange:
                    self.logger.info("Trying search without security type filter")
                    return self.search_securities(query, exchange, None, limit)
            
            return results.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error searching securities: {e}")
            return []
        
    def get_token(self, 
                symbol: str, 
                exchange: str = 'NSE') -> Optional[str]:
        """
        Get token for a specific symbol and exchange
        
        Args:
            symbol: Symbol to look up
            exchange: Exchange code (default: NSE)
        
        Returns:
            Token or None if not found
        """
        key = f"{exchange}|{symbol}"
        token = self.symbol_token_map.get(key)
        
        if token:
            self.logger.info(f"Found token for {symbol} ({exchange}): {token}")
        else:
            self.logger.warning(f"Token not found for {symbol} ({exchange})")
            
            # Try with case-insensitive match
            for map_key, map_token in self.symbol_token_map.items():
                try:
                    parts = map_key.split('|')
                    if len(parts) >= 2 and parts[0] == exchange and parts[1].lower() == symbol.lower():
                        self.logger.info(f"Found case-insensitive token match: {map_token}")
                        return map_token
                except:
                    continue
        
        return token
    
    def get_symbol(self, 
                 token: str) -> Optional[Dict]:
        """
        Get symbol details for a specific token
        
        Args:
            token: Token to look up
        
        Returns:
            Dictionary with symbol details or None if not found
        """
        return self.token_symbol_map.get(token)
    
    def get_option_chain(self,
                       underlying: str,
                       expiry_date: Optional[str] = None,
                       exchange: str = 'NSE') -> List[Dict]:
        """
        Get option chain data for a specific underlying
        
        Args:
            underlying: Underlying symbol (e.g., 'NIFTY', 'RELIANCE')
            expiry_date: Expiry date (optional, format depends on security master)
            exchange: Exchange code (default: NSE)
            
        Returns:
            List of option contracts
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return []
            
        try:
            # Check if we have derivatives data
            if 'OptionType' not in self.security_master_df.columns:
                self.logger.error("Option chain data not available, missing 'OptionType' column")
                return []
                
            # Find column with underlying name
            underlying_cols = ['AssetName', 'InstrumentName', 'ShortName']
            underlying_col = next((col for col in underlying_cols if col in self.security_master_df.columns), None)
            
            if not underlying_col:
                self.logger.error(f"Could not find column with underlying name")
                return []
                
            # Create search mask
            underlying_lower = underlying.lower()
            mask = self.security_master_df[underlying_col].astype(str).str.lower().str.contains(underlying_lower, na=False)
            
            # Filter by option type
            mask = mask & self.security_master_df['OptionType'].isin(['CE', 'PE'])
            
            # Filter by exchange
            if 'ExchangeCode' in self.security_master_df.columns:
                mask = mask & (self.security_master_df['ExchangeCode'] == exchange)
            
            # Filter by expiry date if provided
            if expiry_date and 'ExpiryDate' in self.security_master_df.columns:
                mask = mask & (self.security_master_df['ExpiryDate'] == expiry_date)
                
            # Get results
            results = self.security_master_df[mask]
            
            if len(results) > 0:
                self.logger.info(f"Found {len(results)} option contracts for {underlying}")
                return results.to_dict('records')
            else:
                self.logger.warning(f"No option contracts found for {underlying}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return []
    
    def get_expiry_dates(self,
                       underlying: str,
                       exchange: str = 'NSE') -> List[str]:
        """
        Get available expiry dates for a specific underlying
        
        Args:
            underlying: Underlying symbol (e.g., 'NIFTY', 'RELIANCE')
            exchange: Exchange code (default: NSE)
            
        Returns:
            List of expiry dates
        """
        if self.security_master_df is None:
            self.logger.error("Security master not loaded")
            return []
            
        try:
            # Check if we have derivatives data
            if 'ExpiryDate' not in self.security_master_df.columns:
                self.logger.error("Expiry date data not available, missing 'ExpiryDate' column")
                return []
                
            # Find column with underlying name
            underlying_cols = ['AssetName', 'InstrumentName', 'ShortName']
            underlying_col = next((col for col in underlying_cols if col in self.security_master_df.columns), None)
            
            if not underlying_col:
                self.logger.error(f"Could not find column with underlying name")
                return []
                
            # Create search mask
            underlying_lower = underlying.lower()
            mask = self.security_master_df[underlying_col].astype(str).str.lower().str.contains(underlying_lower, na=False)
            
            # Filter by exchange
            if 'ExchangeCode' in self.security_master_df.columns:
                mask = mask & (self.security_master_df['ExchangeCode'] == exchange)
                
            # Get unique expiry dates
            expiry_dates = sorted(self.security_master_df.loc[mask, 'ExpiryDate'].dropna().unique().tolist())
            
            if expiry_dates:
                self.logger.info(f"Found {len(expiry_dates)} expiry dates for {underlying}")
                return expiry_dates
            else:
                self.logger.warning(f"No expiry dates found for {underlying}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting expiry dates: {e}")
            return []