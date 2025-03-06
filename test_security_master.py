# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 21:58:02 2025

@author: mahes
"""

# -*- coding: utf-8 -*-
"""
Test script for the security master module
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

# Import the security master 
# (Assuming security_master.py is in the api directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.security_master import SecurityMaster

def setup_logging(level=logging.INFO):
    """Set up logging for the test script"""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/test_security_master.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("test_security_master")

def run_download_test(security_master, logger):
    """Test downloading the security master files"""
    print("\n" + "="*80)
    print("TESTING SECURITY MASTER DOWNLOAD")
    print("="*80)
    
    try:
        print("Downloading security master files...")
        success = security_master.download_security_master()
        
        if success:
            print("✅ Security master downloaded successfully")
            
            # Check the downloaded files
            expected_files = [
                'NSEScripMaster.csv',
                'BSEScripMaster.csv',
                'FONSEScripMaster.csv',
                'FOBSEScripMaster.csv'
            ]
            
            for file in expected_files:
                path = Path('cache') / file
                if path.exists():
                    file_size = path.stat().st_size
                    print(f"  - {file}: {file_size} bytes")
                else:
                    print(f"  - {file}: ❌ Not found")
        else:
            print("❌ Security master download failed")
            
        return success
    except Exception as e:
        logger.error(f"Error in download test: {e}")
        print(f"❌ Error in download test: {e}")
        return False

def run_loading_test(security_master, logger):
    """Test loading the security master files"""
    print("\n" + "="*80)
    print("TESTING SECURITY MASTER LOADING")
    print("="*80)
    
    try:
        print("Loading security master files...")
        success = security_master.load_security_master()
        
        if success:
            print("✅ Security master loaded successfully")
            
            # Check loaded data
            df_count = len(security_master.security_master_df)
            token_count = len(security_master.token_symbol_map)
            symbol_count = len(security_master.symbol_token_map)
            
            print(f"  - Total records: {df_count}")
            print(f"  - Token mappings: {token_count}")
            print(f"  - Symbol mappings: {symbol_count}")
            
            # Show sample data
            if df_count > 0:
                print("\nSample data (first 3 rows):")
                columns = security_master.security_master_df.columns.tolist()
                sample_data = security_master.security_master_df.head(3)
                
                # Print in a nicer format
                for i, row in sample_data.iterrows():
                    print(f"\nRow {i+1}:")
                    for j, col in enumerate(columns[:10]):  # Limit to first 10 columns
                        print(f"  {col}: {row[col]}")
                    
                    if len(columns) > 10:
                        print(f"  ... ({len(columns) - 10} more columns)")
            
            # Check token mapping
            if token_count > 0:
                print("\nSample token mappings (first 3):")
                for i, (token, data) in enumerate(list(security_master.token_symbol_map.items())[:3]):
                    print(f"  Token {token}: {data}")
                    
            # Check symbol mapping
            if symbol_count > 0:
                print("\nSample symbol mappings (first 3):")
                for i, (symbol, token) in enumerate(list(security_master.symbol_token_map.items())[:3]):
                    print(f"  Symbol {symbol}: {token}")
        else:
            print("❌ Security master loading failed")
            
        return success
    except Exception as e:
        logger.error(f"Error in loading test: {e}")
        print(f"❌ Error in loading test: {e}")
        return False

def run_equity_search_test(security_master, logger):
    """Test searching for equity securities"""
    print("\n" + "="*80)
    print("TESTING EQUITY SECURITY SEARCH")
    print("="*80)
    
    try:
        test_queries = [
            {"query": "RELIANCE", "exchange": "NSE", "security_type": "equity"},
            {"query": "TATASTEEL", "exchange": "NSE", "security_type": "equity"},
            {"query": "INFY", "exchange": "NSE", "security_type": "equity"},
            {"query": "HDFC", "exchange": "NSE", "security_type": "equity"},
            {"query": "TCS", "exchange": "NSE", "security_type": "equity"}
        ]
        
        for test in test_queries:
            print(f"\nSearching for: {test['query']} ({test['exchange']}, {test['security_type']})")
            results = security_master.search_securities(**test)
            
            if results:
                print(f"✅ Found {len(results)} results")
                
                # Show top 3 results
                for i, result in enumerate(results[:3]):
                    print(f"\nResult {i+1}:")
                    
                    # Display selected fields based on what's available
                    fields = [
                        'Token', 'Symbol', 'ShortName', 'CompanyName', 'ScripName',
                        'Series', 'ExchangeCode', 'TickSize', 'LotSize'
                    ]
                    
                    for field in fields:
                        if field in result:
                            print(f"  {field}: {result[field]}")
            else:
                print(f"❌ No results found")
        
        return True
    except Exception as e:
        logger.error(f"Error in equity search test: {e}")
        print(f"❌ Error in equity search test: {e}")
        return False

def run_index_search_test(security_master, logger):
    """Test searching for index symbols"""
    print("\n" + "="*80)
    print("TESTING INDEX SEARCH")
    print("="*80)
    
    try:
        test_queries = [
            {"query": "NIFTY", "security_type": "index"},
            {"query": "BANKNIFTY", "security_type": "index"},
            {"query": "SENSEX", "security_type": "index"},
            {"query": "MIDCAP", "security_type": "index"}
        ]
        
        for test in test_queries:
            print(f"\nSearching for: {test['query']} ({test['security_type']})")
            results = security_master.search_securities(**test)
            
            if results:
                print(f"✅ Found {len(results)} results")
                
                # Show top 3 results
                for i, result in enumerate(results[:3]):
                    print(f"\nResult {i+1}:")
                    
                    # Display selected fields based on what's available
                    fields = [
                        'Token', 'Symbol', 'ShortName', 'InstrumentName', 'ScripName',
                        'Series', 'ExchangeCode', 'TickSize', 'LotSize'
                    ]
                    
                    for field in fields:
                        if field in result:
                            print(f"  {field}: {result[field]}")
            else:
                print(f"❌ No results found")
        
        return True
    except Exception as e:
        logger.error(f"Error in index search test: {e}")
        print(f"❌ Error in index search test: {e}")
        return False

def run_derivatives_search_test(security_master, logger):
    """Test searching for futures and options"""
    print("\n" + "="*80)
    print("TESTING DERIVATIVES SEARCH")
    print("="*80)
    
    try:
        test_queries = [
            {"query": "NIFTY", "security_type": "futures"},
            {"query": "RELIANCE", "security_type": "futures"},
            {"query": "NIFTY", "security_type": "options"},
            {"query": "BANKNIFTY", "security_type": "options"}
        ]
        
        for test in test_queries:
            print(f"\nSearching for: {test['query']} ({test['security_type']})")
            results = security_master.search_securities(**test)
            
            if results:
                print(f"✅ Found {len(results)} results")
                
                # Show top 3 results
                for i, result in enumerate(results[:3]):
                    print(f"\nResult {i+1}:")
                    
                    # Display selected fields based on what's available
                    fields = [
                        'Token', 'InstrumentName', 'ShortName', 'OptionType',
                        'StrikePrice', 'ExpiryDate', 'ExchangeCode', 'LotSize'
                    ]
                    
                    for field in fields:
                        if field in result:
                            print(f"  {field}: {result[field]}")
            else:
                print(f"❌ No results found")
        
        return True
    except Exception as e:
        logger.error(f"Error in derivatives search test: {e}")
        print(f"❌ Error in derivatives search test: {e}")
        return False

def run_token_retrieval_test(security_master, logger):
    """Test token retrieval for symbols"""
    print("\n" + "="*80)
    print("TESTING TOKEN RETRIEVAL")
    print("="*80)
    
    try:
        test_symbols = [
            {"symbol": "RELIANCE", "exchange": "NSE"},
            {"symbol": "NIFTY", "exchange": "NSE"},
            {"symbol": "TATASTEEL", "exchange": "NSE"},
            {"symbol": "HDFC", "exchange": "NSE"},
            {"symbol": "SBIN", "exchange": "NSE"}
        ]
        
        success_count = 0
        for test in test_symbols:
            symbol = test["symbol"]
            exchange = test["exchange"]
            
            print(f"\nGetting token for: {symbol} ({exchange})")
            token = security_master.get_token(symbol, exchange)
            
            if token:
                print(f"✅ Found token: {token}")
                success_count += 1
                
                # Try to get the symbol back from the token
                symbol_data = security_master.get_symbol(token)
                if symbol_data:
                    print(f"  Symbol data: {symbol_data}")
                else:
                    print(f"  ❌ Could not retrieve symbol data for token")
            else:
                print(f"❌ Token not found")
        
        print(f"\nSuccessfully retrieved {success_count} out of {len(test_symbols)} tokens")
        return success_count > 0
    except Exception as e:
        logger.error(f"Error in token retrieval test: {e}")
        print(f"❌ Error in token retrieval test: {e}")
        return False

def run_expiry_dates_test(security_master, logger):
    """Test retrieving expiry dates for derivatives"""
    print("\n" + "="*80)
    print("TESTING EXPIRY DATES RETRIEVAL")
    print("="*80)
    
    try:
        test_underlyings = [
            {"underlying": "NIFTY", "exchange": "NSE"},
            {"underlying": "BANKNIFTY", "exchange": "NSE"},
            {"underlying": "RELIANCE", "exchange": "NSE"}
        ]
        
        for test in test_underlyings:
            underlying = test["underlying"]
            exchange = test["exchange"]
            
            print(f"\nGetting expiry dates for: {underlying} ({exchange})")
            expiry_dates = security_master.get_expiry_dates(underlying, exchange)
            
            if expiry_dates:
                print(f"✅ Found {len(expiry_dates)} expiry dates")
                
                # Show first 5 expiry dates
                for i, date in enumerate(expiry_dates[:5]):
                    print(f"  {i+1}. {date}")
                    
                if len(expiry_dates) > 5:
                    print(f"  ... ({len(expiry_dates) - 5} more dates)")
                    
                # Try to get option chain for first expiry
                if len(expiry_dates) > 0:
                    first_expiry = expiry_dates[0]
                    print(f"\nGetting option chain for {underlying} with expiry {first_expiry}")
                    
                    options = security_master.get_option_chain(underlying, first_expiry, exchange)
                    if options:
                        print(f"✅ Found {len(options)} option contracts")
                        
                        # Show first 3 call and put options
                        calls = [opt for opt in options if opt.get('OptionType') == 'CE'][:3]
                        puts = [opt for opt in options if opt.get('OptionType') == 'PE'][:3]
                        
                        if calls:
                            print("\nCall Options:")
                            for i, option in enumerate(calls):
                                print(f"  {i+1}. Strike: {option.get('StrikePrice')}, Token: {option.get('Token')}")
                        
                        if puts:
                            print("\nPut Options:")
                            for i, option in enumerate(puts):
                                print(f"  {i+1}. Strike: {option.get('StrikePrice')}, Token: {option.get('Token')}")
                    else:
                        print(f"❌ No option contracts found")
            else:
                print(f"❌ No expiry dates found")
        
        return True
    except Exception as e:
        logger.error(f"Error in expiry dates test: {e}")
        print(f"❌ Error in expiry dates test: {e}")
        return False

def run_diagnostic_test(security_master, logger):
    """Run diagnostic tests to better understand the data structure"""
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)
    
    try:
        if security_master.security_master_df is None:
            print("❌ Security master not loaded")
            return False
        
        # Log details about each source file
        cache_dir = Path('cache')
        source_files = [
            'NSEScripMaster.csv',
            'BSEScripMaster.csv',
            'FONSEScripMaster.csv',
            'FOBSEScripMaster.csv'
        ]
        
        for file in source_files:
            file_path = cache_dir / file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, nrows=5)
                    print(f"\nExamining {file}:")
                    print(f"  - Columns: {', '.join(df.columns.tolist()[:10])}...")
                    
                    # Identify key columns for this file
                    if 'Token' in df.columns:
                        print(f"  - Has Token column: Yes")
                        
                        # Print first few token values
                        tokens = df['Token'].tolist()
                        print(f"  - Sample tokens: {tokens[:5]}")
                    else:
                        print(f"  - Has Token column: No")
                    
                    # Look for symbol columns
                    symbol_cols = [col for col in df.columns if col in 
                                ['Symbol', 'ShortName', 'ScripName', 'InstrumentName']]
                    if symbol_cols:
                        print(f"  - Symbol columns: {', '.join(symbol_cols)}")
                        
                        # Print first few symbol values from first symbol column
                        first_col = symbol_cols[0]
                        symbols = df[first_col].tolist()
                        print(f"  - Sample {first_col}: {symbols[:5]}")
                    else:
                        print(f"  - Symbol columns: None found")
                    
                    # Check for equity vs. derivatives specific columns
                    equity_cols = [col for col in df.columns if col in 
                                ['Series', 'CompanyName', 'ISINCode']]
                    deriv_cols = [col for col in df.columns if col in 
                                ['OptionType', 'StrikePrice', 'ExpiryDate']]
                    
                    if equity_cols:
                        print(f"  - Equity columns: {', '.join(equity_cols)}")
                    if deriv_cols:
                        print(f"  - Derivative columns: {', '.join(deriv_cols)}")
                        
                except Exception as e:
                    print(f"  ❌ Error examining {file}: {e}")
        
        # Check for important missing columns
        combined_columns = security_master.security_master_df.columns.tolist()
        important_columns = [
            'Token', 'Symbol', 'ShortName', 'InstrumentName', 'ScripName',
            'CompanyName', 'Series', 'ExchangeCode', 'OptionType', 'StrikePrice',
            'ExpiryDate'
        ]
        
        print("\nImportant column availability:")
        for col in important_columns:
            present = col in combined_columns
            print(f"  - {col}: {'✅ Present' if present else '❌ Missing'}")
        
        # Check token and symbol map
        print(f"\nMap statistics:")
        print(f"  - Records in DataFrame: {len(security_master.security_master_df)}")
        print(f"  - Tokens in map: {len(security_master.token_symbol_map)}")
        print(f"  - Symbols in map: {len(security_master.symbol_token_map)}")
        
        # Check for duplicates in token column
        if 'Token' in combined_columns:
            duplicate_count = security_master.security_master_df.duplicated(subset=['Token']).sum()
            print(f"  - Duplicate tokens: {duplicate_count}")
        
        return True
    except Exception as e:
        logger.error(f"Error in diagnostic test: {e}")
        print(f"❌ Error in diagnostic test: {e}")
        return False

def main():
    """Main function for the security master test script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Security Master Test Script')
    parser.add_argument('--cache-dir', type=str, default='cache',
                      help='Directory for cached security master files')
    parser.add_argument('--force-download', action='store_true',
                      help='Force download of security master files')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--test', type=str, choices=[
        'download', 'load', 'equity', 'index', 'derivatives', 
        'token', 'expiry', 'diagnostic', 'all'
    ], default='all', help='Test to run')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    
    # Print test header
    print("\n" + "="*80)
    print(f"SECURITY MASTER TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize security master
    security_master = SecurityMaster(
        cache_dir=args.cache_dir,
        log_level=log_level
    )
    
    # Run tests based on command line arguments
    tests_to_run = []
    if args.test == 'all':
        tests_to_run = ['download', 'load', 'equity', 'index', 'derivatives', 'token', 'expiry', 'diagnostic']
    else:
        tests_to_run = [args.test]
    
    results = {}
    
    # Always run download and load tests first if needed
    if 'download' in tests_to_run or args.force_download:
        results['download'] = run_download_test(security_master, logger)
    elif not security_master.is_cache_valid():
        print("\nCache not valid, running download test...")
        results['download'] = run_download_test(security_master, logger)
    
    if 'load' in tests_to_run or not any([r in tests_to_run for r in ['download', 'load']]):
        results['load'] = run_loading_test(security_master, logger)
    
    # Run remaining tests
    if 'equity' in tests_to_run:
        results['equity'] = run_equity_search_test(security_master, logger)
    
    if 'index' in tests_to_run:
        results['index'] = run_index_search_test(security_master, logger)
    
    if 'derivatives' in tests_to_run:
        results['derivatives'] = run_derivatives_search_test(security_master, logger)
    
    if 'token' in tests_to_run:
        results['token'] = run_token_retrieval_test(security_master, logger)
    
    if 'expiry' in tests_to_run:
        results['expiry'] = run_expiry_dates_test(security_master, logger)
    
    if 'diagnostic' in tests_to_run:
        results['diagnostic'] = run_diagnostic_test(security_master, logger)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test, result in results.items():
        print(f"{test.capitalize()} Test: {'✅ PASSED' if result else '❌ FAILED'}")
    
    # Overall result
    overall = all(results.values())
    print("\nOverall Result: " + ("✅ PASSED" if overall else "❌ FAILED"))
    
    return 0 if overall else 1

if __name__ == "__main__":
    sys.exit(main())