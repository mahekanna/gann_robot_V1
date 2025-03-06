# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""


# File: api/auth.py
"""
Authentication module for ICICI Direct Breeze API
Direct adaptation from the working autologin.py
"""

import os
import csv
import datetime
import pyotp
import logging
from pathlib import Path
from breeze_connect import BreezeConnect

# Import patch to fix SECURITY_MASTER_URL
from .breeze_patch import apply_patch

class BreezeAuth:
    """
    Authentication manager for ICICI Direct Breeze API
    """
    
    def __init__(self, 
                 config_dir: str = 'config',
                 env_file: str = '.env',
                 session_file: str = 'session_key.csv',
                 log_level: int = logging.INFO):
        """
        Initialize authentication manager
        
        Args:
            config_dir: Directory for configuration files
            env_file: File containing API credentials
            session_file: File to store session information
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger("breeze_auth")
        self.logger.setLevel(log_level)
        
        # Add handlers if none exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Create log directory
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(log_dir / 'breeze_auth.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Initialize variables
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self.env_file = self.config_dir / env_file
        self.session_file = self.config_dir / session_file
        
        # Load environment variables
        self._load_credentials()
        
        # BreezeConnect instance
        self.breeze = None
    
    def _load_credentials(self):
        """Load credentials from environment variables or .env file"""
        # Try to load from environment variables first
        self.api_key = os.getenv('ICICI_API_KEY')
        self.api_secret = os.getenv('ICICI_API_SECRET')
        self.totp_secret = os.getenv('ICICI_TOTP_SECRET')
        
        # If not found in environment, try loading from .env file
        if not all([self.api_key, self.api_secret, self.totp_secret]):
            if self.env_file.exists():
                self.logger.info(f"Loading credentials from {self.env_file}")
                try:
                    with open(self.env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                    if key == 'ICICI_API_KEY':
                                        self.api_key = value
                                    elif key == 'ICICI_API_SECRET':
                                        self.api_secret = value
                                    elif key == 'ICICI_TOTP_SECRET':
                                        self.totp_secret = value
                except Exception as e:
                    self.logger.error(f"Error reading .env file: {e}")
                    
        if all([self.api_key, self.api_secret, self.totp_secret]):
            self.logger.info("Credentials loaded successfully")
        else:
            self.logger.error("Failed to load all required credentials")
    
    def _generate_totp(self):
        """Generate TOTP code for authentication"""
        totp = pyotp.TOTP(self.totp_secret)
        totp_code = totp.now()
        self.logger.info(f"Generated TOTP: {totp_code}")
        return totp_code
    
    def _save_session_key(self, session_key):
        """Save session key to CSV file"""
        try:
            with open(self.session_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([session_key, datetime.datetime.now().isoformat()])
            self.logger.info("Session key saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error saving session key: {e}")
            return False
    
    def _load_session_key(self):
        """Load session key from CSV file"""
        if self.session_file.exists():
            try:
                with open(self.session_file, mode='r') as file:
                    reader = csv.reader(file)
                    session_data = next(reader, [None, None])
                    session_key, timestamp = session_data
                    if session_key and timestamp:
                        generated_at = datetime.datetime.fromisoformat(timestamp)
                        time_difference = datetime.datetime.now() - generated_at
                        # Check if session is less than 24 hours old
                        if time_difference < datetime.timedelta(hours=24):
                            self.logger.info(f"Session key loaded (generated {time_difference.total_seconds()/3600:.1f} hours ago)")
                            return session_key
                        else:
                            self.logger.info("Session key expired (older than 24 hours)")
                    else:
                        self.logger.warning("Invalid session key data")
            except Exception as e:
                self.logger.error(f"Error reading session key: {e}")
        else:
            self.logger.info("No session key file found")
        
        return None
    
    def get_breeze_client(self):
        """
        Initialize BreezeConnect, authenticate, and return client
        
        Returns:
            BreezeConnect instance with valid session or None on failure
        """
        # Apply patch to ensure SECURITY_MASTER_URL is set
        apply_patch()
        
        # Check credentials
        if not all([self.api_key, self.api_secret, self.totp_secret]):
            self.logger.error("Missing API credentials")
            return None
        
        try:
            # Initialize BreezeConnect
            self.breeze = BreezeConnect(api_key=self.api_key)
            
            # Check if session key is saved and reuse it if valid
            session_key = self._load_session_key()
            if session_key:
                try:
                    print("Using saved session key.")
                    self.breeze.generate_session(api_secret=self.api_secret, session_token=session_key)
                    
                    # Verify session is valid with a simple API call
                    try:
                        customer_details = self.breeze.get_customer_details()
                        # If this succeeds without exception, session is valid
                        self.logger.info("Session validated successfully")
                        return self.breeze
                    except Exception as e:
                        self.logger.warning(f"Saved session appears invalid: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to use saved session key: {e}")
            
            # If we get here, we need a new session
            self.logger.info("Generating new session key.")
            print("Generating new session key.")
            
            # Generate TOTP
            totp_code = self._generate_totp()
            print(f"Generated TOTP: {totp_code}")
            
            # Get session token manually
            print(f"Login URL: https://api.icicidirect.com/apiuser/login?api_key={self.api_key}")
            print("\nInstructions:")
            print("1. Open the login URL in your browser")
            print("2. Enter the TOTP code shown above")
            print("3. After successful login, you'll be redirected to a new URL")
            print("4. Look for 'token=' in the URL and copy the value after it")
            print("   Example: If URL is 'callback?token=12345', enter '12345'\n")
            
            session_token = input("Enter the session token from the redirect URL: ").strip()
            
            # Save session key to file
            self._save_session_key(session_token)
            
            # Generate session
            try:
                self.breeze.generate_session(api_secret=self.api_secret, session_token=session_token)
                self.logger.info("Successfully authenticated with new session token")
                print("Successfully authenticated!")
                return self.breeze
            except Exception as e:
                self.logger.error(f"Failed to generate new session: {e}")
                print(f"Authentication failed: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error initializing BreezeConnect: {e}")
            return None


# Singleton instance
_auth_instance = None

def get_auth(force_new: bool = False, log_level: int = logging.INFO) -> BreezeAuth:
    """
    Get BreezeAuth instance (singleton)
    
    Args:
        force_new: Force creation of new instance
        log_level: Logging level
        
    Returns:
        BreezeAuth instance
    """
    global _auth_instance
    
    if _auth_instance is None or force_new:
        _auth_instance = BreezeAuth(log_level=log_level)
        
    return _auth_instance


# Simplified for API factory module
def get_client(force_new: bool = False, log_level: int = logging.INFO):
    """
    Get authenticated BreezeConnect client
    
    Args:
        force_new: Force new authentication
        log_level: Logging level
        
    Returns:
        Authenticated BreezeConnect client
    """
    auth = get_auth(force_new, log_level)
    return auth.get_breeze_client()


# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get client directly
    client = get_client()
    
    if client:
        # Test with a simple API call
        try:
            details = client.get_customer_details()
            print(f"API call successful. Customer details: {details}")
        except Exception as e:
            print(f"API call failed: {e}")
    else:
        print("Failed to get authenticated client")