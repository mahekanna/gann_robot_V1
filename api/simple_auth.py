"""
Simple authentication module for ICICI Direct API
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

class SimpleAuth:
    """Simple authentication class that just loads and validates credentials"""
    
    def __init__(self, config_dir: str = 'config', env_file: str = '.env'):
        """
        Initialize authentication
        
        Args:
            config_dir: Directory containing configuration files
            env_file: Name of the environment file
        """
        self.config_dir = Path(config_dir)
        self.env_file = self.config_dir / env_file
        
        # Set up logging
        self.logger = logging.getLogger("simple_auth")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize credentials
        self.api_key = None
        self.api_secret = None
        self.totp_secret = None
        
    def load_credentials(self) -> bool:
        """
        Load credentials from environment variables or .env file
        
        Returns:
            True if credentials were loaded successfully, False otherwise
        """
        # Try to load from environment variables first
        self.api_key = os.environ.get('ICICI_API_KEY')
        self.api_secret = os.environ.get('ICICI_API_SECRET')
        self.totp_secret = os.environ.get('ICICI_TOTP_SECRET')
        
        # If not found in environment, try .env file
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
                    return False
            else:
                self.logger.error(f".env file not found at {self.env_file}")
                template_file = self.config_dir / ".env.template"
                if template_file.exists():
                    self.logger.info(f"Please copy {template_file} to {self.env_file} and add your credentials")
                return False
        
        # Check if all credentials are loaded
        if all([self.api_key, self.api_secret, self.totp_secret]):
            self.logger.info("Credentials loaded successfully")
            return True
        else:
            missing = []
            if not self.api_key:
                missing.append("ICICI_API_KEY")
            if not self.api_secret:
                missing.append("ICICI_API_SECRET")
            if not self.totp_secret:
                missing.append("ICICI_TOTP_SECRET")
                
            self.logger.error(f"Missing credentials: {', '.join(missing)}")
            return False
            
    def get_credentials(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Get the loaded credentials
        
        Returns:
            Tuple of (api_key, api_secret, totp_secret)
        """
        return self.api_key, self.api_secret, self.totp_secret
        
    def validate_credentials(self) -> bool:
        """
        Validate that credentials are in the expected format
        
        Returns:
            True if credentials appear valid, False otherwise
        """
        if not all([self.api_key, self.api_secret, self.totp_secret]):
            return False
            
        # Basic validation (could be enhanced based on exact format requirements)
        if len(self.api_key) < 8:
            self.logger.error("API key appears to be too short")
            return False
            
        if len(self.api_secret) < 8:
            self.logger.error("API secret appears to be too short")
            return False
            
        # TOTP secrets are typically base32 encoded
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        if not all(c in allowed_chars for c in self.totp_secret.upper()):
            self.logger.error("TOTP secret contains invalid characters")
            return False
            
        return True
