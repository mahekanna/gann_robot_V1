# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:56:11 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Standalone authentication test for ICICI Direct API
Completely independent of existing code to avoid circular imports
"""

import os
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("auth_test")

class AuthTester:
    """Simple authentication tester with no external dependencies"""
    
    def __init__(self):
        # Check config directory
        self.config_dir = Path("config")
        if not self.config_dir.exists():
            logger.info(f"Creating config directory: {self.config_dir}")
            self.config_dir.mkdir(exist_ok=True)
            
        # Set paths
        self.env_file = self.config_dir / ".env"
        self.template_file = self.config_dir / ".env.template"
        
        # Initialize credentials
        self.api_key = None
        self.api_secret = None
        self.totp_secret = None
        
    def check_env_files(self):
        """Check if necessary env files exist"""
        # Check if .env exists
        if not self.env_file.exists():
            if self.template_file.exists():
                logger.warning(f".env file not found but template exists.")
                logger.info(f"Please copy {self.template_file} to {self.env_file} and add your credentials.")
            else:
                logger.warning(f"Neither .env nor .env.template found.")
                logger.info("Creating .env.template file...")
                
                # Create template file
                with open(self.template_file, "w") as f:
                    f.write("""# ICICI Direct API credentials
ICICI_API_KEY=your_api_key_here
ICICI_API_SECRET=your_api_secret_here
ICICI_TOTP_SECRET=your_totp_secret_here
""")
                
                logger.info(f"Template created at {self.template_file}")
                logger.info(f"Please copy to {self.env_file} and add your credentials.")
                
            return False
        return True
        
    def load_credentials(self):
        """Load credentials from environment or file"""
        # Try to load from environment variables first
        self.api_key = os.environ.get('ICICI_API_KEY')
        self.api_secret = os.environ.get('ICICI_API_SECRET')
        self.totp_secret = os.environ.get('ICICI_TOTP_SECRET')
        
        # If not found in environment, try .env file
        if not all([self.api_key, self.api_secret, self.totp_secret]):
            if self.env_file.exists():
                logger.info(f"Loading credentials from {self.env_file}")
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
                    logger.error(f"Error reading .env file: {e}")
                    return False
            else:
                logger.error(f".env file not found at {self.env_file}")
                return False
        
        # Check if all credentials are loaded
        if all([self.api_key, self.api_secret, self.totp_secret]):
            logger.info("Credentials loaded successfully")
            return True
        else:
            missing = []
            if not self.api_key:
                missing.append("ICICI_API_KEY")
            if not self.api_secret:
                missing.append("ICICI_API_SECRET")
            if not self.totp_secret:
                missing.append("ICICI_TOTP_SECRET")
                
            logger.error(f"Missing credentials: {', '.join(missing)}")
            return False
            
    def validate_credentials(self):
        """Validate credential format"""
        if not all([self.api_key, self.api_secret, self.totp_secret]):
            return False
            
        # Basic validation (could be enhanced based on exact format requirements)
        if len(self.api_key) < 8:
            logger.error("API key appears to be too short")
            return False
            
        if len(self.api_secret) < 8:
            logger.error("API secret appears to be too short")
            return False
            
        # TOTP secrets are typically base32 encoded
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        if not all(c in allowed_chars for c in self.totp_secret.upper()):
            logger.error("TOTP secret contains invalid characters")
            return False
            
        return True
        
    def display_credentials(self):
        """Display partial credentials for verification"""
        if all([self.api_key, self.api_secret, self.totp_secret]):
            logger.info(f"API Key: {self.api_key[:4]}...{self.api_key[-4:] if len(self.api_key) > 8 else ''}")
            logger.info(f"API Secret: {self.api_secret[:2]}...{self.api_secret[-2:] if len(self.api_secret) > 4 else ''}")
            logger.info(f"TOTP Secret: {self.totp_secret[:2]}...{self.totp_secret[-2:] if len(self.totp_secret) > 4 else ''}")
        else:
            logger.error("No credentials to display")
            
    def run_tests(self):
        """Run all auth tests"""
        print("\n=== ICICI Direct API Authentication Test ===")
        
        # Step 1: Check env files
        if not self.check_env_files():
            print("\nPlease create the .env file with your credentials and run this test again.")
            return False
            
        # Step 2: Load credentials
        if not self.load_credentials():
            print("\nFailed to load credentials. Please check your .env file.")
            return False
            
        # Step 3: Validate credentials
        if not self.validate_credentials():
            print("\nCredential validation failed. Please check your credentials format.")
            return False
            
        # Step 4: Display partial credentials
        print("\nCredential validation successful!")
        print("Partial credentials for verification:")
        self.display_credentials()
        
        print("\nAuth test completed successfully! Your credentials are properly configured.")
        return True

def main():
    """Main function"""
    tester = AuthTester()
    success = tester.run_tests()
    
    if success:
        print("\nYou can now proceed with API implementation.")
    else:
        print("\nPlease fix the authentication issues before proceeding.")

if __name__ == "__main__":
    main()