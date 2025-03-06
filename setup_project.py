# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:45:24 2025

@author: mahes
"""

#!/usr/bin/env python3
"""
Project setup utility for Gann Trading System
Creates the necessary directory structure and template files
Works on Windows, macOS, and Linux
"""

import os
from pathlib import Path
import sys

def setup_project():
    """
    Set up the project structure
    Creates directories and template files if they don't exist
    """
    print("Setting up Gann Trading System project structure...")
    
    # Root directories
    directories = [
        "config",
        "logs",
        "cache",
        "core/gann",
        "core/strategy",
        "api",
        "data",
        "tests",
        "backtesting",
        "paper_trading",
        "live_trading"
    ]
    
    # Create directories that don't exist
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")
    
    # Create __init__.py files if they don't exist
    init_dirs = [
        "core",
        "core/gann",
        "core/strategy",
        "api",
        "data",
        "tests",
        "backtesting",
        "paper_trading",
        "live_trading"
    ]
    
    for directory in init_dirs:
        init_path = Path(directory) / "__init__.py"
        if not init_path.exists():
            print(f"Creating file: {init_path}")
            with open(init_path, "w") as f:
                f.write(f"# {directory.replace('/', '.')} package\n")
        else:
            print(f"File already exists: {init_path}")
    
    # Create .env.template if it doesn't exist
    env_template_path = Path("config") / ".env.template"
    if not env_template_path.exists():
        print(f"Creating file: {env_template_path}")
        with open(env_template_path, "w") as f:
            f.write("""# ICICI Direct API credentials
ICICI_API_KEY=your_api_key_here
ICICI_API_SECRET=your_api_secret_here
ICICI_TOTP_SECRET=your_totp_secret_here
""")
    else:
        print(f"File already exists: {env_template_path}")
    
    # Check if .env exists
    env_path = Path("config") / ".env"
    if not env_path.exists():
        print(f"\nNOTE: .env file not found at {env_path}")
        print(f"Please copy {env_template_path} to {env_path} and add your credentials")
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Copy config/.env.template to config/.env if you haven't already")
    print("2. Add your ICICI Direct API credentials to config/.env")
    print("3. Run test_simple_auth.py to verify your credentials")

if __name__ == "__main__":
    setup_project()