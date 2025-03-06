# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:40:01 2025

@author: mahes
"""

# File: api/breeze_patch.py
"""
Patch for breeze_connect package
Fixes issue with missing SECURITY_MASTER_URL
"""

import sys
import os
import logging
import importlib
from pathlib import Path

# Security master URL
SECURITY_MASTER_URL = "https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip"

def apply_patch():
    """
    Apply patch to breeze_connect package
    Sets the SECURITY_MASTER_URL in config module
    """
    logger = logging.getLogger("breeze_patch")
    logger.setLevel(logging.INFO)
    
    # Add console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    try:
        # Try to import breeze_connect.config
        try:
            import breeze_connect.config
            if not hasattr(breeze_connect.config, 'SECURITY_MASTER_URL'):
                setattr(breeze_connect.config, 'SECURITY_MASTER_URL', SECURITY_MASTER_URL)
                logger.info("Added SECURITY_MASTER_URL to breeze_connect.config dynamically")
            else:
                logger.info("SECURITY_MASTER_URL already exists in breeze_connect.config")
            return True
        except (ImportError, AttributeError):
            logger.info("breeze_connect.config module not found, trying to find package location")
        
        # Get breeze_connect package location
        breeze_spec = importlib.util.find_spec("breeze_connect")
        if not breeze_spec or not breeze_spec.origin:
            logger.error("Could not find breeze_connect package")
            return False
            
        breeze_path = Path(breeze_spec.origin).parent
        logger.info(f"Found breeze_connect at {breeze_path}")
        
        # Find config.py
        config_path = breeze_path / "config.py"
        
        if not config_path.exists():
            # Create a minimal config.py file
            with open(config_path, "w") as f:
                f.write(f"""# Auto-generated config file for breeze_connect
SECURITY_MASTER_URL = "{SECURITY_MASTER_URL}"
""")
            logger.info(f"Created config.py at {config_path}")
        else:
            # Check if SECURITY_MASTER_URL already exists
            with open(config_path, "r") as f:
                content = f.read()
                
            if "SECURITY_MASTER_URL" not in content:
                # Append SECURITY_MASTER_URL
                with open(config_path, "a") as f:
                    f.write(f"\n# Added by patch\nSECURITY_MASTER_URL = \"{SECURITY_MASTER_URL}\"\n")
                logger.info(f"Added SECURITY_MASTER_URL to {config_path}")
            else:
                logger.info(f"SECURITY_MASTER_URL already exists in {config_path}")
        
        # Try to import config to verify by forceably reloading
        if "breeze_connect.config" in sys.modules:
            logger.info("Reloading breeze_connect.config module")
            importlib.reload(sys.modules["breeze_connect.config"])
            
        # Verify it worked
        import breeze_connect.config
        if hasattr(breeze_connect.config, "SECURITY_MASTER_URL"):
            logger.info(f"Verification successful: breeze_connect.config.SECURITY_MASTER_URL = {breeze_connect.config.SECURITY_MASTER_URL}")
            return True
        else:
            logger.error("Verification failed: SECURITY_MASTER_URL not in breeze_connect.config module")
            return False
            
    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        return False

# Apply patch when module is imported
success = apply_patch()