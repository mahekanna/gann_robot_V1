# -*- coding: utf-8 -*-
"""
Created on Sun Mar 02 08:51:09 2025

@author: Mahesh Naidu
"""

"""
API module for ICICI Direct Breeze API
Handles authentication, market data, and order execution
"""

# Import order matters to avoid circular dependencies
from .auth import BreezeAuth
from .security_master import SecurityMaster
from .client import BreezeClient
from .websocket import WebSocketClient

# Factory functions for easy access
from .factory import (
    get_auth,
    get_security_master,
    get_client,
    get_websocket
)

__all__ = [
    'BreezeAuth',
    'SecurityMaster',
    'BreezeClient',
    'WebSocketClient',
    'get_auth',
    'get_security_master',
    'get_client',
    'get_websocket'
]