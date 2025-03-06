# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:35:51 2025

@author: mahes
"""

# File: api/factory.py
"""
Factory module for creating API instances
"""

import logging
from typing import Optional

from .auth import BreezeAuth
from .security_master import SecurityMaster
from .client import BreezeClient
from .websocket import WebSocketClient

# Global instances
_auth_instance = None
_security_master_instance = None
_client_instance = None
_websocket_instance = None

def get_auth(force_new: bool = False, log_level: int = logging.INFO) -> BreezeAuth:
    """
    Get a BreezeAuth instance
    
    Args:
        force_new: Force creation of a new instance
        log_level: Logging level
        
    Returns:
        BreezeAuth instance
    """
    global _auth_instance
    
    if _auth_instance is None or force_new:
        _auth_instance = BreezeAuth(log_level=log_level)
        
    return _auth_instance

def get_security_master(force_new: bool = False, log_level: int = logging.INFO) -> SecurityMaster:
    """
    Get a SecurityMaster instance
    
    Args:
        force_new: Force creation of a new instance
        log_level: Logging level
        
    Returns:
        SecurityMaster instance
    """
    global _security_master_instance
    
    if _security_master_instance is None or force_new:
        _security_master_instance = SecurityMaster(log_level=log_level)
        
    return _security_master_instance

def get_client(force_new: bool = False, log_level: int = logging.INFO) -> BreezeClient:
    """
    Get a BreezeClient instance
    
    Args:
        force_new: Force creation of a new instance
        log_level: Logging level
        
    Returns:
        BreezeClient instance
    """
    global _client_instance
    
    if _client_instance is None or force_new:
        auth = get_auth(force_new=force_new, log_level=log_level)
        _client_instance = BreezeClient(auth=auth, log_level=log_level)
        
    return _client_instance

def get_websocket(force_new: bool = False, log_level: int = logging.INFO) -> WebSocketClient:
    """
    Get a WebSocketClient instance
    
    Args:
        force_new: Force creation of a new instance
        log_level: Logging level
        
    Returns:
        WebSocketClient instance
    """
    global _websocket_instance
    
    if _websocket_instance is None or force_new:
        auth = get_auth(force_new=force_new, log_level=log_level)
        _websocket_instance = WebSocketClient(auth=auth, log_level=log_level)
        
    return _websocket_instance