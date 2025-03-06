# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 20:54:14 2025

@author: mahes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:50:17 2024

@author: mnaidu
"""

# autologin.py
import os
import csv
import datetime
import pyotp
from breeze_connect import BreezeConnect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment variables
api_key = os.getenv('ICICI_API_KEY')
api_secret = os.getenv('ICICI_API_SECRET')
totp_secret = os.getenv('ICICI_TOTP_SECRET')

session_file = 'session_key.csv'

# Function to generate TOTP dynamically
def generate_totp(secret_key):
    totp = pyotp.TOTP(secret_key)
    return totp.now()

# Function to save session key to CSV
def save_session_key(session_key):
    with open(session_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([session_key, datetime.datetime.now().isoformat()])

# Function to read session key from CSV
def load_session_key():
    if os.path.exists(session_file):
        try:
            with open(session_file, mode='r') as file:
                reader = csv.reader(file)
                session_data = next(reader, [None, None])
                session_key, timestamp = session_data
                if session_key and timestamp:
                    generated_at = datetime.datetime.fromisoformat(timestamp)
                    return session_key, generated_at
                else:
                    print("Invalid or missing session key or timestamp.")
        except (ValueError, IndexError) as e:
            print(f"Error reading session key file: {e}")
            return None, None
    return None, None

# Function to generate a new session and save it
def breeze_auto_login(api_key, api_secret, totp_secret):
    # Initialize BreezeConnect with API key
    breeze = BreezeConnect(api_key=api_key)

    # Check if session key is saved and reuse it if valid
    session_key, generated_at = load_session_key()
    if session_key and generated_at:
        time_difference = datetime.datetime.now() - generated_at
        if time_difference < datetime.timedelta(hours=24):
            try:
                print("Using saved session key.")
                breeze.generate_session(api_secret=api_secret, session_token=session_key)
                return breeze
            except Exception as e:
                print(f"Failed to use saved session key: {e}")
    
    # If no valid session key or saved key failed, generate a new one
    print("Generating new session key.")
    totp_code = generate_totp(totp_secret)
    print(f"Generated TOTP: {totp_code}")
    
    # Get session token manually (open login URL, copy session token from redirect)
    print(f"Login URL: https://api.icicidirect.com/apiuser/login?api_key={api_key}")
    session_token = input("Enter the session token obtained from login URL after TOTP: ")

    # Save session key to file
    save_session_key(session_token)

    # Generate session with the API secret and the new session token
    try:
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        print("Session key saved successfully.")
        return breeze
    except Exception as e:
        print(f"Failed to generate new session: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Perform auto-login using stored credentials
    breeze = breeze_auto_login(api_key, api_secret, totp_secret)

    if breeze:
        print("Successfully authenticated!")
    else:
        print("Authentication failed. Please check your credentials or session.")   
