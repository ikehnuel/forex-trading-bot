import time
import logging
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, logger

class MT5Connector:
    """
    Handles connection and basic interaction with MetaTrader 5 terminal.
    """
    def __init__(self):
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        self.connection_attempts = 0
        self.max_attempts = 5
        self.connect()
        
    def connect(self):
        """Initialize connection to MetaTrader 5 terminal"""
        try:
            # Initialize MT5 if not already done
            if not self.initialized:
                # Shutdown any existing MT5 connections
                mt5.shutdown()
                
                # Set path if provided
                if MT5_PATH:
                    self.logger.info(f"Using custom MT5 path: {MT5_PATH}")
                    mt5_init = mt5.initialize(path=MT5_PATH)
                else:
                    mt5_init = mt5.initialize()
                
                if not mt5_init:
                    self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                    self.connection_attempts += 1
                    if self.connection_attempts < self.max_attempts:
                        self.logger.info(f"Retrying MT5 connection (attempt {self.connection_attempts}/{self.max_attempts})")
                        time.sleep(5)
                        return self.connect()
                    return False
                
                self.logger.info("MT5 initialized successfully")
                
                # Login if credentials are provided
                if MT5_LOGIN and MT5_PASSWORD:
                    login_result = mt5.login(
                        login=MT5_LOGIN,
                        password=MT5_PASSWORD,
                        server=MT5_SERVER
                    )
                    
                    if not login_result:
                        self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                        return False
                    
                    self.logger.info(f"MT5 login successful as {MT5_LOGIN} on {MT5_SERVER}")
                
                self.initialized = True
                self.connection_attempts = 0
                return True
            
            # Check if MT5 is still connected
            if not mt5.terminal_info():
                self.logger.warning("MT5 connection lost, attempting to reconnect")
                self.initialized = False
                return self.connect()
                
            return True
                
        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Shutdown connection to MetaTrader 5"""
        mt5.shutdown()
        self.initialized = False
        self.logger.info("MT5 connection shutdown")
    
    def check_connection(self):
        """Check if MT5 is still connected and reconnect if needed"""
        if not self.initialized or not mt5.terminal_info():
            self.logger.warning("MT5 connection check failed, attempting to reconnect")
            return self.connect()
        return True
    
    def get_account_info(self):
        """Get account information from MT5"""
        if not self.check_connection():
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None
            
            # Convert named tuple to dict for easier handling
            info_dict = {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'currency': account_info.currency,
                'name': account_info.name,
                'server': account_info.server
            }
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_terminal_info(self):
        """Get terminal information"""
        if not self.check_connection():
            return None
            
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error(f"Failed to get terminal info: {mt5.last_error()}")
                return None
                
            # Convert named tuple to dict
            info_dict = {
                'path': terminal_info.path,
                'data_path': terminal_info.data_path,
                'connected': terminal_info.connected,
                'community_account': terminal_info.community_account,
                'community_connection': terminal_info.community_connection,
                'build': terminal_info.build,
                'retransmission': terminal_info.retransmission,
                'build_date': terminal_info.build_date,
                'community_balance': terminal_info.community_balance,
                'dlls_allowed': terminal_info.dlls_allowed,
                'trade_allowed': terminal_info.trade_allowed,
                'max_bars_in_chart': terminal_info.maxbars
            }
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"Error getting terminal info: {e}")
            return None
    
    def get_symbols(self):
        """Get list of available symbols"""
        if not self.check_connection():
            return []
            
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                self.logger.error(f"Failed to get symbols: {mt5.last_error()}")
                return []
                
            # Extract symbol names
            symbol_names = [symbol.name for symbol in symbols]
            return symbol_names
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol):
        """Get detailed information about a specific symbol"""
        if not self.check_connection():
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Failed to get info for symbol {symbol}: {mt5.last_error()}")
                return None
                
            # Convert to dict
            info_dict = {
                'name': symbol_info.name,
                'description': symbol_info.description,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_tick_size': symbol_info.trade_tick_size,
                'trade_tick_value': symbol_info.trade_tick_value,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_mode': symbol_info.trade_mode,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'point': symbol_info.point,
                'swap_mode': symbol_info.swap_mode,
                'swap_long': symbol_info.swap_long,
                'swap_short': symbol_info.swap_short,
                'is_forex': 'forex' in symbol_info.path.lower()  # Check if this is a forex pair
            }
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def is_market_open(self, symbol):
        """Check if the market is currently open for the given symbol"""
        if not self.check_connection():
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # Check if trading is enabled for this symbol
            return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED
            
        except Exception as e:
            self.logger.error(f"Error checking if market is open for {symbol}: {e}")
            return False
    
    def ping(self):
        """Simple ping to check if MT5 is responsive"""
        if not self.check_connection():
            return False
            
        try:
            # Just get a simple piece of info to check responsiveness
            return mt5.terminal_info() is not None
            
        except Exception as e:
            self.logger.error(f"Error pinging MT5: {e}")
            return False