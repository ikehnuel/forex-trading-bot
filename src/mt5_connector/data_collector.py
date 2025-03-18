import time
import logging
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TIMEFRAME_MAP, MARKET_DATA_DIR, logger
from mt5_connector.connection import MT5Connector

class DataCollector:
    """
    Handles data collection from MetaTrader 5, including
    OHLC data and technical indicators.
    """
    def __init__(self, mt5_connector=None):
        self.logger = logging.getLogger(__name__)
        
        # Use provided connector or create a new one
        if mt5_connector is None:
            self.mt5 = MT5Connector()
        else:
            self.mt5 = mt5_connector
            
        # Ensure MT5 is connected
        if not self.mt5.check_connection():
            raise ConnectionError("Could not connect to MetaTrader 5")
            
    def get_ohlc_data(self, symbol, timeframe, num_candles=100, with_indicators=True):
        """
        Get OHLC data from MT5 with optional indicators
        
        Args:
            symbol (str): Symbol to get data for
            timeframe (str): Timeframe (e.g. 'M1', 'H1', 'D1')
            num_candles (int): Number of candles to retrieve
            with_indicators (bool): Whether to calculate indicators
            
        Returns:
            pandas.DataFrame: DataFrame with OHLC data and indicators
        """
        try:
            # Convert timeframe string to MT5 timeframe constant
            tf = TIMEFRAME_MAP.get(timeframe)
            if tf is None:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
                
            # Get OHLC data
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_candles)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get OHLC data for {symbol} {timeframe}: {mt5.last_error()}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate indicators if requested
            if with_indicators:
                df = self._add_indicators(df)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {symbol} {timeframe}: {e}")
            return None
            
    def get_tick_data(self, symbol, num_ticks=1000):
        """
        Get recent tick data from MT5
        
        Args:
            symbol (str): Symbol to get data for
            num_ticks (int): Number of ticks to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame with tick data
        """
        try:
            # Get current symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None
                
            # Get ticks
            ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(days=1), num_ticks, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                self.logger.error(f"Failed to get tick data for {symbol}: {mt5.last_error()}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {e}")
            return None
            
    def get_current_prices(self, symbols):
        """
        Get current prices for a list of symbols
        
        Args:
            symbols (list): List of symbols to get prices for
            
        Returns:
            dict: Dictionary with bid and ask prices for each symbol
        """
        if not isinstance(symbols, list):
            symbols = [symbols]
            
        try:
            prices = {}
            
            for symbol in symbols:
                tick = mt5.symbol_info_tick(symbol)
                
                if tick is None:
                    self.logger.warning(f"Could not get tick data for {symbol}")
                    continue
                    
                prices[symbol] = {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': datetime.fromtimestamp(tick.time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'spread': tick.ask - tick.bid
                }
                
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting current prices: {e}")
            return {}
            
    def get_multiple_timeframes(self, symbol, timeframes, num_candles=100):
        """
        Get data for multiple timeframes
        
        Args:
            symbol (str): Symbol to get data for
            timeframes (list): List of timeframes
            num_candles (int): Number of candles to retrieve
            
        Returns:
            dict: Dictionary with dataframes for each timeframe
        """
        try:
            data = {}
            
            for tf in timeframes:
                # Get data for this timeframe
                df = self.get_ohlc_data(symbol, tf, num_candles)
                
                if df is not None:
                    data[tf] = df
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting multiple timeframes for {symbol}: {e}")
            return {}
            
    def save_data_to_csv(self, df, symbol, timeframe):
        """
        Save OHLC data to CSV file
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLC data
            symbol (str): Symbol
            timeframe (str): Timeframe
            
        Returns:
            bool: Success or failure
        """
        try:
            # Create directory if it doesn't exist
            symbol_dir = os.path.join(MARKET_DATA_DIR, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Create filename
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(symbol_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved data to {filepath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}")
            return False
            
    def load_data_from_csv(self, symbol, timeframe, date=None):
        """
        Load OHLC data from CSV file
        
        Args:
            symbol (str): Symbol
            timeframe (str): Timeframe
            date (str): Date in YYYYMMDD format, or None for latest
            
        Returns:
            pandas.DataFrame: DataFrame with OHLC data
        """
        try:
            # Get symbol directory
            symbol_dir = os.path.join(MARKET_DATA_DIR, symbol)
            if not os.path.exists(symbol_dir):
                self.logger.warning(f"No data directory found for {symbol}")
                return None
                
            # Find the file
            if date is None:
                # Get latest file
                files = [f for f in os.listdir(symbol_dir) if f.startswith(f"{symbol}_{timeframe}")]
                files.sort(reverse=True)
                
                if not files:
                    self.logger.warning(f"No data files found for {symbol} {timeframe}")
                    return None
                    
                filepath = os.path.join(symbol_dir, files[0])
            else:
                # Get specific date
                filepath = os.path.join(symbol_dir, f"{symbol}_{timeframe}_{date}.csv")
                
                if not os.path.exists(filepath):
                    self.logger.warning(f"No data file found for {symbol} {timeframe} on {date}")
                    return None
                    
            # Load from CSV
            df = pd.read_csv(filepath)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from CSV: {e}")
            return None
            
    def _add_indicators(self, df):
        """
        Add technical indicators to dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLC data
            
        Returns:
            pandas.DataFrame: DataFrame with indicators added
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_indicators = df.copy()
            
            # Simple Moving Averages
            df_indicators['sma20'] = df_indicators['close'].rolling(window=20).mean()
            df_indicators['sma50'] = df_indicators['close'].rolling(window=50).mean()
            df_indicators['sma200'] = df_indicators['close'].rolling(window=200).mean() if len(df) >= 200 else np.nan
            
            # Exponential Moving Averages
            df_indicators['ema20'] = df_indicators['close'].ewm(span=20, adjust=False).mean()
            df_indicators['ema50'] = df_indicators['close'].ewm(span=50, adjust=False).mean()
            df_indicators['ema200'] = df_indicators['close'].ewm(span=200, adjust=False).mean() if len(df) >= 200 else np.nan
            
            # RSI
            delta = df_indicators['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['rsi14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
            std = df_indicators['close'].rolling(window=20).std()
            df_indicators['bb_upper'] = df_indicators['bb_middle'] + 2 * std
            df_indicators['bb_lower'] = df_indicators['bb_middle'] - 2 * std
            
            # Average True Range (ATR)
            high_low = df_indicators['high'] - df_indicators['low']
            high_close = (df_indicators['high'] - df_indicators['close'].shift()).abs()
            low_close = (df_indicators['low'] - df_indicators['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_indicators['atr14'] = true_range.rolling(window=14).mean()
            
            # MACD
            ema12 = df_indicators['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_indicators['close'].ewm(span=26, adjust=False).mean()
            df_indicators['macd'] = ema12 - ema26
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
            df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']
            
            # Stochastic Oscillator
            low_14 = df_indicators['low'].rolling(window=14).min()
            high_14 = df_indicators['high'].rolling(window=14).max()
            df_indicators['stoch_k'] = 100 * ((df_indicators['close'] - low_14) / (high_14 - low_14))
            df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()
            
            # Ichimoku Cloud
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df_indicators['high'].rolling(window=9).max()
            low_9 = df_indicators['low'].rolling(window=9).min()
            df_indicators['ichimoku_tenkan'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df_indicators['high'].rolling(window=26).max()
            low_26 = df_indicators['low'].rolling(window=26).min()
            df_indicators['ichimoku_kijun'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df_indicators['ichimoku_senkou_a'] = ((df_indicators['ichimoku_tenkan'] + df_indicators['ichimoku_kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df_indicators['high'].rolling(window=52).max()
            low_52 = df_indicators['low'].rolling(window=52).min()
            df_indicators['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted backwards
            df_indicators['ichimoku_chikou'] = df_indicators['close'].shift(-26)
            
            return df_indicators
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            return df