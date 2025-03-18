import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional

def filter_dataframe(df: pd.DataFrame, min_date=None, max_date=None, 
                    min_value=None, max_value=None, value_column='close') -> pd.DataFrame:
    """
    Filter a DataFrame by date range and/or value range
    
    Args:
        df: DataFrame to filter
        min_date: Minimum date to include
        max_date: Maximum date to include
        min_value: Minimum value to include
        max_value: Maximum value to include
        value_column: Column to filter by value
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by date
    if min_date is not None and 'time' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['time'] >= min_date]
        
    if max_date is not None and 'time' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['time'] <= max_date]
        
    # Filter by value
    if min_value is not None and value_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[value_column] >= min_value]
        
    if max_value is not None and value_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[value_column] <= max_value]
        
    return filtered_df

def calculate_returns(df: pd.DataFrame, price_column='close', log_returns=False) -> pd.Series:
    """
    Calculate returns from a price series
    
    Args:
        df: DataFrame with price data
        price_column: Column containing price data
        log_returns: Whether to calculate log returns
        
    Returns:
        Series with returns
    """
    if log_returns:
        return np.log(df[price_column] / df[price_column].shift(1))
    else:
        return df[price_column].pct_change()

def extract_features_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for machine learning
    
    Args:
        df: DataFrame with OHLC and indicator data
        
    Returns:
        DataFrame with features
    """
    # Create a copy to avoid modifying the original
    feature_df = df.copy()
    
    # Price features
    if all(col in feature_df.columns for col in ['open', 'high', 'low', 'close']):
        # Candle body and wick ratios
        feature_df['body_ratio'] = abs(feature_df['close'] - feature_df['open']) / (feature_df['high'] - feature_df['low'])
        feature_df['upper_wick'] = (feature_df['high'] - feature_df[['open', 'close']].max(axis=1)) / (feature_df['high'] - feature_df['low'])
        feature_df['lower_wick'] = (feature_df[['open', 'close']].min(axis=1) - feature_df['low']) / (feature_df['high'] - feature_df['low'])
        
        # Returns
        feature_df['return_1'] = feature_df['close'].pct_change(1)
        feature_df['return_5'] = feature_df['close'].pct_change(5)
        
        # Volatility
        feature_df['highlow_delta'] = (feature_df['high'] - feature_df['low']) / feature_df['low']
        feature_df['close_delta_5'] = (feature_df['close'] - feature_df['close'].shift(5)) / feature_df['close'].shift(5)
        
    # Indicator features (if available)
    if 'rsi14' in feature_df.columns:
        feature_df['rsi_delta'] = feature_df['rsi14'] - feature_df['rsi14'].shift(1)
        feature_df['rsi_cross_30'] = ((feature_df['rsi14'] > 30) & (feature_df['rsi14'].shift(1) <= 30)).astype(int)
        feature_df['rsi_cross_70'] = ((feature_df['rsi14'] < 70) & (feature_df['rsi14'].shift(1) >= 70)).astype(int)
        
    if all(col in feature_df.columns for col in ['macd', 'macd_signal']):
        feature_df['macd_delta'] = feature_df['macd'] - feature_df['macd_signal']
        feature_df['macd_cross_up'] = ((feature_df['macd'] > feature_df['macd_signal']) & 
                                     (feature_df['macd'].shift(1) <= feature_df['macd_signal'].shift(1))).astype(int)
        feature_df['macd_cross_down'] = ((feature_df['macd'] < feature_df['macd_signal']) & 
                                       (feature_df['macd'].shift(1) >= feature_df['macd_signal'].shift(1))).astype(int)
        
    if all(col in feature_df.columns for col in ['sma20', 'sma50']):
        feature_df['sma_delta'] = feature_df['sma20'] - feature_df['sma50']
        feature_df['sma_cross_up'] = ((feature_df['sma20'] > feature_df['sma50']) & 
                                    (feature_df['sma20'].shift(1) <= feature_df['sma50'].shift(1))).astype(int)
        feature_df['sma_cross_down'] = ((feature_df['sma20'] < feature_df['sma50']) & 
                                      (feature_df['sma20'].shift(1) >= feature_df['sma50'].shift(1))).astype(int)
        
    # Drop NaN values created by indicators and features
    feature_df = feature_df.dropna()
    
    return feature_df