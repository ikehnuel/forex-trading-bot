import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class BaseStrategy:
    """Base class for all trading strategies"""
    def __init__(self, name="BaseStrategy"):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.parameters = {}
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        self.parameters = parameters
    
    def analyze(self, data: pd.DataFrame, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            data: DataFrame with OHLC and indicator data
            market_context: Additional market context information
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        # Base implementation returns no signal
        return {
            "strategy": self.name,
            "signal": "NEUTRAL",
            "confidence": 0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None
        }

class TrendFollowingStrategy(BaseStrategy):
    """Strategy for trending markets"""
    def __init__(self, **kwargs):
        super().__init__(name="Trend Following")
        
        # Default parameters
        self.parameters = {
            "ma_fast": 20,
            "ma_slow": 50,
            "atr_multiplier": 2.0,
            "min_trend_strength": 0.3,
            "profit_target_multiplier": 2.0  # Risk:reward ratio
        }
        
        # Override with any provided parameters
        self.parameters.update(kwargs)
    
    def analyze(self, data: pd.DataFrame, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for trend following setups
        
        Args:
            data: DataFrame with OHLC and indicator data
            market_context: Additional market context information
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        if data.empty or len(data) < 50:
            return {
                "strategy": self.name,
                "signal": "NEUTRAL",
                "confidence": 0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "message": "Insufficient data for analysis"
            }
        
        # Extract the most recent data
        recent_data = data.tail(20).copy()
        last_row = recent_data.iloc[-1]
        
        # Ensure we have necessary indicators
        if 'sma20' not in data.columns or 'sma50' not in data.columns or 'atr14' not in data.columns:
            # Calculate indicators if not present
            if 'sma20' not in data.columns:
                data['sma20'] = data['close'].rolling(window=self.parameters["ma_fast"]).mean()
            if 'sma50' not in data.columns:
                data['sma50'] = data['close'].rolling(window=self.parameters["ma_slow"]).mean()
            if 'atr14' not in data.columns:
                # Calculate ATR
                tr1 = data['high'] - data['low']
                tr2 = (data['high'] - data['close'].shift()).abs()
                tr3 = (data['low'] - data['close'].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                data['atr14'] = tr.rolling(window=14).mean()
                
            # Update with the recalculated data
            recent_data = data.tail(20).copy()
            last_row = recent_data.iloc[-1]
        
        # Initialize result
        result = {
            "strategy": self.name,
            "signal": "NEUTRAL",
            "confidence": 0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
        }
        
        # Get current values
        current_price = last_row['close']
        ma_fast = last_row['sma20']
        ma_slow = last_row['sma50']
        atr = last_row['atr14']
        
        # Get market context
        trend_strength = market_context.get('trend_strength', 0)
        
        # Check for bullish trend
        if (current_price > ma_fast > ma_slow) and trend_strength > self.parameters["min_trend_strength"]:
            # Calculate entry price (current price or slight improvement)
            entry_price = current_price
            
            # Calculate stop loss based on ATR
            stop_loss = current_price - (atr * self.parameters["atr_multiplier"])
            
            # Calculate take profit based on risk:reward ratio
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * self.parameters["profit_target_multiplier"])
            
            # Calculate confidence
            confidence = min(0.8, 0.5 + trend_strength)
            
            result.update({
                "signal": "BUY",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": self.parameters["profit_target_multiplier"],
                "message": "Bullish trend detected with sufficient strength"
            })
            
        # Check for bearish trend
        elif (current_price < ma_fast < ma_slow) and trend_strength < -self.parameters["min_trend_strength"]:
            # Calculate entry price
            entry_price = current_price
            
            # Calculate stop loss based on ATR
            stop_loss = current_price + (atr * self.parameters["atr_multiplier"])
            
            # Calculate take profit based on risk:reward ratio
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * self.parameters["profit_target_multiplier"])
            
            # Calculate confidence
            confidence = min(0.8, 0.5 + abs(trend_strength))
            
            result.update({
                "signal": "SELL",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": self.parameters["profit_target_multiplier"],
                "message": "Bearish trend detected with sufficient strength"
            })
        
        return result

class MeanReversionStrategy(BaseStrategy):
    """Strategy for ranging markets"""
    def __init__(self, **kwargs):
        super().__init__(name="Mean Reversion")
        
        # Default parameters
        self.parameters = {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "atr_multiplier": 1.5,
            "profit_target_multiplier": 1.5,
            "min_range_score": 0.6
        }
        
        # Override with any provided parameters
        self.parameters.update(kwargs)
    
    def analyze(self, data: pd.DataFrame, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for mean reversion setups
        
        Args:
            data: DataFrame with OHLC and indicator data
            market_context: Additional market context information
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        if data.empty or len(data) < 30:
            return {
                "strategy": self.name,
                "signal": "NEUTRAL",
                "confidence": 0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "message": "Insufficient data for analysis"
            }
        
        # Extract the most recent data
        recent_data = data.tail(30).copy()
        last_row = recent_data.iloc[-1]
        
        # Ensure we have necessary indicators
        if 'rsi14' not in data.columns:
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.parameters["rsi_period"]).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.parameters["rsi_period"]).mean()
            rs = gain / loss
            data['rsi14'] = 100 - (100 / (1 + rs))
        
        if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
            # Calculate Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=self.parameters["bb_period"]).mean()
            std = data['close'].rolling(window=self.parameters["bb_period"]).std()
            data['bb_upper'] = data['bb_middle'] + (std * self.parameters["bb_std_dev"])
            data['bb_lower'] = data['bb_middle'] - (std * self.parameters["bb_std_dev"])
        
        if 'atr14' not in data.columns:
            # Calculate ATR
            tr1 = data['high'] - data['low']
            tr2 = (data['high'] - data['close'].shift()).abs()
            tr3 = (data['low'] - data['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['atr14'] = tr.rolling(window=14).mean()
            
        # Update with the recalculated data
        recent_data = data.tail(30).copy()
        last_row = recent_data.iloc[-1]
        
        # Initialize result
        result = {
            "strategy": self.name,
            "signal": "NEUTRAL",
            "confidence": 0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
        }
        
        # Get current values
        current_price = last_row['close']
        rsi = last_row['rsi14']
        bb_upper = last_row['bb_upper']
        bb_lower = last_row['bb_lower']
        bb_middle = last_row['bb_middle']
        atr = last_row['atr14']
        
        # Check if market is in a ranging state
        is_range_bound = market_context.get('range_bound', False)
        range_score = market_context.get('range_score', 0)
        
        # Only generate signals in ranging markets
        if not is_range_bound and range_score < self.parameters["min_range_score"]:
            result["message"] = "Market not in ranging state"
            return result
        
        # Check for oversold condition (buy signal)
        if rsi <= self.parameters["rsi_oversold"] and current_price <= bb_lower:
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss = current_price - (atr * self.parameters["atr_multiplier"])
            
            # Take profit at middle band or better
            take_profit = max(bb_middle, current_price + (atr * self.parameters["profit_target_multiplier"]))
            
            # Calculate risk-reward ratio
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on how extreme the oversold condition is
            confidence = min(0.8, 0.5 + ((self.parameters["rsi_oversold"] - rsi) / 10))
            
            result.update({
                "signal": "BUY",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio,
                "message": "Oversold condition in ranging market"
            })
            
        # Check for overbought condition (sell signal)
        elif rsi >= self.parameters["rsi_overbought"] and current_price >= bb_upper:
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss = current_price + (atr * self.parameters["atr_multiplier"])
            
            # Take profit at middle band or better
            take_profit = min(bb_middle, current_price - (atr * self.parameters["profit_target_multiplier"]))
            
            # Calculate risk-reward ratio
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on how extreme the overbought condition is
            confidence = min(0.8, 0.5 + ((rsi - self.parameters["rsi_overbought"]) / 10))
            
            result.update({
                "signal": "SELL",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio,
                "message": "Overbought condition in ranging market"
            })
        
        return result

class BreakoutStrategy(BaseStrategy):
    """Strategy for breakout conditions"""
    def __init__(self, **kwargs):
        super().__init__(name="Breakout")
        
        # Default parameters
        self.parameters = {
            "lookback_period": 20,
            "confirmation_candles": 1,
            "volume_confirmation": True,
            "min_range_days": 5,
            "atr_multiplier": 1.0,
            "profit_target_multiplier": 2.0,
            "min_breakout_strength": 0.5
        }
        
        # Override with any provided parameters
        self.parameters.update(kwargs)
    
    def analyze(self, data: pd.DataFrame, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for breakout setups
        
        Args:
            data: DataFrame with OHLC and indicator data
            market_context: Additional market context information
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        if data.empty or len(data) < self.parameters["lookback_period"] + 10:
            return {
                "strategy": self.name,
                "signal": "NEUTRAL",
                "confidence": 0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "message": "Insufficient data for analysis"
            }
        
        # Extract necessary data
        lookback = self.parameters["lookback_period"]
        recent_data = data.tail(lookback + 10).copy()
        
        # Calculate atr if not present
        if 'atr14' not in recent_data.columns:
            tr1 = recent_data['high'] - recent_data['low']
            tr2 = (recent_data['high'] - recent_data['close'].shift()).abs()
            tr3 = (recent_data['low'] - recent_data['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            recent_data['atr14'] = tr.rolling(window=14).mean()
        
        # Get the most recent candle
        current_candle = recent_data.iloc[-1]
        
        # Determine the range before the potential breakout
        range_start = len(recent_data) - lookback - 1
        range_end = len(recent_data) - self.parameters["confirmation_candles"] - 1
        
        if range_start < 0 or range_end < 0 or range_end <= range_start:
            return {
                "strategy": self.name,
                "signal": "NEUTRAL",
                "confidence": 0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "message": "Invalid range for breakout analysis"
            }
        
        # Get the price range
        price_range = recent_data.iloc[range_start:range_end]
        range_high = price_range['high'].max()
        range_low = price_range['low'].min()
        range_size = range_high - range_low
        
        # Get current price and ATR
        current_price = current_candle['close']
        atr = current_candle['atr14']
        
        # Check if we have a valid range
        if len(price_range) < self.parameters["min_range_days"]:
            return {
                "strategy": self.name,
                "signal": "NEUTRAL",
                "confidence": 0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "message": f"Range too short: {len(price_range)} days"
            }
        
        # Check for volume confirmation if required
        volume_confirmed = True
        if self.parameters["volume_confirmation"] and 'volume' in recent_data.columns:
            avg_volume = price_range['volume'].mean()
            current_volume = current_candle['volume']
            volume_confirmed = current_volume > avg_volume * 1.5
        
        # Check if price is breaking out of the range
        buffer = atr * 0.1  # Small buffer to avoid false breakouts
        
        # Initialize result
        result = {
            "strategy": self.name,
            "signal": "NEUTRAL",
            "confidence": 0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
        }
        
        # Get breakout detection from market context
        is_breakout = market_context.get('is_breakout', False)
        is_breakout_up = market_context.get('is_breakout_up', False)
        is_breakout_down = market_context.get('is_breakout_down', False)
        breakout_strength = market_context.get('breakout_strength', 0)
        
        # Check if market context confirms breakout
        if not is_breakout or breakout_strength < self.parameters["min_breakout_strength"]:
            # Alternatively, check using price data
            is_breakout_up = current_price > (range_high + buffer)
            is_breakout_down = current_price < (range_low - buffer)
            is_breakout = is_breakout_up or is_breakout_down
        
        # Check for upward breakout
        if is_breakout_up and volume_confirmed:
            # Calculate entry, stop loss, and take profit
            entry_price = current_price  # Or slightly above for confirmation
            
            # Stop loss below the breakout level or recent low
            stop_loss = max(range_high - (atr * self.parameters["atr_multiplier"]), 
                          price_range.iloc[-5:]['low'].min())
            
            # Take profit based on the range size
            take_profit = entry_price + (range_size * self.parameters["profit_target_multiplier"])
            
            # Calculate confidence
            confidence = min(0.8, 0.5 + breakout_strength)
            
            result.update({
                "signal": "BUY",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0,
                "message": "Upward breakout detected"
            })
            
        # Check for downward breakout
        elif is_breakout_down and volume_confirmed:
            # Calculate entry, stop loss, and take profit
            entry_price = current_price  # Or slightly below for confirmation
            
            # Stop loss above the breakout level or recent high
            stop_loss = min(range_low + (atr * self.parameters["atr_multiplier"]), 
                          price_range.iloc[-5:]['high'].max())
            
            # Take profit based on the range size
            take_profit = entry_price - (range_size * self.parameters["profit_target_multiplier"])
            
            # Calculate confidence
            confidence = min(0.8, 0.5 + breakout_strength)
            
            result.update({
                "signal": "SELL",
                "confidence": confidence,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0,
                "message": "Downward breakout detected"
            })
        
        return result

class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
        """
        Create a strategy instance based on type
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Strategy instance
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type == "trend_following":
            return TrendFollowingStrategy(**kwargs)
        elif strategy_type == "mean_reversion":
            return MeanReversionStrategy(**kwargs)
        elif strategy_type == "breakout":
            return BreakoutStrategy(**kwargs)
        else:
            # Default to base strategy
            return BaseStrategy(name=strategy_type)
    
    @staticmethod
    def create_strategy_for_regime(regime: str, **kwargs) -> BaseStrategy:
        """
        Create a strategy based on market regime
        
        Args:
            regime: Market regime
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Strategy instance appropriate for the regime
        """
        if regime in ["trending_up", "trending_down"]:
            return TrendFollowingStrategy(**kwargs)
        elif regime == "ranging":
            return MeanReversionStrategy(**kwargs)
        elif "breakout" in regime:
            return BreakoutStrategy(**kwargs)
        else:
            # Default to trend following for unknown regimes
            return TrendFollowingStrategy(**kwargs)