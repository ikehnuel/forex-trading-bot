import pandas as pd
import numpy as np
from scipy import stats
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# For stationarity test
try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    adfuller = None
    logging.warning("statsmodels not installed. Stationarity test will be unavailable.")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger

class MarketRegimeDetector:
    """
    Detects the current market regime (trending, ranging, volatile, or breakout)
    based on statistical analysis of price action.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volatility_lookback = 20
        self.trend_lookback = 50
        self.range_lookback = 14
        self.regime_history = []
        
    def detect_regime(self, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime
        
        Args:
            ohlc_data: DataFrame with OHLC data and indicators
            
        Returns:
            Dict with regime information:
            - trend_strength: -1 to 1 (strong downtrend to strong uptrend)
            - volatility: 'low', 'normal', 'high'
            - range_bound: True/False
            - momentum: -1 to 1 (strong negative to strong positive)
            - regime: 'trending_up', 'trending_down', 'ranging', 'breakout', 'reversal'
        """
        try:
            if ohlc_data is None or len(ohlc_data) < self.volatility_lookback:
                self.logger.warning("Insufficient data for regime detection")
                return self._get_default_regime()
                
            # Make a copy to avoid modifying the original DataFrame
            df = ohlc_data.copy()
            
            # Ensure we have required indicators, calculate if missing
            if 'sma20' not in df.columns:
                df['sma20'] = df['close'].rolling(window=20).mean()
            if 'sma50' not in df.columns:
                df['sma50'] = df['close'].rolling(window=50).mean()
            if 'atr14' not in df.columns:
                df['atr14'] = self._calculate_atr(df, 14)
                
            # 1. Detect trend strength
            trend_info = self._detect_trend_strength(df)
            trend_direction = trend_info['direction']
            trend_strength = trend_info['strength']
                
            # 2. Detect volatility regime
            volatility_regime = self._detect_volatility(df)
                
            # 3. Detect if market is range-bound
            is_range_bound = self._detect_range_bound(df)
                
            # 4. Detect momentum
            normalized_momentum = self._detect_momentum(df)
                
            # 5. Breakout detection
            breakout_info = self._detect_breakout(df)
            is_breakout_up = breakout_info['is_breakout_up']
            is_breakout_down = breakout_info['is_breakout_down']
            is_breakout = is_breakout_up or is_breakout_down
                
            # 6. Reversal detection
            is_reversal = self._detect_reversal(df)
                
            # 7. Determine overall regime
            regime = self._determine_regime(
                trend_strength, 
                is_range_bound, 
                is_breakout_up, 
                is_breakout_down, 
                is_reversal
            )
            
            # Get ATR for volatility reference
            latest_atr = df['atr14'].iloc[-1] if not df['atr14'].isna().all() else 0
            
            # Compile results
            result = {
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volatility': volatility_regime,
                'range_bound': is_range_bound,
                'momentum': normalized_momentum,
                'regime': regime,
                'atr': latest_atr,
                'is_breakout': is_breakout,
                'is_reversal': is_reversal,
                'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to history
            self.regime_history.append(result)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return self._get_default_regime()
        
    def get_optimal_strategy(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the optimal trading strategy based on the detected market regime
        
        Args:
            regime_info: Result from detect_regime()
            
        Returns:
            Dict with strategy recommendations
        """
        regime = regime_info['regime']
        volatility = regime_info['volatility']
        
        if regime == 'trending_up':
            return {
                'strategy': 'trend_following',
                'entry': 'pullbacks', 
                'stop_loss': 'wide',
                'take_profit': 'trailing',
                'timeframes': ['higher'],
                'indicators': ['moving_averages', 'macd', 'adx'],
                'confidence': 0.8 if volatility != 'high' else 0.6
            }
            
        elif regime == 'trending_down':
            return {
                'strategy': 'trend_following',
                'entry': 'rallies',
                'stop_loss': 'wide',
                'take_profit': 'trailing',
                'timeframes': ['higher'],
                'indicators': ['moving_averages', 'macd', 'adx'],
                'confidence': 0.8 if volatility != 'high' else 0.6
            }
            
        elif regime == 'ranging':
            return {
                'strategy': 'mean_reversion',
                'entry': 'extremes',
                'stop_loss': 'beyond_range',
                'take_profit': 'fixed',
                'timeframes': ['current', 'lower'],
                'indicators': ['rsi', 'bollinger', 'stochastic'],
                'confidence': 0.7 if volatility == 'low' else 0.5
            }
            
        elif 'breakout' in regime:
            return {
                'strategy': 'breakout',
                'entry': 'confirmation',
                'stop_loss': 'tight',
                'take_profit': 'trailing',
                'timeframes': ['current', 'lower'],
                'indicators': ['volume', 'atr', 'macd'],
                'confidence': 0.6 if volatility == 'high' else 0.4
            }
            
        elif regime == 'reversal':
            return {
                'strategy': 'reversal',
                'entry': 'confirmation',
                'stop_loss': 'moderate',
                'take_profit': 'fixed',
                'timeframes': ['current', 'higher'],
                'indicators': ['divergence', 'candlestick', 'support_resistance'],
                'confidence': 0.5
            }
            
        else:  # choppy or unknown
            return {
                'strategy': 'selective',
                'entry': 'avoid',
                'stop_loss': 'tight',
                'take_profit': 'quick',
                'timeframes': ['higher'],
                'indicators': ['volatility', 'atr'],
                'confidence': 0.2
            }
            
    def get_regime_history(self) -> List[Dict[str, Any]]:
        """
        Get history of regime detections
        
        Returns:
            List of past regime detections
        """
        return self.regime_history
        
    def get_regime_distribution(self) -> Dict[str, float]:
        """
        Get distribution of regimes over history
        
        Returns:
            Dict with percentage of time spent in each regime
        """
        if not self.regime_history:
            return {}
            
        regimes = [r['regime'] for r in self.regime_history]
        total = len(regimes)
        
        distribution = {}
        for regime in set(regimes):
            count = regimes.count(regime)
            distribution[regime] = count / total
            
        return distribution
        
    def _detect_trend_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect trend direction and strength
        
        Args:
            df: DataFrame with OHLC and indicators
            
        Returns:
            Dict with trend direction and strength
        """
        try:
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine trend direction based on moving averages
            ma_alignment = 0
            
            # Check price vs SMA relationship
            if latest['close'] > latest['sma20']:
                ma_alignment += 1
            elif latest['close'] < latest['sma20']:
                ma_alignment -= 1
                
            # Check SMA relationship
            if latest['sma20'] > latest['sma50']:
                ma_alignment += 1
            elif latest['sma20'] < latest['sma50']:
                ma_alignment -= 1
                
            # Determine direction (-1, 0, 1)
            if ma_alignment >= 1:
                trend_direction = 1  # Bullish
            elif ma_alignment <= -1:
                trend_direction = -1  # Bearish
            else:
                trend_direction = 0  # Neutral
                
            # Calculate trend strength based on slope of moving averages
            ma_slope = 0
            
            # Calculate slope of SMA20
            sma20_current = latest['sma20']
            sma20_past = df['sma20'].iloc[-10] if len(df) >= 10 else df['sma20'].iloc[0]
            
            if not pd.isna(sma20_current) and not pd.isna(sma20_past) and sma20_past != 0:
                sma20_change = (sma20_current - sma20_past) / sma20_past
                ma_slope += sma20_change * 100  # Convert to percentage
                
            # Normalize slope to -1 to 1 range
            max_slope = 2.0  # 2% is considered a strong trend
            normalized_slope = np.clip(ma_slope / max_slope, -1, 1)
            
            # Combine direction and strength
            trend_strength = normalized_slope
            
            # Additional factor: consistency of price movement
            # Count consecutive candles in trend direction
            consecutive_up = 0
            consecutive_down = 0
            
            for i in range(min(10, len(df) - 1)):
                idx = len(df) - 1 - i
                if idx > 0:
                    if df['close'].iloc[idx] > df['close'].iloc[idx-1]:
                        consecutive_up += 1
                        consecutive_down = 0
                    elif df['close'].iloc[idx] < df['close'].iloc[idx-1]:
                        consecutive_down += 1
                        consecutive_up = 0
                    else:
                        # No change
                        pass
                        
            consistency_factor = max(consecutive_up, consecutive_down) / 10
            if (consecutive_up > consecutive_down and trend_direction < 0) or \
               (consecutive_down > consecutive_up and trend_direction > 0):
                # Inconsistency between recent price action and overall trend
                consistency_factor *= -0.5
                
            # Adjust trend strength with consistency
            trend_strength = 0.7 * trend_strength + 0.3 * consistency_factor
            
            # Final scaling to ensure -1 to 1 range
            trend_strength = np.clip(trend_strength, -1, 1)
            
            return {
                'direction': trend_direction,
                'strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting trend strength: {e}")
            return {'direction': 0, 'strength': 0}
            
    def _detect_volatility(self, df: pd.DataFrame) -> str:
        """
        Detect if volatility is high, normal, or low
        
        Args:
            df: DataFrame with OHLC and indicators
            
        Returns:
            str: 'high', 'normal', or 'low'
        """
        try:
            # Use ATR as volatility measure
            if 'atr14' in df.columns and not df['atr14'].isna().all():
                current_atr = df['atr14'].iloc[-1]
                avg_atr = df['atr14'].iloc[-self.volatility_lookback:].mean()
                
                if avg_atr > 0:
                    volatility_ratio = current_atr / avg_atr
                    
                    if volatility_ratio > 1.5:
                        return 'high'
                    elif volatility_ratio < 0.7:
                        return 'low'
                
            # Alternative: use price range
            recent_ranges = []
            for i in range(min(self.volatility_lookback, len(df))):
                idx = len(df) - 1 - i
                if idx >= 0:
                    candle_range = (df['high'].iloc[idx] - df['low'].iloc[idx]) / df['close'].iloc[idx]
                    recent_ranges.append(candle_range)
                    
            if recent_ranges:
                current_range = recent_ranges[0]
                avg_range = np.mean(recent_ranges)
                
                if avg_range > 0:
                    range_ratio = current_range / avg_range
                    
                    if range_ratio > 1.5:
                        return 'high'
                    elif range_ratio < 0.7:
                        return 'low'
                        
            return 'normal'
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility: {e}")
            return 'normal'
            
    def _detect_range_bound(self, df: pd.DataFrame) -> bool:
        """
        Detect if the market is range-bound
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            bool: True if market is range-bound
        """
        try:
            # Method 1: Use ADFuller test for stationarity if available
            if adfuller is not None:
                price_series = df['close'].iloc[-self.range_lookback:]
                if len(price_series) >= 8:  # Minimum length for reliable test
                    adf_result = adfuller(price_series)
                    if adf_result[1] < 0.05:  # p-value less than 0.05 suggests stationarity
                        return True
            
            # Method 2: Check if price is contained within a narrow range
            recent_prices = df['close'].iloc[-self.range_lookback:]
            price_range = recent_prices.max() - recent_prices.min()
            avg_price = recent_prices.mean()
            
            # Consider range-bound if range is less than 2% of average price
            if avg_price > 0 and price_range / avg_price < 0.02:
                return True
                
            # Method 3: Check if moving averages are flat
            if 'sma20' in df.columns and not df['sma20'].isna().all():
                recent_sma = df['sma20'].iloc[-self.range_lookback:]
                sma_range = recent_sma.max() - recent_sma.min()
                avg_sma = recent_sma.mean()
                
                if avg_sma > 0 and sma_range / avg_sma < 0.005:  # Very flat SMA
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting range-bound market: {e}")
            return False
            
    def _detect_momentum(self, df: pd.DataFrame) -> float:
        """
        Detect momentum strength (-1 to 1)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            float: Momentum strength from -1 (strong negative) to 1 (strong positive)
        """
        try:
            # Calculate rate of change
            if len(df) >= 10:
                current_price = df['close'].iloc[-1]
                past_price = df['close'].iloc[-10]
                
                if past_price > 0:
                    roc = (current_price - past_price) / past_price
                    
                    # Normalize to -1 to 1 range (5% change is considered strong)
                    normalized_momentum = np.clip(roc * 20, -1, 1)
                    return normalized_momentum
                    
            # Alternative: use RSI if available
            if 'rsi14' in df.columns and not df['rsi14'].isna().all():
                rsi = df['rsi14'].iloc[-1]
                
                # Convert RSI (0-100) to momentum (-1 to 1)
                normalized_momentum = (rsi - 50) / 50
                return normalized_momentum
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum: {e}")
            return 0.0
            
    def _detect_breakout(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect if price is breaking out of a range
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with breakout information
        """
        try:
            # Calculate recent range
            lookback = 20
            
            if len(df) < lookback + 1:
                return {'is_breakout_up': False, 'is_breakout_down': False}
                
            # Recent high and low (excluding the last candle)
            recent_high = df['high'].iloc[-(lookback+1):-1].max()
            recent_low = df['low'].iloc[-(lookback+1):-1].min()
            
            # Current close
            current_close = df['close'].iloc[-1]
            
            # Check for breakouts with a buffer (0.1% to avoid false breakouts)
            buffer = recent_high * 0.001
            
            is_breakout_up = current_close > (recent_high + buffer)
            is_breakout_down = current_close < (recent_low - buffer)
            
            # Add volume confirmation if available
            if 'volume' in df.columns and not df['volume'].isna().all():
                avg_volume = df['volume'].iloc[-lookback:-1].mean()
                current_volume = df['volume'].iloc[-1]
                
                # Strong breakouts should have above-average volume
                if current_volume <= avg_volume:
                    # Weaken breakout signals if volume is not confirming
                    is_breakout_up = is_breakout_up and current_close > recent_high * 1.005  # Need stronger signal
                    is_breakout_down = is_breakout_down and current_close < recent_low * 0.995  # Need stronger signal
                    
            return {
                'is_breakout_up': is_breakout_up,
                'is_breakout_down': is_breakout_down
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout: {e}")
            return {'is_breakout_up': False, 'is_breakout_down': False}
            
    def _detect_reversal(self, df: pd.DataFrame) -> bool:
        """
        Detect potential reversal patterns
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            bool: True if reversal pattern detected
        """
        try:
            if len(df) < 10:
                return False
                
            # Get recent trend
            trend_lookback = 20
            short_lookback = 3
            
            if len(df) < trend_lookback:
                return False
                
            # Determine recent trend
            trend_start_price = df['close'].iloc[-trend_lookback]
            trend_end_price = df['close'].iloc[-short_lookback-1]
            
            recent_trend = 1 if trend_end_price > trend_start_price else -1
            
            # Check for potential reversal
            short_term_move = df['close'].iloc[-1] - df['close'].iloc[-short_lookback]
            
            # Reversal is when short term move is against the recent trend
            potential_reversal = (recent_trend > 0 and short_term_move < 0) or \
                                (recent_trend < 0 and short_term_move > 0)
                                
            # Check for momentum divergence if available
            divergence = False
            
            if 'rsi14' in df.columns and not df['rsi14'].isna().all():
                rsi_trend_start = df['rsi14'].iloc[-trend_lookback]
                rsi_trend_end = df['rsi14'].iloc[-short_lookback-1]
                rsi_current = df['rsi14'].iloc[-1]
                
                # Bearish divergence: price making higher high, RSI making lower high
                if recent_trend > 0 and trend_end_price > trend_start_price and rsi_trend_end < rsi_trend_start:
                    divergence = True
                    
                # Bullish divergence: price making lower low, RSI making higher low
                if recent_trend < 0 and trend_end_price < trend_start_price and rsi_trend_end > rsi_trend_start:
                    divergence = True
                    
            # Strong reversal signal if both price action and divergence align
            return potential_reversal and divergence
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal: {e}")
            return False
            
    def _determine_regime(self, trend_strength, is_range_bound, is_breakout_up, is_breakout_down, is_reversal):
        """
        Determine the overall market regime
        
        Args:
            trend_strength: Strength of the trend (-1 to 1)
            is_range_bound: Whether market is range-bound
            is_breakout_up: Whether price is breaking out upward
            is_breakout_down: Whether price is breaking out downward
            is_reversal: Whether a reversal pattern is detected
            
        Returns:
            str: Market regime
        """
        # Prioritize different signals
        if is_breakout_up:
            return 'breakout_up'
        elif is_breakout_down:
            return 'breakout_down'
        elif is_reversal:
            return 'reversal'
        elif is_range_bound:
            return 'ranging'
        elif trend_strength > 0.3:
            return 'trending_up'
        elif trend_strength < -0.3:
            return 'trending_down'
        else:
            return 'choppy'
            
    def _get_default_regime(self):
        """
        Return default regime when detection fails
        
        Returns:
            Dict with default regime values
        """
        return {
            'trend_strength': 0,
            'trend_direction': 0,
            'volatility': 'normal',
            'range_bound': False,
            'momentum': 0,
            'regime': 'unknown',
            'atr': 0,
            'is_breakout': False,
            'is_reversal': False,
            'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series(np.nan, index=df.index)