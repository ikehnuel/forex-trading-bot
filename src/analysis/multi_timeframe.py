# src/analysis/multi_timeframe.py
import pandas as pd
import numpy as np
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TIMEFRAME_MAP, TIMEFRAME_RELATIONSHIPS, logger
from mt5_connector.data_collector import DataCollector

class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes to find confluent signals
    and identify high-probability trading opportunities.
    """
    def __init__(self, data_collector=None):
        self.logger = logging.getLogger(__name__)
        
        # Use provided data collector or create a new one
        if data_collector is None:
            try:
                self.data_collector = DataCollector()
            except Exception as e:
                self.logger.error(f"Could not initialize data collector: {e}")
                self.data_collector = None
        else:
            self.data_collector = data_collector
            
        # Store timeframe relationships
        self.timeframe_relationships = TIMEFRAME_RELATIONSHIPS
        
    def get_data_for_timeframes(self, symbol: str, base_timeframe: str, num_candles: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes based on the base timeframe
        
        Args:
            symbol: Trading pair symbol
            base_timeframe: Base timeframe for analysis
            num_candles: Number of candles to retrieve for each timeframe
            
        Returns:
            Dictionary of dataframes for each relevant timeframe
        """
        try:
            if self.data_collector is None:
                self.logger.error("Data collector not available")
                return {}
                
            # Get relevant timeframes to analyze
            if base_timeframe not in self.timeframe_relationships:
                self.logger.warning(f"Base timeframe {base_timeframe} not found in relationships, using default")
                base_timeframe = 'H1'  # Default to H1 if not found
                
            relevant_tfs = self.timeframe_relationships[base_timeframe].copy()
            if base_timeframe not in relevant_tfs:
                relevant_tfs.append(base_timeframe)  # Add base timeframe if not already included
                
            data = {}
            
            for tf in relevant_tfs:
                # Get more candles for lower timeframes
                tf_candles = num_candles
                if self._is_lower_timeframe(tf, base_timeframe):
                    tf_candles = num_candles * 3  # More data for lower timeframes
                    
                # Get data from MT5 via data collector
                df = self.data_collector.get_ohlc_data(symbol, tf, tf_candles)
                
                if df is not None and not df.empty:
                    data[tf] = df
                else:
                    self.logger.warning(f"Could not get data for {symbol} on {tf} timeframe")
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data: {e}")
            return {}
            
    def analyze_timeframes(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze data across multiple timeframes to find confluent signals
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not timeframe_data:
                self.logger.warning("No timeframe data provided for analysis")
                return self._get_default_analysis()
                
            results = {}
            
            # 1. Trend analysis across timeframes
            trend_info = self._analyze_trend_across_timeframes(timeframe_data)
            results['trend_analysis'] = trend_info
            
            # 2. Support and resistance detection
            support_resistance = self._find_support_resistance(timeframe_data)
            results['support_resistance'] = support_resistance
            
            # 3. Entry point detection
            entry_points = self._detect_entry_points(timeframe_data, trend_info['overall_trend'])
            results['entry_points'] = entry_points
            
            # 4. Confluence analysis
            confluence = self._analyze_confluence(timeframe_data, support_resistance, trend_info['overall_trend'])
            results['confluence'] = confluence
            
            # 5. Calculate timeframe alignment score
            alignment_score = self._calculate_alignment_score(timeframe_data)
            results['alignment_score'] = alignment_score
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframes: {e}")
            return self._get_default_analysis()
            
    def generate_trading_signal(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal based on multi-timeframe analysis
        
        Args:
            analysis_results: Results from analyze_timeframes()
            
        Returns:
            Trading signal dictionary
        """
        try:
            if not analysis_results or 'trend_analysis' not in analysis_results:
                self.logger.warning("Invalid analysis results for signal generation")
                return self._get_default_signal()
                
            overall_trend = analysis_results['trend_analysis']['overall_trend']
            trend_strength = analysis_results['trend_analysis']['trend_strength']
            confluence = analysis_results['confluence']
            entry_points = analysis_results['entry_points']
            alignment_score = analysis_results['alignment_score']
            
            # Default signal - no trade
            signal = {
                'action': 'HOLD',
                'confidence': 0,
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'timeframes_aligned': alignment_score,
                'trend_strength': trend_strength
            }
            
            # Minimum thresholds for trading
            min_trend_strength = 0.4
            min_confluence_score = 0.5
            min_alignment_score = 0.5
            
            # Check if we have a potential trade
            valid_entry = entry_points.get('valid', False)
            confluence_direction = confluence.get('direction', 'neutral')
            confluence_score = confluence.get('score', 0)
            
            if valid_entry and abs(trend_strength) >= min_trend_strength and \
               confluence_score >= min_confluence_score and alignment_score >= min_alignment_score:
                
                # Determine direction
                if trend_strength > 0 and confluence_direction == 'buy':
                    signal['action'] = 'BUY'
                    signal['confidence'] = (abs(trend_strength) + confluence_score + alignment_score) / 3
                    signal['entry'] = entry_points.get('entry_price')
                    signal['stop_loss'] = entry_points.get('stop_loss')
                    signal['take_profit'] = entry_points.get('take_profit')
                    
                elif trend_strength < 0 and confluence_direction == 'sell':
                    signal['action'] = 'SELL'
                    signal['confidence'] = (abs(trend_strength) + confluence_score + alignment_score) / 3
                    signal['entry'] = entry_points.get('entry_price')
                    signal['stop_loss'] = entry_points.get('stop_loss')
                    signal['take_profit'] = entry_points.get('take_profit')
                    
            # Calculate risk-reward ratio
            if signal['action'] in ['BUY', 'SELL'] and signal['entry'] is not None and \
               signal['stop_loss'] is not None and signal['take_profit'] is not None:
                
                if signal['action'] == 'BUY':
                    risk = signal['entry'] - signal['stop_loss']
                    reward = signal['take_profit'] - signal['entry']
                else:
                    risk = signal['stop_loss'] - signal['entry']
                    reward = signal['entry'] - signal['take_profit']
                    
                if risk > 0:
                    signal['risk_reward_ratio'] = reward / risk
                else:
                    signal['risk_reward_ratio'] = 0
                    
            # Add reasoning
            signal['reasoning'] = self._generate_reasoning(analysis_results, signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return self._get_default_signal()
            
    def _analyze_trend_across_timeframes(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze trend direction and strength across all timeframes
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            
        Returns:
            Trend analysis results
        """
        try:
            trend_scores = []
            timeframe_trends = {}
            
            # Analyze each timeframe
            for tf, df in timeframe_data.items():
                if len(df) < 20:
                    continue
                    
                # Get latest data
                latest_idx = len(df) - 1
                
                # Calculate trend for this timeframe
                tf_trend = self._calculate_timeframe_trend(df)
                timeframe_trends[tf] = tf_trend
                
                # Apply weighting based on timeframe
                weight = self._get_timeframe_weight(tf)
                weighted_score = tf_trend['trend_direction'] * weight
                
                trend_scores.append({
                    'timeframe': tf,
                    'trend_direction': tf_trend['trend_direction'],
                    'trend_strength': tf_trend['trend_strength'],
                    'weight': weight,
                    'weighted_score': weighted_score
                })
                
            # Calculate overall trend
            overall_trend = 0
            total_weight = 0
            
            for score in trend_scores:
                overall_trend += score['weighted_score']
                total_weight += score['weight']
                
            if total_weight > 0:
                overall_trend = overall_trend / total_weight
            
            # Calculate average trend strength
            avg_trend_strength = np.mean([score['trend_strength'] for score in trend_scores]) if trend_scores else 0
            
            # Normalize to -1 to 1 range
            trend_strength = np.clip(overall_trend, -1, 1)
            
            return {
                'timeframe_trends': timeframe_trends,
                'trend_scores': trend_scores,
                'overall_trend': overall_trend,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends across timeframes: {e}")
            return {
                'timeframe_trends': {},
                'trend_scores': [],
                'overall_trend': 0,
                'trend_strength': 0
            }
            
    def _calculate_timeframe_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trend direction and strength for a single timeframe
        
        Args:
            df: DataFrame with OHLC and indicators
            
        Returns:
            Trend direction and strength
        """
        try:
            # Ensure we have all needed indicators
            if 'sma20' not in df.columns:
                df['sma20'] = df['close'].rolling(20).mean()
            if 'sma50' not in df.columns:
                df['sma50'] = df['close'].rolling(50).mean()
                
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine trend direction using moving averages
            trend_direction = 0
            
            if latest['close'] > latest['sma20'] and latest['sma20'] > latest['sma50']:
                trend_direction = 1  # Bullish
            elif latest['close'] < latest['sma20'] and latest['sma20'] < latest['sma50']:
                trend_direction = -1  # Bearish
                
            # Calculate trend strength using moving average divergence
            trend_strength = 0
            
            if 'sma20' in df.columns and 'sma50' in df.columns:
                ma_diff = abs(latest['sma20'] - latest['sma50'])
                avg_price = (latest['sma20'] + latest['sma50']) / 2
                
                # Normalize difference as percentage of average price
                if avg_price > 0:
                    trend_strength = min(1.0, ma_diff / avg_price * 20)  # Scale to max of 1.0
                    if trend_direction < 0:
                        trend_strength = -trend_strength
                        
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe trend: {e}")
            return {'trend_direction': 0, 'trend_strength': 0}
            
    def _find_support_resistance(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Find support and resistance levels across timeframes
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            
        Returns:
            Support and resistance levels
        """
        try:
            all_levels = []
            
            # Process timeframes from higher to lower
            sorted_timeframes = self._sort_timeframes(list(timeframe_data.keys()))
            
            for tf in sorted_timeframes:
                df = timeframe_data[tf]
                if len(df) < 20:
                    continue
                    
                # Find local swing highs and lows
                highs, lows = self._find_swings(df)
                
                # Calculate strength based on timeframe and number of touches
                weight = self._get_timeframe_weight(tf)
                
                # Add support levels
                for idx in lows:
                    if idx < len(df):
                        all_levels.append({
                            'type': 'support',
                            'level': df['low'].iloc[idx],
                            'timeframe': tf,
                            'strength': weight,
                            'time': df['time'].iloc[idx]
                        })
                        
                # Add resistance levels
                for idx in highs:
                    if idx < len(df):
                        all_levels.append({
                            'type': 'resistance',
                            'level': df['high'].iloc[idx],
                            'timeframe': tf,
                            'strength': weight,
                            'time': df['time'].iloc[idx]
                        })
                        
            # Cluster nearby levels
            clustered_levels = self._cluster_levels(all_levels)
            
            # Sort by price level
            clustered_levels.sort(key=lambda x: x['level'])
            
            # Find closest levels to current price
            current_price = None
            for tf, df in timeframe_data.items():
                if len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    break
                    
            if current_price is not None:
                closest_support = None
                closest_resistance = None
                min_support_dist = float('inf')
                min_resistance_dist = float('inf')
                
                for level in clustered_levels:
                    if level['type'] == 'support' and current_price > level['level']:
                        dist = current_price - level['level']
                        if dist < min_support_dist:
                            min_support_dist = dist
                            closest_support = level
                    elif level['type'] == 'resistance' and current_price < level['level']:
                        dist = level['level'] - current_price
                        if dist < min_resistance_dist:
                            min_resistance_dist = dist
                            closest_resistance = level
                            
                return {
                    'levels': all_levels,
                    'clustered_levels': clustered_levels,
                    'closest_support': closest_support,
                    'closest_resistance': closest_resistance,
                    'current_price': current_price
                }
            else:
                return {
                    'levels': all_levels,
                    'clustered_levels': clustered_levels,
                    'closest_support': None,
                    'closest_resistance': None,
                    'current_price': None
                }
                
        except Exception as e:
            self.logger.error(f"Error finding support and resistance: {e}")
            return {
                'levels': [],
                'clustered_levels': [],
                'closest_support': None,
                'closest_resistance': None,
                'current_price': None
            }
            
    def _find_swings(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows in price data
        
        Args:
            df: DataFrame with OHLC data
            window: Window size for swing detection
            
        Returns:
            Tuple of (swing high indices, swing low indices)
        """
        try:
            highs = []
            lows = []
            
            # Need at least 2*window+1 candles
            if len(df) < 2 * window + 1:
                return highs, lows
                
            # Find swing highs
            for i in range(window, len(df) - window):
                if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                    highs.append(i)
                    
            # Find swing lows
            for i in range(window, len(df) - window):
                if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                    lows.append(i)
                    
            return highs, lows
            
        except Exception as e:
            self.logger.error(f"Error finding swings: {e}")
            return [], []
            
    def _cluster_levels(self, levels: List[Dict[str, Any]], tolerance: float = 0.0005) -> List[Dict[str, Any]]:
        """
        Group nearby support/resistance levels
        
        Args:
            levels: List of support/resistance levels
            tolerance: Price distance tolerance for clustering (as fraction)
            
        Returns:
            List of clustered levels
        """
        try:
            if not levels:
                return []
                
            # Sort by price level
            sorted_levels = sorted(levels, key=lambda x: x['level'])
            
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]['level']
                prev_level = current_cluster[-1]['level']
                
                # Calculate relative distance
                relative_dist = abs(current_level - prev_level) / prev_level
                
                # If levels are close, add to current cluster
                if relative_dist < tolerance:
                    current_cluster.append(sorted_levels[i])
                else:
                    # Process the completed cluster
                    level_sum = sum(level['level'] for level in current_cluster)
                    avg_level = level_sum / len(current_cluster)
                    
                    # Sum up strength from all levels
                    strength_sum = sum(level['strength'] for level in current_cluster)
                    
                    # Determine majority type (support or resistance)
                    support_count = sum(1 for level in current_cluster if level['type'] == 'support')
                    resistance_count = len(current_cluster) - support_count
                    level_type = 'support' if support_count >= resistance_count else 'resistance'
                    
                    # Create clustered level
                    clusters.append({
                        'level': avg_level,
                        'strength': strength_sum,
                        'type': level_type,
                        'count': len(current_cluster),
                        'timeframes': list(set(level['timeframe'] for level in current_cluster))
                    })
                    
                    # Start new cluster
                    current_cluster = [sorted_levels[i]]
                    
            # Process the last cluster
            if current_cluster:
                level_sum = sum(level['level'] for level in current_cluster)
                avg_level = level_sum / len(current_cluster)
                
                strength_sum = sum(level['strength'] for level in current_cluster)
                
                support_count = sum(1 for level in current_cluster if level['type'] == 'support')
                resistance_count = len(current_cluster) - support_count
                level_type = 'support' if support_count >= resistance_count else 'resistance'
                
                clusters.append({
                    'level': avg_level,
                    'strength': strength_sum,
                    'type': level_type,
                    'count': len(current_cluster),
                    'timeframes': list(set(level['timeframe'] for level in current_cluster))
                })
                
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering levels: {e}")
            return []
            
    def _detect_entry_points(self, timeframe_data: Dict[str, pd.DataFrame], overall_trend: float) -> Dict[str, Any]:
        """
        Detect potential entry points based on multiple timeframes
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            overall_trend: Overall trend direction from trend analysis
            
        Returns:
            Dictionary with entry point information
        """
        try:
            # Use base timeframe for entry detection
            base_tf = self._get_base_timeframe(timeframe_data)
            
            if base_tf is None or base_tf not in timeframe_data:
                return {'valid': False}
                
            df = timeframe_data[base_tf]
            
            if len(df) < 20:
                return {'valid': False}
                
            # Get current price
            latest_idx = len(df) - 1
            current_close = df['close'].iloc[latest_idx]
            current_high = df['high'].iloc[latest_idx]
            current_low = df['low'].iloc[latest_idx]
            
            # Default entry point
            entry_point = {
                'valid': False,
                'entry_price': current_close,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': 0
            }
            
            # Get ATR for stop loss calculation
            atr = 0
            if 'atr14' in df.columns:
                atr = df['atr14'].iloc[latest_idx]
            else:
                # Calculate ATR manually if not available
                true_ranges = []
                for i in range(1, min(15, len(df))):
                    idx = latest_idx - i
                    if idx >= 0:
                        high_low = df['high'].iloc[idx] - df['low'].iloc[idx]
                        high_close = abs(df['high'].iloc[idx] - df['close'].iloc[idx-1])
                        low_close = abs(df['low'].iloc[idx] - df['close'].iloc[idx-1])
                        true_ranges.append(max(high_low, high_close, low_close))
                if true_ranges:
                    atr = sum(true_ranges) / len(true_ranges)
                    
            # Different entry strategies based on trend
            if overall_trend > 0.3:  # Uptrend
                # Check for pullback to support or moving average
                entry_conditions_met = False
                
                # 1. Pullback to moving average
                if 'sma20' in df.columns:
                    ma_value = df['sma20'].iloc[latest_idx]
                    
                    # Price pulled back to MA
                    if current_low <= ma_value <= current_high:
                        entry_conditions_met = True
                        
                # 2. Bullish engulfing pattern
                if latest_idx > 0:
                    prev_open = df['open'].iloc[latest_idx-1]
                    prev_close = df['close'].iloc[latest_idx-1]
                    curr_open = df['open'].iloc[latest_idx]
                    curr_close = df['close'].iloc[latest_idx]
                    
                    if prev_close < prev_open and curr_close > curr_open and \
                       curr_close > prev_open and curr_open < prev_close:
                        entry_conditions_met = True
                        
                if entry_conditions_met:
                    # Calculate entry above current bar's high
                    entry_price = current_high + atr * 0.1
                    
                    # Stop loss below recent swing low or below MA
                    recent_low = df['low'].iloc[max(0, latest_idx-5):latest_idx+1].min()
                    stop_loss = min(recent_low - atr * 0.5, df['sma20'].iloc[latest_idx] - atr * 0.2) \
                               if 'sma20' in df.columns else recent_low - atr * 0.5
                    
                    # Take profit based on ATR or next resistance
                    take_profit = entry_price + (entry_price - stop_loss) * 1.5
                    
                    entry_point['valid'] = True
                    entry_point['entry_price'] = entry_price
                    entry_point['stop_loss'] = stop_loss
                    entry_point['take_profit'] = take_profit
                    entry_point['risk_reward'] = (take_profit - entry_price) / (entry_price - stop_loss) \
                                               if (entry_price - stop_loss) > 0 else 0
                    
            elif overall_trend < -0.3:  # Downtrend
                # Check for rallies to resistance or moving average
                entry_conditions_met = False
                
                # 1. Rally to moving average
                if 'sma20' in df.columns:
                    ma_value = df['sma20'].iloc[latest_idx]
                    
                    # Price rallied to MA
                    if current_low <= ma_value <= current_high:
                        entry_conditions_met = True
                        
                # 2. Bearish engulfing pattern
                if latest_idx > 0:
                    prev_open = df['open'].iloc[latest_idx-1]
                    prev_close = df['close'].iloc[latest_idx-1]
                    curr_open = df['open'].iloc[latest_idx]
                    curr_close = df['close'].iloc[latest_idx]
                    
                    if prev_close > prev_open and curr_close < curr_open and \
                       curr_close < prev_open and curr_open > prev_close:
                        entry_conditions_met = True
                        
                if entry_conditions_met:
                    # Calculate entry below current bar's low
                    entry_price = current_low - atr * 0.1
                    
                    # Stop loss above recent swing high or above MA
                    recent_high = df['high'].iloc[max(0, latest_idx-5):latest_idx+1].max()
                    stop_loss = max(recent_high + atr * 0.5, df['sma20'].iloc[latest_idx] + atr * 0.2) \
                               if 'sma20' in df.columns else recent_high + atr * 0.5
                    
                    # Take profit based on ATR or next support
                    take_profit = entry_price - (stop_loss - entry_price) * 1.5
                    
                    entry_point['valid'] = True
                    entry_point['entry_price'] = entry_price
                    entry_point['stop_loss'] = stop_loss
                    entry_point['take_profit'] = take_profit
                    entry_point['risk_reward'] = (entry_price - take_profit) / (stop_loss - entry_price) \
                                               if (stop_loss - entry_price) > 0 else 0
                    
            return entry_point
            
        except Exception as e:
            self.logger.error(f"Error detecting entry points: {e}")
            return {'valid': False}
            
    def _analyze_confluence(self, 
                          timeframe_data: Dict[str, pd.DataFrame], 
                          support_resistance: Dict[str, Any], 
                          overall_trend: float) -> Dict[str, Any]:
        """
        Analyze confluence of signals across timeframes
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            support_resistance: Support and resistance levels
            overall_trend: Overall trend direction
            
        Returns:
            Dictionary with confluence analysis
        """
        try:
            # Count how many timeframes show the same signal
            buy_signals = 0
            sell_signals = 0
            neutral_signals = 0
            total_weight = 0
            
            signals_by_timeframe = {}
            
            for tf, df in timeframe_data.items():
                if len(df) < 20:
                    continue
                    
                # Get latest values
                latest = df.iloc[-1]
                
                # Weight by timeframe
                weight = self._get_timeframe_weight(tf)
                total_weight += weight
                
                # Trend signal
                trend_signal = 0
                if 'sma20' in df.columns and 'sma50' in df.columns:
                    if latest['close'] > latest['sma20'] and latest['sma20'] > latest['sma50']:
                        trend_signal = 1
                    elif latest['close'] < latest['sma20'] and latest['sma20'] < latest['sma50']:
                        trend_signal = -1
                        
                # RSI signal
                rsi_signal = 0
                if 'rsi14' in df.columns:
                    if latest['rsi14'] < 30:
                        rsi_signal = 1  # Oversold - buy signal
                    elif latest['rsi14'] > 70:
                        rsi_signal = -1  # Overbought - sell signal
                        
                # MACD signal
                macd_signal = 0
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    if latest['macd'] > latest['macd_signal']:
                        macd_signal = 1
                    elif latest['macd'] < latest['macd_signal']:
                        macd_signal = -1
                        
                # Combined signal with weighting
                combined = 0
                signal_count = 0
                
                if trend_signal != 0:
                    combined += trend_signal * 0.5
                    signal_count += 1
                    
                if rsi_signal != 0:
                    combined += rsi_signal * 0.2
                    signal_count += 1
                    
                if macd_signal != 0:
                    combined += macd_signal * 0.3
                    signal_count += 1
                    
                # Normalize if we have signals
                if signal_count > 0:
                    combined = combined / signal_count
                    
                # Store signal for this timeframe
                signals_by_timeframe[tf] = {
                    'trend': trend_signal,
                    'rsi': rsi_signal,
                    'macd': macd_signal,
                    'combined': combined,
                    'weight': weight
                }
                
                # Add to totals
                if combined > 0.3:
                    buy_signals += weight
                elif combined < -0.3:
                    sell_signals += weight
                else:
                    neutral_signals += weight
                    
            # Determine overall direction and confidence
            direction = 'neutral'
            confidence = 0
            
            if buy_signals > sell_signals:
                direction = 'buy'
                confidence = buy_signals / total_weight if total_weight > 0 else 0
            elif sell_signals > buy_signals:
                direction = 'sell'
                confidence = sell_signals / total_weight if total_weight > 0 else 0
                
            # Adjust confidence based on alignment with overall trend
            aligned_with_trend = 0
            if (overall_trend > 0 and direction == 'buy') or \
               (overall_trend < 0 and direction == 'sell'):
                aligned_with_trend = max(buy_signals, sell_signals)
                confidence *= 1.2  # Boost confidence when aligned with trend
            else:
                confidence *= 0.8  # Reduce confidence when against trend
                
            # Cap confidence at 1.0
            confidence = min(1.0, confidence)
            
            # Check for confluence with support/resistance
            sr_confluence = 0
            
            if support_resistance['current_price'] is not None:
                current_price = support_resistance['current_price']
                
                if direction == 'buy' and support_resistance['closest_support'] is not None:
                    # Check if price is near support
                    support_level = support_resistance['closest_support']['level']
                    rel_distance = (current_price - support_level) / current_price
                    
                    # Price within 0.3% of support is strong confluence
                    if rel_distance < 0.003:
                        sr_confluence = 1.0
                    elif rel_distance < 0.01:
                        sr_confluence = 0.5
                        
                elif direction == 'sell' and support_resistance['closest_resistance'] is not None:
                    # Check if price is near resistance
                    resistance_level = support_resistance['closest_resistance']['level']
                    rel_distance = (resistance_level - current_price) / current_price
                    
                    # Price within 0.3% of resistance is strong confluence
                    if rel_distance < 0.003:
                        sr_confluence = 1.0
                    elif rel_distance < 0.01:
                        sr_confluence = 0.5
                        
            # Add S/R confluence to confidence
            confidence = 0.8 * confidence + 0.2 * sr_confluence
            
            return {
                'direction': direction,
                'score': confidence,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'neutral_signals': neutral_signals,
                'signals_by_timeframe': signals_by_timeframe,
                'sr_confluence': sr_confluence,
                'timeframes_aligned': aligned_with_trend / total_weight if total_weight > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing confluence: {e}")
            return {
                'direction': 'neutral',
                'score': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'neutral_signals': 0,
                'signals_by_timeframe': {},
                'sr_confluence': 0,
                'timeframes_aligned': 0
            }
            
    def _calculate_alignment_score(self, timeframe_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate alignment score across timeframes
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            
        Returns:
            Alignment score from 0 to 1
        """
        try:
            if len(timeframe_data) < 2:
                return 0
                
            # Count bullish and bearish timeframes
            bullish_tfs = 0
            bearish_tfs = 0
            total_weight = 0
            
            for tf, df in timeframe_data.items():
                if len(df) < 20:
                    continue
                    
                # Weight by timeframe
                weight = self._get_timeframe_weight(tf)
                total_weight += weight
                
                # Check if trend is bullish or bearish
                latest = df.iloc[-1]
                
                if 'sma20' in df.columns and 'sma50' in df.columns:
                    if latest['close'] > latest['sma20'] and latest['sma20'] > latest['sma50']:
                        bullish_tfs += weight
                    elif latest['close'] < latest['sma20'] and latest['sma20'] < latest['sma50']:
                        bearish_tfs += weight
                        
            # Calculate alignment score
            if total_weight > 0:
                max_aligned = max(bullish_tfs, bearish_tfs)
                alignment_score = max_aligned / total_weight
            else:
                alignment_score = 0
                
            return alignment_score
            
        except Exception as e:
            self.logger.error(f"Error calculating alignment score: {e}")
            return 0
        
    def _get_base_timeframe(self, timeframe_data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Get the base timeframe from the data
        
        Args:
            timeframe_data: Dictionary of dataframes for each timeframe
            
        Returns:
            Base timeframe or None
        """
        if not timeframe_data:
            return None
            
        # Sort timeframes by priority
        sorted_tfs = self._sort_timeframes(list(timeframe_data.keys()))
        
        # Return middle timeframe if available
        if len(sorted_tfs) >= 3:
            return sorted_tfs[len(sorted_tfs) // 2]
        elif sorted_tfs:
            return sorted_tfs[0]
        else:
            return None
            
    def _sort_timeframes(self, timeframes: List[str]) -> List[str]:
        """
        Sort timeframes from higher to lower
        
        Args:
            timeframes: List of timeframe strings
            
        Returns:
            Sorted list of timeframes
        """
        # Define timeframe priorities (higher number = higher timeframe)
        priorities = {
            'MN1': 7,
            'W1': 6,
            'D1': 5,
            'H4': 4,
            'H1': 3,
            'M30': 2,
            'M15': 1,
            'M5': 0,
            'M1': -1
        }
        
        # Sort based on priority
        return sorted(timeframes, key=lambda tf: priorities.get(tf, 0), reverse=True)
        
    def _get_timeframe_weight(self, timeframe: str) -> float:
        """
        Get weight for a timeframe
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Weight value
        """
        weights = {
            'MN1': 6.0,
            'W1': 5.0,
            'D1': 4.0,
            'H4': 3.0,
            'H1': 2.0,
            'M30': 1.5,
            'M15': 1.0,
            'M5': 0.5,
            'M1': 0.25
        }
        
        return weights.get(timeframe, 1.0)
        
    def _is_lower_timeframe(self, tf1: str, tf2: str) -> bool:
        """
        Check if tf1 is a lower timeframe than tf2
        
        Args:
            tf1: First timeframe
            tf2: Second timeframe
            
        Returns:
            True if tf1 is lower than tf2
        """
        priorities = {
            'MN1': 7,
            'W1': 6,
            'D1': 5,
            'H4': 4,
            'H1': 3,
            'M30': 2,
            'M15': 1,
            'M5': 0,
            'M1': -1
        }
        
        return priorities.get(tf1, 0) < priorities.get(tf2, 0)
        
    def _generate_reasoning(self, analysis_results: Dict[str, Any], signal: Dict[str, Any]) -> str:
        """
        Generate human-readable reasoning for the signal
        
        Args:
            analysis_results: Results from analyze_timeframes
            signal: Generated trading signal
            
        Returns:
            Reasoning string
        """
        if signal['action'] == 'HOLD':
            return "No clear setup detected across timeframes"
            
        # Get key metrics
        trend = "uptrend" if analysis_results['trend_analysis']['overall_trend'] > 0 else "downtrend"
        aligned = f"{int(signal['timeframes_aligned'] * 100)}% of timeframes aligned"
        
        reasoning = f"Signal based on {trend} detected across multiple timeframes with {aligned}. "
        
        if signal['action'] == 'BUY':
            reasoning += "Buy setup confirmed with favorable risk-reward ratio."
        else:
            reasoning += "Sell setup confirmed with favorable risk-reward ratio."
            
        return reasoning
        
    def _get_default_analysis(self) -> Dict[str, Any]:
        """
        Get default analysis results when analysis fails
        
        Returns:
            Default analysis dictionary
        """
        return {
            'trend_analysis': {
                'timeframe_trends': {},
                'trend_scores': [],
                'overall_trend': 0,
                'trend_strength': 0
            },
            'support_resistance': {
                'levels': [],
                'clustered_levels': [],
                'closest_support': None,
                'closest_resistance': None,
                'current_price': None
            },
            'entry_points': {
                'valid': False
            },
            'confluence': {
                'direction': 'neutral',
                'score': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'neutral_signals': 0,
                'signals_by_timeframe': {},
                'sr_confluence': 0,
                'timeframes_aligned': 0
            },
            'alignment_score': 0
        }
        
    def _get_default_signal(self) -> Dict[str, Any]:
        """
        Get default trading signal when generation fails
        
        Returns:
            Default signal dictionary
        """
        return {
            'action': 'HOLD',
            'confidence': 0,
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'timeframes_aligned': 0,
            'trend_strength': 0,
            'reasoning': "Insufficient data for analysis"
        }