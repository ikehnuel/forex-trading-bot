import pandas as pd
import numpy as np
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_RISK_PER_TRADE, MAX_DAILY_RISK, MAX_DRAWDOWN_THRESHOLD, logger

class RiskManager:
    """
    Manages risk assessment, position sizing, and risk controls
    for forex trading operations.
    """
    def __init__(self, 
                max_risk_per_trade: float = MAX_RISK_PER_TRADE,
                max_daily_risk: float = MAX_DAILY_RISK,
                max_drawdown_threshold: float = MAX_DRAWDOWN_THRESHOLD,
                position_sizing_method: str = "fixed"):
        """
        Initialize the risk management system.
        
        Args:
            max_risk_per_trade: Maximum account percentage to risk per trade (default 2%)
            max_daily_risk: Maximum account percentage to risk per day (default 5%)
            max_drawdown_threshold: Maximum drawdown before reducing position sizes (default 15%)
            position_sizing_method: Method to calculate position size ("fixed", "atr", "volatility")
        """
        self.logger = logging.getLogger(__name__)
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown_threshold = max_drawdown_threshold
        self.position_sizing_method = position_sizing_method
        self.trade_history = []
        self.daily_risk_used = 0
        self.daily_reset_time = None
        self.current_drawdown = 0
        self.peak_balance = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
    def reset_daily_risk(self) -> None:
        """Reset daily risk tracker at the start of a new trading day"""
        current_time = datetime.now()
        if self.daily_reset_time is None or current_time.date() > self.daily_reset_time.date():
            self.daily_risk_used = 0
            self.daily_reset_time = current_time
            self.logger.info(f"Daily risk reset at {current_time}")
    
    def update_drawdown(self, current_balance: float) -> float:
        """
        Update current drawdown calculation based on peak balance
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Current drawdown as a percentage
        """
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
        return self.current_drawdown
        
    def can_place_trade(self, account_info: Dict[str, Any], symbol_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if a new trade can be placed based on risk parameters
        
        Args:
            account_info: Account information dict
            symbol_info: Symbol information dict
            
        Returns:
            Tuple of (can_trade, reason)
        """
        try:
            # Reset daily risk if needed
            self.reset_daily_risk()
            
            # Check if we've exceeded daily risk limit
            if self.daily_risk_used >= self.max_daily_risk:
                return False, f"Daily risk limit of {self.max_daily_risk:.1%} reached"
            
            # Check if we're in excessive drawdown
            if self.current_drawdown >= self.max_drawdown_threshold:
                return False, f"Maximum drawdown threshold exceeded: {self.current_drawdown:.2%}"
            
            # Check if there's adequate free margin
            required_margin = account_info['balance'] * self.max_risk_per_trade
            if required_margin > account_info['free_margin'] * 0.5:  # Only use up to 50% of free margin
                return False, "Insufficient free margin for trade"
                
            # Check market conditions
            if not symbol_info.get('trade_mode', 0) == 0:  # 0 is usually TRADE_MODE_FULL
                return False, "Trading not allowed for this symbol"
                
            # Check if market is open
            if not symbol_info.get('is_market_open', False):
                return False, "Market is closed for this symbol"
                
            # Additional checks based on consecutive losses
            if self.consecutive_losses >= 4:
                return False, f"Trading paused after {self.consecutive_losses} consecutive losses"
                
            return True, "Trade allowed"
            
        except Exception as e:
            self.logger.error(f"Error in can_place_trade: {e}")
            return False, f"Risk assessment error: {e}"
    
    def calculate_position_size(self, 
                              account_info: Dict[str, Any], 
                              entry_price: float, 
                              stop_loss: float, 
                              symbol_info: Dict[str, Any],
                              atr_value: float = None) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            account_info: Account information dictionary
            entry_price: Planned entry price
            stop_loss: Planned stop loss level
            symbol_info: Symbol information dictionary
            atr_value: ATR value for ATR-based sizing
            
        Returns:
            Position size in lots
        """
        try:
            if entry_price == stop_loss or stop_loss is None or entry_price is None:
                return symbol_info.get('volume_min', 0.01)  # Minimum lot size if no stop loss
                
            # Calculate risk amount in account currency
            risk_amount = account_info.get('balance', 0) * self.max_risk_per_trade
            
            # Apply drawdown scaling - reduce position sizes during drawdown
            if self.current_drawdown > 0.05:  # More than 5% drawdown
                # Linear reduction: at max_drawdown we trade at 50% size
                scale_factor = 1 - (self.current_drawdown / self.max_drawdown_threshold) * 0.5
                risk_amount *= max(0.5, scale_factor)  # Never go below 50% sizing
                
            # Apply consecutive loss scaling
            if self.consecutive_losses >= 1:
                loss_factor = 1.0 - (min(3, self.consecutive_losses) * 0.15)  # Reduce by up to 45%
                risk_amount *= loss_factor
                
            # Apply consecutive win scaling (anti-martingale)
            if self.consecutive_wins >= 2:
                win_factor = 1.0 + (min(3, self.consecutive_wins - 1) * 0.1)  # Increase by up to 30%
                risk_amount *= win_factor
            
            # Calculate based on selected method
            if self.position_sizing_method == "fixed":
                # Simple fixed percentage risk
                price_distance = abs(entry_price - stop_loss)
                
                # Get pip value for this symbol
                pip_value = self._calculate_pip_value(symbol_info)
                
                # Convert to number of pips
                pips_risked = price_distance / symbol_info.get('point', 0.0001) * 10
                
                # Calculate position size
                if pips_risked > 0 and pip_value > 0:
                    position_size = risk_amount / (pips_risked * pip_value)
                else:
                    position_size = symbol_info.get('volume_min', 0.01)
                    
            elif self.position_sizing_method == "atr":
                # ATR-based position sizing
                if atr_value is None or atr_value <= 0:
                    # Fallback to fixed if ATR not provided
                    return self.calculate_position_size(account_info, entry_price, stop_loss, symbol_info)
                    
                # Risk based on ATR
                pip_value = self._calculate_pip_value(symbol_info)
                
                # Use a multiple of ATR for risk calculation
                atr_multiple = 1.5  # Risk = 1.5 x ATR
                risk_pips = atr_value / symbol_info.get('point', 0.0001) * atr_multiple
                
                position_size = risk_amount / (risk_pips * pip_value) if risk_pips * pip_value > 0 else symbol_info.get('volume_min', 0.01)
                
            elif self.position_sizing_method == "volatility":
                # Volatility-based position sizing
                # Lower volatility = larger position, higher volatility = smaller position
                volatility = atr_value / entry_price if atr_value else 0.01
                vol_factor = 0.01 / max(0.001, volatility)  # Normalize volatility
                
                # Fixed position sizing adjusted by volatility
                price_distance = abs(entry_price - stop_loss)
                
                # Get pip value
                pip_value = self._calculate_pip_value(symbol_info)
                
                # Convert to pips
                pips_risked = price_distance / symbol_info.get('point', 0.0001) * 10
                
                # Calculate position size with volatility adjustment
                position_size = risk_amount * vol_factor / (pips_risked * pip_value) if pips_risked * pip_value > 0 else symbol_info.get('volume_min', 0.01)
                
            else:
                # Default to fixed sizing
                return self.calculate_position_size(account_info, entry_price, stop_loss, symbol_info)
            
            # Ensure position size is within allowed limits
            min_volume = symbol_info.get('volume_min', 0.01)
            max_volume = symbol_info.get('volume_max', 100.0)
            step = symbol_info.get('volume_step', 0.01)
            
            position_size = max(min_volume, min(position_size, max_volume))
            
            # Round to allowed step size
            position_size = round(position_size / step) * step
            
            # Update daily risk tracking
            self.daily_risk_used += self.max_risk_per_trade
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return symbol_info.get('volume_min', 0.01)
    
    def record_trade(self, trade_info: Dict[str, Any]) -> None:
        """
        Record trade information for analysis
        
        Args:
            trade_info: Trade information dictionary
        """
        try:
            if not trade_info:
                return
                
            # Add timestamp if not present
            if 'time' not in trade_info:
                trade_info['time'] = datetime.now()
                
            # Append to trade history
            self.trade_history.append(trade_info)
            
            # Update consecutive win/loss counters
            if trade_info.get('result') == 'win':
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            elif trade_info.get('result') == 'loss':
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
            # Log the trade
            self.logger.info(f"Trade recorded: {trade_info.get('symbol')} {trade_info.get('type')} "
                           f"Result: {trade_info.get('result')} Profit: {trade_info.get('profit_amount')}")
                           
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
        
    def should_adjust_positions(self, profit_factor: float) -> Tuple[bool, str, float]:
        """
        Determine if position sizing should be adjusted based on performance
        
        Args:
            profit_factor: Current profit factor
            
        Returns:
            Tuple of (should_adjust, adjustment_type, adjustment_factor)
        """
        try:
            if self.consecutive_losses >= 3:
                return True, "reduce", 0.5  # Reduce by 50% after 3 consecutive losses
                
            if profit_factor < 0.8 and len(self.trade_history) >= 10:
                return True, "reduce", 0.7  # Reduce by 30% if profit factor is poor
                
            if profit_factor > 1.5 and len(self.trade_history) >= 10:
                return True, "increase", 1.2  # Increase by 20% if profit factor is strong
                
            return False, "maintain", 1.0
            
        except Exception as e:
            self.logger.error(f"Error in should_adjust_positions: {e}")
            return False, "maintain", 1.0
        
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk metrics based on trade history
        
        Returns:
            Dict with risk metrics
        """
        try:
            if not self.trade_history:
                return {
                    "trades_taken": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "avg_risk_reward": 0,
                    "current_drawdown": self.current_drawdown,
                    "consecutive_wins": self.consecutive_wins,
                    "consecutive_losses": self.consecutive_losses
                }
                
            # Basic metrics
            total_trades = len(self.trade_history)
            wins = sum(1 for trade in self.trade_history if trade.get('result') == 'win')
            losses = sum(1 for trade in self.trade_history if trade.get('result') == 'loss')
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(trade.get('profit_amount', 0) for trade in self.trade_history 
                             if trade.get('profit_amount', 0) > 0)
            gross_loss = abs(sum(trade.get('profit_amount', 0) for trade in self.trade_history 
                              if trade.get('profit_amount', 0) < 0))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average risk-reward ratio
            avg_rr = np.mean([trade.get('risk_reward', 0) for trade in self.trade_history
                             if trade.get('risk_reward', 0) > 0])
            
            return {
                "trades_taken": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_risk_reward": avg_rr,
                "current_drawdown": self.current_drawdown,
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {
                "trades_taken": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_risk_reward": 0,
                "current_drawdown": self.current_drawdown,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "error": str(e)
            }
    
    def _calculate_pip_value(self, symbol_info: Dict[str, Any]) -> float:
        """
        Calculate the value of 1 pip for the symbol
        
        Args:
            symbol_info: Symbol information dictionary
            
        Returns:
            Value of 1 pip in account currency
        """
        try:
            # Extract values from symbol info
            contract_size = symbol_info.get('trade_contract_size', 100000)
            tick_size = symbol_info.get('trade_tick_size', 0.00001)
            tick_value = symbol_info.get('trade_tick_value', 1.0)
            
            # For most forex pairs, 1 pip = 0.0001, but for JPY pairs 1 pip = 0.01
            pip_size = 0.0001
            if 'JPY' in symbol_info.get('name', ''):
                pip_size = 0.01
                
            # Calculate pip value for 1 lot
            pips_per_tick = pip_size / tick_size
            pip_value = pips_per_tick * tick_value
            
            return pip_value
            
        except Exception as e:
            self.logger.error(f"Error calculating pip value: {e}")
            return 10.0  # Default estimate
            
    def analyze_exposure(self, open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze current exposure across all open positions
        
        Args:
            open_positions: List of open positions
            
        Returns:
            Dict with exposure analysis
        """
        try:
            if not open_positions:
                return {
                    "total_exposure": 0,
                    "exposure_by_currency": {},
                    "exposure_by_direction": {"long": 0, "short": 0},
                    "largest_position": None
                }
                
            # Calculate total exposure
            total_exposure = sum(pos.get('volume', 0) for pos in open_positions)
            
            # Calculate exposure by currency
            exposure_by_currency = {}
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                volume = pos.get('volume', 0)
                
                # Extract base and quote currencies
                if len(symbol) >= 6:
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:6]
                    
                    # Add to base currency
                    if base_currency not in exposure_by_currency:
                        exposure_by_currency[base_currency] = 0
                        
                    # Add to quote currency
                    if quote_currency not in exposure_by_currency:
                        exposure_by_currency[quote_currency] = 0
                        
                    # For long positions, +base, -quote
                    if pos.get('type', '') == 'buy':
                        exposure_by_currency[base_currency] += volume
                        exposure_by_currency[quote_currency] -= volume
                    # For short positions, -base, +quote
                    else:
                        exposure_by_currency[base_currency] -= volume
                        exposure_by_currency[quote_currency] += volume
                        
            # Calculate exposure by direction
            long_exposure = sum(pos.get('volume', 0) for pos in open_positions if pos.get('type', '') == 'buy')
            short_exposure = sum(pos.get('volume', 0) for pos in open_positions if pos.get('type', '') == 'sell')
            
            # Find largest position
            if open_positions:
                largest_position = max(open_positions, key=lambda pos: pos.get('volume', 0))
            else:
                largest_position = None
                
            return {
                "total_exposure": total_exposure,
                "exposure_by_currency": exposure_by_currency,
                "exposure_by_direction": {"long": long_exposure, "short": short_exposure},
                "largest_position": largest_position,
                "net_direction": "long" if long_exposure > short_exposure else "short",
                "direction_bias": abs(long_exposure - short_exposure) / total_exposure if total_exposure > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing exposure: {e}")
            return {
                "total_exposure": 0,
                "exposure_by_currency": {},
                "exposure_by_direction": {"long": 0, "short": 0},
                "largest_position": None,
                "error": str(e)
            }