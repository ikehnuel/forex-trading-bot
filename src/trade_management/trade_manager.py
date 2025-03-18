import pandas as pd
import numpy as np
import logging
import sys
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_CONFIG, TRADE_HISTORY_DIR, logger
from mt5_connector.order_executor import OrderExecutor

class TradeManager:
    """
    Manages the lifecycle of trades including entry, exit,
    position management, and dynamic stop loss/take profit adjustments.
    """
    def __init__(self, order_executor=None):
        self.logger = logging.getLogger(__name__)
        
        # Use provided order executor or create a new one
        if order_executor is None:
            try:
                self.order_executor = OrderExecutor()
            except Exception as e:
                self.logger.error(f"Could not initialize order executor: {e}")
                self.order_executor = None
        else:
            self.order_executor = order_executor
            
        # Initialize trade tracking
        self.active_trades = {}
        self.trade_history = []
        
        # Load configuration
        self.trailing_stop_params = TRADING_CONFIG.get('trailing_stop', {
            'enable': True,
            'activation_percent': 0.5,
            'step_percent': 0.2,
            'lock_percent': 0.6
        })
        
        self.partial_exit_params = TRADING_CONFIG.get('partial_exit', {
            'enable': True,
            'first_target_percent': 0.4,
            'first_exit_percent': 0.3,
            'second_target_percent': 0.7,
            'second_exit_percent': 0.3
        })
        
        self.breakeven_params = TRADING_CONFIG.get('breakeven', {
            'enable': True,
            'activation_percent': 0.3,
            'buffer_pips': 5
        })
        
        # Load historical trades
        self._load_trade_history()
        
    def add_trade(self, ticket: int, symbol: str, order_type: str, volume: float,
                entry_price: float, stop_loss: float, take_profit: float, risk_amount: float) -> bool:
        """
        Add a new trade to be managed
        
        Args:
            ticket: Trade ticket number
            symbol: Symbol being traded
            order_type: 'buy' or 'sell'
            volume: Trade volume in lots
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            risk_amount: Amount risked in account currency
            
        Returns:
            bool: Success or failure
        """
        try:
            if ticket in self.active_trades:
                self.logger.warning(f"Trade {ticket} already exists in active trades")
                return False
                
            # Create trade object
            self.active_trades[ticket] = {
                'ticket': ticket,
                'symbol': symbol,
                'type': order_type,  # 'buy' or 'sell'
                'volume': volume,
                'initial_volume': volume,  # Track initial volume for partial exits
                'entry_price': entry_price,
                'initial_stop_loss': stop_loss,
                'current_stop_loss': stop_loss,
                'initial_take_profit': take_profit,
                'take_profit': take_profit,
                'risk_amount': risk_amount,
                'entry_time': datetime.now(),
                'partial_exits': [],
                'moved_to_breakeven': False,
                'trailing_activated': False,
                'risk_reward': abs(take_profit - entry_price) / abs(entry_price - stop_loss) if stop_loss else 0,
                'status': 'open',
                'max_favorable_excursion': 0,  # Track maximum profit reached
                'max_adverse_excursion': 0,    # Track maximum drawdown reached
                'profit_pips': 0,
                'profit_amount': 0,
                'management_actions': []
            }
            
            self.logger.info(f"Added trade {ticket} for {symbol} ({order_type}) at {entry_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding trade {ticket}: {e}")
            return False
        
    def update_trades(self, mt5_positions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Update trade status based on current positions
        
        Args:
            mt5_positions: List of current positions from MT5
            
        Returns:
            Dict with update statistics
        """
        try:
            if mt5_positions is None:
                mt5_positions = []
                
            # Convert to dictionary for easy lookup
            position_dict = {pos.get('ticket'): pos for pos in mt5_positions}
            
            # Find closed positions
            current_tickets = set(position_dict.keys())
            managed_tickets = set(self.active_trades.keys())
            closed_tickets = managed_tickets - current_tickets
            
            # Update closed positions
            for ticket in closed_tickets:
                trade = self.active_trades.get(ticket)
                if trade:
                    # Mark as closed and move to history
                    self._close_trade(ticket, position_dict)
                    
            # Update active positions
            for ticket, trade in list(self.active_trades.items()):
                position = position_dict.get(ticket)
                
                if position:
                    # Update current values
                    self._update_trade_metrics(trade, position)
                    
            # Save updated history
            self._save_trade_history()
            
            return {
                'active_count': len(self.active_trades),
                'closed_count': len(closed_tickets)
            }
            
        except Exception as e:
            self.logger.error(f"Error updating trades: {e}")
            return {
                'active_count': len(self.active_trades),
                'closed_count': 0,
                'error': str(e)
            }
        
    def manage_all_trades(self) -> int:
        """
        Actively manage all open trades
        
        Returns:
            int: Number of trades modified
        """
        try:
            if not self.order_executor:
                self.logger.error("Order executor not available, cannot manage trades")
                return 0
                
            # Get current open positions
            open_positions = self.order_executor.get_open_positions()
            if not open_positions:
                return 0
                
            # Update trade status first
            self.update_trades(open_positions)
            
            # Apply trade management strategies
            modified_count = 0
            
            for ticket, trade in list(self.active_trades.items()):
                symbol = trade['symbol']
                
                # Get current price from MT5
                current_prices = self.order_executor.get_current_prices([symbol])
                if not current_prices or symbol not in current_prices:
                    continue
                    
                current_price = current_prices[symbol]['bid'] if trade['type'] == 'buy' else current_prices[symbol]['ask']
                
                # Skip if price unavailable
                if current_price <= 0:
                    continue
                    
                # Apply trade management strategies
                modified = False
                
                # 1. Check partial exits
                if self._check_partial_exits(trade, current_price):
                    modified = True
                    modified_count += 1
                    
                # 2. Check moving to breakeven
                if self._check_breakeven(trade, current_price):
                    modified = True
                    modified_count += 1
                    
                # 3. Check trailing stop
                if self._check_trailing_stop(trade, current_price):
                    modified = True
                    modified_count += 1
                    
                # Apply any stop loss modifications
                if modified:
                    result = self.order_executor.modify_position(
                        ticket=trade['ticket'],
                        stop_loss=trade['current_stop_loss'],
                        take_profit=trade['take_profit']
                    )
                    
                    if result and result.get('result') == 'success':
                        self.logger.info(f"Modified trade {ticket}: SL={trade['current_stop_loss']}, TP={trade['take_profit']}")
                    else:
                        self.logger.warning(f"Failed to modify trade {ticket}: {result}")
                        
            return modified_count
            
        except Exception as e:
            self.logger.error(f"Error managing trades: {e}")
            return 0
        
    def close_all_trades(self, reason: str = "Manual close all") -> Dict[str, Any]:
        """
        Close all open trades
        
        Args:
            reason: Reason for closing all trades
            
        Returns:
            Dict with results
        """
        try:
            if not self.order_executor:
                self.logger.error("Order executor not available, cannot close trades")
                return {'success': False, 'message': 'Order executor not available'}
                
            if not self.active_trades:
                return {'success': True, 'message': 'No active trades to close', 'closed_count': 0}
                
            # Close each trade
            success_count = 0
            failed_count = 0
            results = []
            
            for ticket in list(self.active_trades.keys()):
                result = self.order_executor.close_position(ticket, f"Close all: {reason}")
                
                if result and result.get('result') == 'success':
                    success_count += 1
                    self.active_trades[ticket]['status'] = 'closed'
                    self.active_trades[ticket]['close_reason'] = reason
                    results.append({
                        'ticket': ticket,
                        'status': 'closed',
                        'profit': result.get('profit', 0)
                    })
                else:
                    failed_count += 1
                    results.append({
                        'ticket': ticket,
                        'status': 'failed',
                        'message': result.get('retcode_text', 'Unknown error')
                    })
                    
            # Update trades to get final status
            open_positions = self.order_executor.get_open_positions()
            self.update_trades(open_positions)
            
            return {
                'success': failed_count == 0,
                'message': f"Closed {success_count} trades, {failed_count} failed",
                'closed_count': success_count,
                'failed_count': failed_count,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error closing all trades: {e}")
            return {
                'success': False,
                'message': f"Error: {e}",
                'closed_count': 0,
                'failed_count': len(self.active_trades)
            }
        
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of all active trades
        
        Returns:
            List of active trade dictionaries
        """
        return list(self.active_trades.values())
        
    def get_trade_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get trade history for the specified number of days
        
        Args:
            days: Number of days of history to return
            
        Returns:
            List of trade dictionaries
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter trades by date
        recent_trades = [
            trade for trade in self.trade_history
            if (isinstance(trade.get('entry_time'), datetime) and trade.get('entry_time') >= cutoff_date)
            or (isinstance(trade.get('entry_time'), str) and 
                datetime.strptime(trade.get('entry_time'), '%Y-%m-%d %H:%M:%S') >= cutoff_date)
        ]
        
        return recent_trades
        
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate trading statistics
        
        Returns:
            Dict with trading statistics
        """
        try:
            # Combine active and closed trades
            all_trades = self.trade_history + list(self.active_trades.values())
            
            if not all_trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_pips': 0,
                    'total_profit': 0
                }
                
            # Calculate statistics
            closed_trades = [t for t in all_trades if t.get('status') == 'closed']
            winning_trades = [t for t in closed_trades if t.get('profit_amount', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('profit_amount', 0) <= 0]
            
            # Basic metrics
            total_trades = len(closed_trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            total_profit = sum(t.get('profit_amount', 0) for t in closed_trades)
            gross_profit = sum(t.get('profit_amount', 0) for t in winning_trades)
            gross_loss = sum(t.get('profit_amount', 0) for t in losing_trades)
            
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0
            
            # Pip metrics
            total_pips = sum(t.get('profit_pips', 0) for t in closed_trades)
            
            # Current open profit
            open_profit = sum(t.get('profit_amount', 0) for t in all_trades if t.get('status') == 'open')
            
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_pips': total_pips,
                'total_profit': total_profit,
                'open_profit': open_profit,
                'avg_win': gross_profit / win_count if win_count > 0 else 0,
                'avg_loss': gross_loss / loss_count if loss_count > 0 else 0,
                'largest_win': max([t.get('profit_amount', 0) for t in winning_trades]) if winning_trades else 0,
                'largest_loss': min([t.get('profit_amount', 0) for t in losing_trades]) if losing_trades else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pips': 0,
                'total_profit': 0,
                'error': str(e)
            }
        
    def _check_partial_exits(self, trade: Dict[str, Any], current_price: float) -> bool:
        """
        Check and execute partial exits based on profit targets
        
        Args:
            trade: Trade dictionary
            current_price: Current price
            
        Returns:
            bool: True if partial exit was executed
        """
        try:
            if not self.partial_exit_params.get('enable', False):
                return False
                
            # Skip if already fully exited or volume too small
            if len(trade.get('partial_exits', [])) >= 2 or trade.get('volume', 0) <= 0.01:
                return False
                
            # Calculate distance to target
            entry_price = trade.get('entry_price', 0)
            take_profit = trade.get('initial_take_profit', 0)
            target_distance = abs(take_profit - entry_price)
            
            if target_distance <= 0:
                return False
                
            # Check first target
            first_target_percent = self.partial_exit_params.get('first_target_percent', 0.4)
            first_target_distance = target_distance * first_target_percent
            first_target_reached = False
            
            if trade.get('type') == 'buy':
                first_target_price = entry_price + first_target_distance
                first_target_reached = current_price >= first_target_price
            else:
                first_target_price = entry_price - first_target_distance
                first_target_reached = current_price <= first_target_price
                
            # Check if first target reached and not yet executed
            first_exit_done = any(exit_info.get('target') == 'first' for exit_info in trade.get('partial_exits', []))
            
            if first_target_reached and not first_exit_done:
                # Calculate exit volume
                exit_percent = self.partial_exit_params.get('first_exit_percent', 0.3)
                exit_volume = trade.get('volume', 0) * exit_percent
                
                # Ensure minimum volume
                if exit_volume < 0.01:
                    exit_volume = min(0.01, trade.get('volume', 0))
                    
                # Execute partial exit if we have an order executor
                if self.order_executor and exit_volume > 0:
                    result = self.order_executor.close_position_partial(
                        ticket=trade.get('ticket'),
                        volume=exit_volume,
                        comment="Partial exit - first target"
                    )
                    
                    if result and result.get('result') == 'success':
                        # Record the partial exit
                        trade['partial_exits'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'price': current_price,
                            'percent': exit_percent,
                            'volume': exit_volume,
                            'target': 'first',
                            'profit': result.get('profit', 0)
                        })
                        
                        # Update trade volume
                        trade['volume'] -= exit_volume
                        
                        # Log the action
                        trade['management_actions'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'partial_exit',
                            'target': 'first',
                            'volume': exit_volume,
                            'price': current_price
                        })
                        
                        self.logger.info(f"Executed first partial exit for trade {trade.get('ticket')}: "
                                         f"{exit_volume} lots at {current_price}")
                        
                        return True
                    else:
                        self.logger.warning(f"Failed to execute first partial exit for trade {trade.get('ticket')}: {result}")
                        
            # Check second target
            second_target_percent = self.partial_exit_params.get('second_target_percent', 0.7)
            second_target_distance = target_distance * second_target_percent
            second_target_reached = False
            
            if trade.get('type') == 'buy':
                second_target_price = entry_price + second_target_distance
                second_target_reached = current_price >= second_target_price
            else:
                second_target_price = entry_price - second_target_distance
                second_target_reached = current_price <= second_target_price
                
            # Check if second target reached and not yet executed
            second_exit_done = any(exit_info.get('target') == 'second' for exit_info in trade.get('partial_exits', []))
            
            if second_target_reached and not second_exit_done and first_exit_done:
                # Calculate exit volume
                exit_percent = self.partial_exit_params.get('second_exit_percent', 0.3)
                exit_volume = trade.get('initial_volume', 0) * exit_percent  # Based on initial volume
                
                # Ensure minimum volume and don't exceed remaining
                if exit_volume < 0.01:
                    exit_volume = min(0.01, trade.get('volume', 0))
                    
                exit_volume = min(exit_volume, trade.get('volume', 0))
                
                # Execute partial exit if we have an order executor
                if self.order_executor and exit_volume > 0:
                    result = self.order_executor.close_position_partial(
                        ticket=trade.get('ticket'),
                        volume=exit_volume,
                        comment="Partial exit - second target"
                    )
                    
                    if result and result.get('result') == 'success':
                        # Record the partial exit
                        trade['partial_exits'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'price': current_price,
                            'percent': exit_percent,
                            'volume': exit_volume,
                            'target': 'second',
                            'profit': result.get('profit', 0)
                        })
                        
                        # Update trade volume
                        trade['volume'] -= exit_volume
                        
                        # Log the action
                        trade['management_actions'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'partial_exit',
                            'target': 'second',
                            'volume': exit_volume,
                            'price': current_price
                        })
                        
                        self.logger.info(f"Executed second partial exit for trade {trade.get('ticket')}: "
                                         f"{exit_volume} lots at {current_price}")
                        
                        return True
                    else:
                        self.logger.warning(f"Failed to execute second partial exit for trade {trade.get('ticket')}: {result}")
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking partial exits for trade {trade.get('ticket')}: {e}")
            return False
        
    def _check_breakeven(self, trade: Dict[str, Any], current_price: float) -> bool:
        """
        Check and move stop loss to breakeven if conditions met
        
        Args:
            trade: Trade dictionary
            current_price: Current price
            
        Returns:
            bool: True if stop loss was moved to breakeven
        """
        try:
            if not self.breakeven_params.get('enable', False) or trade.get('moved_to_breakeven', False):
                return False
                
            # Calculate distance to target
            entry_price = trade.get('entry_price', 0)
            take_profit = trade.get('initial_take_profit', 0)
            target_distance = abs(take_profit - entry_price)
            
            if target_distance <= 0:
                return False
                
            # Calculate activation threshold
            activation_percent = self.breakeven_params.get('activation_percent', 0.3)
            activation_distance = target_distance * activation_percent
            
            # Check if price has moved enough to activate breakeven
            activation_reached = False
            
            if trade.get('type') == 'buy':
                activation_price = entry_price + activation_distance
                activation_reached = current_price >= activation_price
            else:
                activation_price = entry_price - activation_distance
                activation_reached = current_price <= activation_price
                
            if activation_reached:
                # Calculate breakeven level with buffer
                buffer_pips = self.breakeven_params.get('buffer_pips', 5)
                
                # Convert pips to price
                symbol = trade.get('symbol', '')
                pip_size = 0.0001
                if 'JPY' in symbol:
                    pip_size = 0.01
                    
                buffer = buffer_pips * pip_size
                
                # Set breakeven level
                if trade.get('type') == 'buy':
                    breakeven_level = entry_price + buffer
                else:
                    breakeven_level = entry_price - buffer
                    
                # Only update if better than current stop
                if trade.get('type') == 'buy' and breakeven_level > trade.get('current_stop_loss', 0):
                    trade['current_stop_loss'] = breakeven_level
                    trade['moved_to_breakeven'] = True
                    
                    # Log the action
                    trade['management_actions'].append({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'move_to_breakeven',
                        'old_stop_loss': trade.get('current_stop_loss', 0),
                        'new_stop_loss': breakeven_level,
                        'price': current_price
                    })
                    
                    self.logger.info(f"Moved trade {trade.get('ticket')} to breakeven+: SL={breakeven_level}")
                    return True
                    
                elif trade.get('type') == 'sell' and breakeven_level < trade.get('current_stop_loss', float('inf')):
                    trade['current_stop_loss'] = breakeven_level
                    trade['moved_to_breakeven'] = True
                    
                    # Log the action
                    trade['management_actions'].append({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'move_to_breakeven',
                        'old_stop_loss': trade.get('current_stop_loss', 0),
                        'new_stop_loss': breakeven_level,
                        'price': current_price
                    })
                    
                    self.logger.info(f"Moved trade {trade.get('ticket')} to breakeven+: SL={breakeven_level}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking breakeven for trade {trade.get('ticket')}: {e}")
            return False
        
    def _check_trailing_stop(self, trade: Dict[str, Any], current_price: float) -> bool:
        """
        Check and update trailing stop if conditions met
        
        Args:
            trade: Trade dictionary
            current_price: Current price
            
        Returns:
            bool: True if trailing stop was updated
        """
        try:
            if not self.trailing_stop_params.get('enable', False):
                return False
                
            # First check if trailing stop should be activated
            if not trade.get('trailing_activated', False):
                # Calculate distance to target
                entry_price = trade.get('entry_price', 0)
                take_profit = trade.get('initial_take_profit', 0)
                target_distance = abs(take_profit - entry_price)
                
                if target_distance <= 0:
                    return False
                    
                # Calculate activation threshold
                activation_percent = self.trailing_stop_params.get('activation_percent', 0.5)
                activation_distance = target_distance * activation_percent
                
                # Check if price has moved enough to activate trailing stop
                activation_reached = False
                
                if trade.get('type') == 'buy':
                    activation_price = entry_price + activation_distance
                    activation_reached = current_price >= activation_price
                else:
                    activation_price = entry_price - activation_distance
                    activation_reached = current_price <= activation_price
                    
                if activation_reached:
                    trade['trailing_activated'] = True
                    
                    # Log the action
                    trade['management_actions'].append({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'trailing_stop_activated',
                        'price': current_price
                    })
                    
                    self.logger.info(f"Trailing stop activated for trade {trade.get('ticket')}")
            
            # Update trailing stop if activated
            if trade.get('trailing_activated', False):
                # Calculate current profit in price
                entry_price = trade.get('entry_price', 0)
                current_profit = 0
                
                if trade.get('type') == 'buy':
                    current_profit = current_price - entry_price
                else:
                    current_profit = entry_price - current_price
                    
                # Only trail if in profit
                if current_profit <= 0:
                    return False
                    
                # Calculate how much of the profit to lock in
                lock_percent = self.trailing_stop_params.get('lock_percent', 0.6)
                lock_amount = current_profit * lock_percent
                
                # Calculate new stop loss level
                if trade.get('type') == 'buy':
                    new_stop = entry_price + lock_amount
                    if new_stop > trade.get('current_stop_loss', 0):
                        old_stop = trade.get('current_stop_loss', 0)
                        trade['current_stop_loss'] = new_stop
                        
                        # Log the action
                        trade['management_actions'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'trailing_stop_update',
                            'old_stop_loss': old_stop,
                            'new_stop_loss': new_stop,
                            'price': current_price
                        })
                        
                        self.logger.info(f"Trailing stop updated for trade {trade.get('ticket')}: SL={new_stop}")
                        return True
                        
                else:
                    new_stop = entry_price - lock_amount
                    if new_stop < trade.get('current_stop_loss', float('inf')):
                        old_stop = trade.get('current_stop_loss', 0)
                        trade['current_stop_loss'] = new_stop
                        
                        # Log the action
                        trade['management_actions'].append({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'trailing_stop_update',
                            'old_stop_loss': old_stop,
                            'new_stop_loss': new_stop,
                            'price': current_price
                        })
                        
                        self.logger.info(f"Trailing stop updated for trade {trade.get('ticket')}: SL={new_stop}")
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trailing stop for trade {trade.get('ticket')}: {e}")
            return False
        
    def _update_trade_metrics(self, trade: Dict[str, Any], position: Dict[str, Any]) -> None:
        """
        Update trade metrics based on current position data
        
        Args:
            trade: Trade dictionary to update
            position: Current position data from MT5
        """
        try:
            # Update current values
            current_price = position.get('current_price', 0)
            
            # Calculate profit in pips
            entry_price = trade.get('entry_price', 0)
            symbol = trade.get('symbol', '')
            
            # Determine pip size
            pip_size = 0.0001
            if 'JPY' in symbol:
                pip_size = 0.01
                
            # Calculate pip profit
            if trade.get('type') == 'buy':
                current_profit_pips = (current_price - entry_price) / pip_size
            else:
                current_profit_pips = (entry_price - current_price) / pip_size
                
            # Update trade metrics
            trade['profit_pips'] = current_profit_pips
            trade['profit_amount'] = position.get('profit', 0)
            
            # Track maximum favorable excursion (maximum profit)
            if current_profit_pips > trade.get('max_favorable_excursion', -float('inf')):
                trade['max_favorable_excursion'] = current_profit_pips
                
            # Track maximum adverse excursion (maximum drawdown)
            if current_profit_pips < trade.get('max_adverse_excursion', float('inf')):
                trade['max_adverse_excursion'] = current_profit_pips
                
        except Exception as e:
            self.logger.error(f"Error updating trade metrics for trade {trade.get('ticket')}: {e}")
        
    def _close_trade(self, ticket: int, position_dict: Dict[int, Dict[str, Any]]) -> None:
        """
        Process a closed trade
        
        Args:
            ticket: Trade ticket number
            position_dict: Dictionary of current positions
        """
        try:
            trade = self.active_trades.get(ticket)
            if not trade:
                return
                
            # Get close details from MT5 history if possible
            if self.order_executor:
                # Get deal history
                deal_history = self.order_executor.get_deal_history(1)  # Last day should be enough
                
                # Find the closing deal for this ticket
                closing_deal = None
                for deal in deal_history:
                    if deal.get('order') == ticket and deal.get('entry') == 'out':
                        closing_deal = deal
                        break
                        
                if closing_deal:
                    # Update trade with closing details
                    trade['close_price'] = closing_deal.get('price', 0)
                    trade['close_time'] = closing_deal.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    trade['profit_amount'] = closing_deal.get('profit', 0)
                    
                    # Calculate pip profit
                    entry_price = trade.get('entry_price', 0)
                    close_price = trade.get('close_price', 0)
                    symbol = trade.get('symbol', '')
                    
                    # Determine pip size
                    pip_size = 0.0001
                    if 'JPY' in symbol:
                        pip_size = 0.01
                        
                    # Calculate pip profit
                    if trade.get('type') == 'buy':
                        trade['profit_pips'] = (close_price - entry_price) / pip_size
                    else:
                        trade['profit_pips'] = (entry_price - close_price) / pip_size
                        
                    # Set result
                    trade['result'] = 'win' if trade.get('profit_amount', 0) > 0 else 'loss'
                    
            # If we couldn't get details from history, estimate them
            if 'close_price' not in trade:
                # Get last price from position if available
                last_position = position_dict.get(ticket)
                
                if last_position:
                    trade['close_price'] = last_position.get('current_price', trade.get('entry_price', 0))
                    trade['profit_amount'] = last_position.get('profit', 0)
                else:
                    # Use entry price as fallback
                    trade['close_price'] = trade.get('entry_price', 0)
                    trade['profit_amount'] = 0
                    
                trade['close_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate pip profit
                entry_price = trade.get('entry_price', 0)
                close_price = trade.get('close_price', 0)
                symbol = trade.get('symbol', '')
                
                # Determine pip size
                pip_size = 0.0001
                if 'JPY' in symbol:
                    pip_size = 0.01
                    
                # Calculate pip profit
                if trade.get('type') == 'buy':
                    trade['profit_pips'] = (close_price - entry_price) / pip_size
                else:
                    trade['profit_pips'] = (entry_price - close_price) / pip_size
                    
                # Set result
                trade['result'] = 'win' if trade.get('profit_amount', 0) > 0 else 'loss'
                
            # Set status to closed
            trade['status'] = 'closed'
            
            # Calculate trade duration
            if isinstance(trade.get('entry_time'), datetime) and isinstance(trade.get('close_time'), datetime):
                duration = trade.get('close_time') - trade.get('entry_time')
                trade['duration_hours'] = duration.total_seconds() / 3600
            elif isinstance(trade.get('entry_time'), str) and isinstance(trade.get('close_time'), str):
                try:
                    entry_time = datetime.strptime(trade.get('entry_time'), '%Y-%m-%d %H:%M:%S')
                    close_time = datetime.strptime(trade.get('close_time'), '%Y-%m-%d %H:%M:%S')
                    duration = close_time - entry_time
                    trade['duration_hours'] = duration.total_seconds() / 3600
                except:
                    trade['duration_hours'] = 0
            else:
                trade['duration_hours'] = 0
                
            # Move to history
            self.trade_history.append(trade)
            del self.active_trades[ticket]
            
            self.logger.info(f"Trade {ticket} closed: {trade.get('result')} "
                           f"Profit: {trade.get('profit_pips', 0):.1f} pips, "
                           f"{trade.get('profit_amount', 0):.2f} {trade.get('currency', '')}")
                           
        except Exception as e:
            self.logger.error(f"Error processing closed trade {ticket}: {e}")
            
            # In case of error, still mark as closed and move to history
            if ticket in self.active_trades:
                trade = self.active_trades[ticket]
                trade['status'] = 'closed'
                trade['close_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                trade['error'] = str(e)
                
                self.trade_history.append(trade)
                del self.active_trades[ticket]
        
    def _load_trade_history(self) -> None:
        """Load trade history from file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
            
            # Find latest history file
            history_files = [f for f in os.listdir(TRADE_HISTORY_DIR) if f.endswith('.json')]
            
            if not history_files:
                self.logger.info("No trade history files found")
                return
                
            # Sort by date
            history_files.sort(reverse=True)
            latest_file = os.path.join(TRADE_HISTORY_DIR, history_files[0])
            
            # Load history
            with open(latest_file, 'r') as f:
                history_data = json.load(f)
                
            if isinstance(history_data, dict):
                self.trade_history = history_data.get('trade_history', [])
            else:
                self.trade_history = history_data
                
            self.logger.info(f"Loaded {len(self.trade_history)} trades from history")
            
        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
            self.trade_history = []
        
    def _save_trade_history(self) -> None:
        """Save trade history to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)
            
            # Create filename with current date
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = os.path.join(TRADE_HISTORY_DIR, filename)
            
            # Prepare data
            history_data = {
                'trade_history': self.trade_history,
                'active_trades': list(self.active_trades.values()),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2, default=self._json_serializer)
                
            self.logger.debug(f"Saved trade history to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")
        
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default"""
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Type {type(obj)} not serializable")