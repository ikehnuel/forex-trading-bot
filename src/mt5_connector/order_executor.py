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
from config import logger
from mt5_connector.connection import MT5Connector

class OrderExecutor:
    """
    Handles trade execution, modification, and closure via MetaTrader 5.
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
            
        # Initialize order counters
        self.order_count = 0
        
    def open_position(self, symbol, order_type, volume, price=0.0, stop_loss=0.0, take_profit=0.0, comment=None, magic=0):
        """
        Open a new position
        
        Args:
            symbol (str): Symbol to trade
            order_type (str): 'buy' or 'sell'
            volume (float): Trade volume in lots
            price (float): Price to open at (0 for market)
            stop_loss (float): Stop loss level
            take_profit (float): Take profit level
            comment (str): Comment for the order
            magic (int): Magic number for the order
            
        Returns:
            dict: Order result or None if failed
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot open position")
                return None
                
            # Check if trading is allowed
            terminal_info = mt5.terminal_info()
            if not terminal_info.trade_allowed:
                self.logger.error("Trading is not allowed in the terminal")
                return None
                
            # Check if symbol is valid
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None
                
            # Ensure symbol is selected for trading
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}")
                    return None
                    
            # Convert order type to MT5 constant
            if order_type.lower() == 'buy':
                mt5_order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask if price <= 0 else price
            elif order_type.lower() == 'sell':
                mt5_order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid if price <= 0 else price
            else:
                self.logger.error(f"Invalid order type: {order_type}")
                return None
                
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5_order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,  # Price deviation in points
                "magic": magic,
                "comment": comment or f"Claude Bot {datetime.now().strftime('%Y%m%d%H%M')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Log order details
            self.logger.info(f"Sending order: {order_type.upper()} {symbol} volume={volume} price={price} SL={stop_loss} TP={take_profit}")
            
            # Send order
            result = mt5.order_send(request)
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.order_count += 1
                
                # Return success result
                success_result = {
                    'ticket': result.order,
                    'result': 'success',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'volume': volume,
                    'price': price,
                    'symbol': symbol,
                    'type': order_type.lower(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.info(f"Order executed successfully: ticket={result.order}")
                return success_result
            else:
                # Return failure result
                failure_result = {
                    'result': 'failure',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'comment': result.comment,
                    'symbol': symbol,
                    'type': order_type.lower(),
                    'volume': volume,
                    'price': price,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.error(f"Order failed: {result.retcode} - {self._get_retcode_text(result.retcode)} - {result.comment}")
                return failure_result
                
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return None
            
    def close_position(self, ticket, comment=None):
        """
        Close an existing position by ticket number
        
        Args:
            ticket (int): Position ticket number
            comment (str): Comment for the order
            
        Returns:
            dict: Result of the close operation
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot close position")
                return None
                
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                self.logger.error(f"Position {ticket} not found: {mt5.last_error()}")
                return None
                
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": comment or f"Claude Bot Close {datetime.now().strftime('%Y%m%d%H%M')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Log close details
            self.logger.info(f"Closing position: ticket={ticket} symbol={position.symbol} volume={position.volume}")
            
            # Send close request
            result = mt5.order_send(request)
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Return success result
                success_result = {
                    'ticket': ticket,
                    'result': 'success',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'symbol': position.symbol,
                    'volume': position.volume,
                    'close_price': request['price'],
                    'profit': position.profit,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.info(f"Position {ticket} closed successfully, profit: {position.profit}")
                return success_result
            else:
                # Return failure result
                failure_result = {
                    'ticket': ticket,
                    'result': 'failure',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'comment': result.comment,
                    'symbol': position.symbol,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.error(f"Failed to close position {ticket}: {result.retcode} - {self._get_retcode_text(result.retcode)} - {result.comment}")
                return failure_result
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return None
            
    def close_position_partial(self, ticket, volume, comment=None):
        """
        Partially close an existing position
        
        Args:
            ticket (int): Position ticket number
            volume (float): Volume to close
            comment (str): Comment for the order
            
        Returns:
            dict: Result of the partial close operation
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot partial close position")
                return None
                
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                self.logger.error(f"Position {ticket} not found: {mt5.last_error()}")
                return None
                
            position = position[0]
            
            # Check if requested volume is valid
            if volume >= position.volume:
                self.logger.warning(f"Requested volume {volume} is >= position volume {position.volume}, closing full position")
                return self.close_position(ticket, comment)
                
            # Prepare partial close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": comment or f"Claude Bot Partial Close {datetime.now().strftime('%Y%m%d%H%M')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Log partial close details
            self.logger.info(f"Partially closing position: ticket={ticket} symbol={position.symbol} volume={volume}/{position.volume}")
            
            # Send partial close request
            result = mt5.order_send(request)
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Return success result
                success_result = {
                    'ticket': ticket,
                    'result': 'success',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'symbol': position.symbol,
                    'volume_closed': volume,
                    'volume_remaining': position.volume - volume,
                    'close_price': request['price'],
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.info(f"Position {ticket} partially closed successfully, volume closed: {volume}")
                return success_result
            else:
                # Return failure result
                failure_result = {
                    'ticket': ticket,
                    'result': 'failure',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'comment': result.comment,
                    'symbol': position.symbol,
                    'volume': volume,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.error(f"Failed to partially close position {ticket}: {result.retcode} - {self._get_retcode_text(result.retcode)} - {result.comment}")
                return failure_result
                
        except Exception as e:
            self.logger.error(f"Error partially closing position {ticket}: {e}")
            return None
            
    def modify_position(self, ticket, stop_loss=None, take_profit=None):
        """
        Modify stop loss and take profit for an existing position
        
        Args:
            ticket (int): Position ticket number
            stop_loss (float): New stop loss level, or None to leave unchanged
            take_profit (float): New take profit level, or None to leave unchanged
            
        Returns:
            dict: Result of the modification operation
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot modify position")
                return None
                
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                self.logger.error(f"Position {ticket} not found: {mt5.last_error()}")
                return None
                
            position = position[0]
            
            # Use current values if not specified
            sl = stop_loss if stop_loss is not None else position.sl
            tp = take_profit if take_profit is not None else position.tp
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": position.symbol,
                "sl": sl,
                "tp": tp
            }
            
            # Log modification details
            self.logger.info(f"Modifying position: ticket={ticket} symbol={position.symbol} new_sl={sl} new_tp={tp}")
            
            # Send modification request
            result = mt5.order_send(request)
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Return success result
                success_result = {
                    'ticket': ticket,
                    'result': 'success',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'symbol': position.symbol,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.info(f"Position {ticket} modified successfully")
                return success_result
            else:
                # Return failure result
                failure_result = {
                    'ticket': ticket,
                    'result': 'failure',
                    'retcode': result.retcode,
                    'retcode_text': self._get_retcode_text(result.retcode),
                    'comment': result.comment,
                    'symbol': position.symbol,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.logger.error(f"Failed to modify position {ticket}: {result.retcode} - {self._get_retcode_text(result.retcode)} - {result.comment}")
                return failure_result
                
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {e}")
            return None
            
    def get_open_positions(self, symbol=None):
        """
        Get open positions
        
        Args:
            symbol (str): Symbol to filter positions, or None for all positions
            
        Returns:
            list: List of open positions
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot get positions")
                return []
                
            # Get positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                self.logger.error(f"Failed to get positions: {mt5.last_error()}")
                return []
                
            # Convert to list of dictionaries
            position_list = []
            
            for position in positions:
                position_list.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'buy' if position.type == mt5.POSITION_TYPE_BUY else 'sell',
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'stop_loss': position.sl,
                    'take_profit': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'open_time': datetime.fromtimestamp(position.time).strftime('%Y-%m-%d %H:%M:%S'),
                    'magic': position.magic
                })
                
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
            
    def get_order_history(self, days=7):
        """
        Get order history for the specified number of days
        
        Args:
            days (int): Number of days to get history for
            
        Returns:
            list: List of historical orders
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot get order history")
                return []
                
            # Set time range
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            # Get history
            history = mt5.history_orders_get(from_date, to_date)
            
            if history is None:
                self.logger.error(f"Failed to get order history: {mt5.last_error()}")
                return []
                
            # Convert to list of dictionaries
            order_list = []
            
            for order in history:
                order_list.append({
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': self._get_order_type_text(order.type),
                    'state': self._get_order_state_text(order.state),
                    'volume': order.volume_initial,
                    'price': order.price_open,
                    'stop_loss': order.sl,
                    'take_profit': order.tp,
                    'time_setup': datetime.fromtimestamp(order.time_setup).strftime('%Y-%m-%d %H:%M:%S'),
                    'time_done': datetime.fromtimestamp(order.time_done).strftime('%Y-%m-%d %H:%M:%S') if order.time_done > 0 else None,
                    'magic': order.magic,
                    'comment': order.comment
                })
                
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return []
            
    def get_deal_history(self, days=7):
        """
        Get deal history for the specified number of days
        
        Args:
            days (int): Number of days to get history for
            
        Returns:
            list: List of historical deals
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot get deal history")
                return []
                
            # Set time range
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            # Get history
            history = mt5.history_deals_get(from_date, to_date)
            
            if history is None:
                self.logger.error(f"Failed to get deal history: {mt5.last_error()}")
                return []
                
            # Convert to list of dictionaries
            deal_list = []
            
            for deal in history:
                deal_list.append({
                    'ticket': deal.ticket,
                    'symbol': deal.symbol,
                    'type': self._get_deal_type_text(deal.type),
                    'entry': self._get_deal_entry_text(deal.entry),
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'swap': deal.swap,
                    'commission': deal.commission,
                    'time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                    'order': deal.order,
                    'magic': deal.magic,
                    'comment': deal.comment
                })
                
            return deal_list
            
        except Exception as e:
            self.logger.error(f"Error getting deal history: {e}")
            return []
            
    def calculate_pip_value(self, symbol, volume=0.01):
        """
        Calculate the value of 1 pip for the given symbol and volume
        
        Args:
            symbol (str): Symbol to calculate for
            volume (float): Volume in lots
            
        Returns:
            float: Value of 1 pip in account currency
        """
        try:
            # Verify MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 not connected, cannot calculate pip value")
                return 0
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return 0
                
            # Calculate pip value
            contract_size = symbol_info.trade_contract_size
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            
            # For most forex pairs, 1 pip = 0.0001, but for JPY pairs 1 pip = 0.01
            pip_size = 0.0001
            if 'JPY' in symbol:
                pip_size = 0.01
                
            # Calculate pip value
            pip_value = (pip_size / tick_size) * tick_value * volume
            
            return pip_value
            
        except Exception as e:
            self.logger.error(f"Error calculating pip value for {symbol}: {e}")
            return 0
            
    def _get_retcode_text(self, retcode):
        """
        Get text description of a return code
        
        Args:
            retcode (int): Return code
            
        Returns:
            str: Text description
        """
        retcodes = {
            mt5.TRADE_RETCODE_DONE: "Done",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Done partially",
            mt5.TRADE_RETCODE_ERROR: "Error",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_NO_MONEY: "Not enough money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timed out",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Limit orders reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Position already closed",
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
        }
        
        return retcodes.get(retcode, f"Unknown code: {retcode}")
    
    def _get_order_type_text(self, order_type):
        """
        Get text description of an order type
        
        Args:
            order_type (int): Order type
            
        Returns:
            str: Text description
        """
        order_types = {
            mt5.ORDER_TYPE_BUY: "Buy",
            mt5.ORDER_TYPE_SELL: "Sell",
            mt5.ORDER_TYPE_BUY_LIMIT: "Buy Limit",
            mt5.ORDER_TYPE_SELL_LIMIT: "Sell Limit",
            mt5.ORDER_TYPE_BUY_STOP: "Buy Stop",
            mt5.ORDER_TYPE_SELL_STOP: "Sell Stop",
        }
        
        return order_types.get(order_type, f"Unknown type: {order_type}")
    
    def _get_order_state_text(self, order_state):
        """
        Get text description of an order state
        
        Args:
            order_state (int): Order state
            
        Returns:
            str: Text description
        """
        order_states = {
            mt5.ORDER_STATE_STARTED: "Started",
            mt5.ORDER_STATE_PLACED: "Placed",
            mt5.ORDER_STATE_CANCELED: "Canceled",
            mt5.ORDER_STATE_PARTIAL: "Partial",
            mt5.ORDER_STATE_FILLED: "Filled",
            mt5.ORDER_STATE_REJECTED: "Rejected",
            mt5.ORDER_STATE_EXPIRED: "Expired",
        }
        
        return order_states.get(order_state, f"Unknown state: {order_state}")
    
    def _get_deal_type_text(self, deal_type):
        """
        Get text description of a deal type
        
        Args:
            deal_type (int): Deal type
            
        Returns:
            str: Text description
        """
        deal_types = {
            mt5.DEAL_TYPE_BUY: "Buy",
            mt5.DEAL_TYPE_SELL: "Sell",
            mt5.DEAL_TYPE_BALANCE: "Balance",
            mt5.DEAL_TYPE_CREDIT: "Credit",
            mt5.DEAL_TYPE_CHARGE: "Charge",
            mt5.DEAL_TYPE_BONUS: "Bonus",
            mt5.DEAL_TYPE_COMMISSION: "Commission",
        }
        
        return deal_types.get(deal_type, f"Unknown type: {deal_type}")
    
    def _get_deal_entry_text(self, deal_entry):
        """
        Get text description of a deal entry
        
        Args:
            deal_entry (int): Deal entry
            
        Returns:
            str: Text description
        """
        deal_entries = {
            mt5.DEAL_ENTRY_IN: "Entry",
            mt5.DEAL_ENTRY_OUT: "Exit",
            mt5.DEAL_ENTRY_INOUT: "Reverse",
            mt5.DEAL_ENTRY_OUT_BY: "Close by",
        }
        
        return deal_entries.get(deal_entry, f"Unknown entry: {deal_entry}")