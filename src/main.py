import os
import time
import logging
import signal
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import configuration
from config import (
    setup_logging, 
    logger, 
    DEFAULT_SYMBOL, 
    DEFAULT_TIMEFRAME, 
    CHECK_INTERVAL,
    MAX_RISK_PER_TRADE,
    MAX_DAILY_RISK
)

# Import modules
from mt5_connector.connection import MT5Connector
from mt5_connector.data_collector import DataCollector
from mt5_connector.order_executor import OrderExecutor
from analysis.market_regime import MarketRegimeDetector
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from analysis.claude_analyzer import ClaudeAnalyzer
from trade_management.risk_manager import RiskManager
from trade_management.trade_manager import TradeManager
from performance.performance_analyzer import PerformanceAnalyzer, StrategyOptimizer

class ForexTradingBot:
    """
    Main forex trading bot application that orchestrates all components.
    """
    def __init__(self, symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME):
        """
        Initialize the forex trading bot
        
        Args:
            symbol: Symbol to trade
            timeframe: Timeframe to analyze
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.last_check_time = None
        self.initialize_components()
        
    def initialize_components(self) -> bool:
        """
        Initialize all bot components
        
        Returns:
            bool: Success or failure
        """
        try:
            self.logger.info("Initializing forex trading bot components")
            
            # Initialize MT5 connection
            self.mt5 = MT5Connector()
            if not self.mt5.check_connection():
                self.logger.error("Failed to connect to MT5")
                return False
                
            # Initialize data collection
            self.data_collector = DataCollector(self.mt5)
            
            # Initialize order execution
            self.order_executor = OrderExecutor(self.mt5)
            
            # Initialize analysis components
            self.market_regime_detector = MarketRegimeDetector()
            self.mtf_analyzer = MultiTimeframeAnalyzer(self.data_collector)
            self.claude_analyzer = ClaudeAnalyzer()
            
            # Initialize trade management
            self.risk_manager = RiskManager(
                max_risk_per_trade=MAX_RISK_PER_TRADE,
                max_daily_risk=MAX_DAILY_RISK
            )
            self.trade_manager = TradeManager(self.order_executor)
            
            # Initialize performance analysis
            self.performance_analyzer = PerformanceAnalyzer()
            self.strategy_optimizer = StrategyOptimizer(self.performance_analyzer)
            
            # Load current account info
            self.account_info = self.mt5.get_account_info()
            if not self.account_info:
                self.logger.warning("Could not retrieve account information")
                
            # Check symbol info
            self.symbol_info = self.mt5.get_symbol_info(self.symbol)
            if not self.symbol_info:
                self.logger.error(f"Symbol {self.symbol} not found or not available")
                return False
                
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
        
    def start(self) -> None:
        """Start the trading bot"""
        self.logger.info(f"Starting forex trading bot for {self.symbol} on {self.timeframe} timeframe")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Main trading loop
            while self.running:
                try:
                    # Check if it's time to run analysis
                    current_time = datetime.now()
                    
                    if (self.last_check_time is None or 
                        (current_time - self.last_check_time).total_seconds() >= CHECK_INTERVAL):
                        
                        self.logger.info(f"Running analysis for {self.symbol} on {self.timeframe}")
                        self.last_check_time = current_time
                        
                        # Run the trading cycle
                        self.run_trading_cycle()
                        
                    # Sleep for a short time to prevent busy waiting
                    time.sleep(10)
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait a minute before retrying on error
                    
            self.logger.info("Trading bot stopped")
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping bot")
            self.stop()
        
    def stop(self) -> None:
        """Stop the trading bot gracefully"""
        self.logger.info("Stopping forex trading bot")
        self.running = False
        
        # Close MT5 connection
        if hasattr(self, 'mt5') and self.mt5:
            self.mt5.disconnect()
            
        self.logger.info("Bot shutdown complete")
        
    def signal_handler(self, sig, frame) -> None:
        """Handle termination signals"""
        self.logger.info(f"Received signal {sig}, shutting down")
        self.stop()
        
    def run_trading_cycle(self) -> None:
        """Execute a complete trading cycle"""
        try:
            # 1. Check MT5 connection
            if not self.mt5.check_connection():
                self.logger.error("MT5 connection lost, reconnecting")
                if not self.mt5.connect():
                    self.logger.error("Failed to reconnect to MT5, skipping cycle")
                    return
                    
            # 2. Update account and symbol info
            self.account_info = self.mt5.get_account_info()
            self.symbol_info = self.mt5.get_symbol_info(self.symbol)
            
            if not self.account_info or not self.symbol_info:
                self.logger.error("Failed to get account or symbol info, skipping cycle")
                return
                
            # 3. Check if market is open
            if not self.mt5.is_market_open(self.symbol):
                self.logger.info(f"Market is closed for {self.symbol}, skipping analysis")
                return
                
            # 4. Manage existing trades
            open_positions = self.order_executor.get_open_positions()
            self.trade_manager.update_trades(open_positions)
            
            # Apply trade management strategies
            modified_count = self.trade_manager.manage_all_trades()
            if modified_count > 0:
                self.logger.info(f"Modified {modified_count} trades")
                
            # 5. Collect market data for analysis
            self.logger.info(f"Collecting data for {self.symbol} on {self.timeframe}")
            
            # Get data for multiple timeframes
            timeframe_data = self.mtf_analyzer.get_data_for_timeframes(
                self.symbol, 
                self.timeframe, 
                num_candles=100
            )
            
            if not timeframe_data or self.timeframe not in timeframe_data:
                self.logger.error(f"Failed to get data for {self.symbol} on {self.timeframe}")
                return
                
            base_df = timeframe_data[self.timeframe]
            
            # 6. Detect market regime
            regime_info = self.market_regime_detector.detect_regime(base_df)
            self.logger.info(f"Detected market regime: {regime_info['regime']}")
            
            # 7. Multi-timeframe analysis
            mtf_analysis = self.mtf_analyzer.analyze_timeframes(timeframe_data)
            self.logger.info(f"Multi-timeframe analysis: trend strength = {mtf_analysis['trend_analysis']['trend_strength']:.2f}, alignment = {mtf_analysis['alignment_score']:.2f}")
            
            # 8. Update risk calculations
            if self.account_info:
                current_balance = self.account_info.get('balance', 0)
                self.risk_manager.update_drawdown(current_balance)
                
            # 9. Check if we can place a new trade
            can_trade, reason = self.risk_manager.can_place_trade(self.account_info, self.symbol_info)
            
            if not can_trade:
                self.logger.info(f"Trading restricted: {reason}")
                return
                
            # 10. Prepare data for Claude analysis
            # Extract the most recent data for Claude
            recent_df = base_df.tail(20).copy()
            
            # Convert DataFrame to dict for Claude
            ohlc_data = recent_df.to_dict('records')
            
            # Extract indicators
            indicators = {}
            for col in recent_df.columns:
                if col not in ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'spread']:
                    indicators[col] = recent_df[col].tolist()
                    
            # Prepare market context
            market_context = {
                'market_regime': regime_info['regime'],
                'volatility': regime_info['volatility'],
                'trend_strength': mtf_analysis['trend_analysis']['trend_strength'],
                'timeframe_alignment': mtf_analysis['alignment_score'],
                'account_balance': self.account_info.get('balance', 0),
                'open_positions': len(open_positions),
                'current_drawdown': self.risk_manager.current_drawdown,
                'support_levels': [level['level'] for level in mtf_analysis['support_resistance']['clustered_levels'] 
                                if level['type'] == 'support'][:3],
                'resistance_levels': [level['level'] for level in mtf_analysis['support_resistance']['clustered_levels']
                                    if level['type'] == 'resistance'][:3]
            }
            
            # 11. Send to Claude for analysis
            self.logger.info("Sending data to Claude for analysis")
            trading_advice = self.claude_analyzer.analyze_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                ohlc_data=ohlc_data,
                indicators=indicators,
                market_context=market_context,
                existing_positions=open_positions
            )
            
            # Log Claude's token usage
            usage_stats = self.claude_analyzer.get_usage_stats()
            self.logger.info(f"Claude API usage: {usage_stats['token_usage']['prompt_tokens']} prompt tokens, "
                           f"{usage_stats['token_usage']['completion_tokens']} completion tokens, "
                           f"estimated cost: ${usage_stats['token_usage']['estimated_cost']:.4f}")
            
            # 12. Process Claude's trading recommendation
            if 'error' in trading_advice:
                self.logger.error(f"Error in Claude analysis: {trading_advice['error']}")
                return
                
            trade_recommendation = trading_advice.get('trade_recommendation', {})
            action = trade_recommendation.get('action', 'HOLD')
            confidence = trade_recommendation.get('confidence', 0)
            
            self.logger.info(f"Claude's recommendation: {action} with {confidence:.0%} confidence")
            self.logger.info(f"Reasoning: {trade_recommendation.get('reasoning', 'No reasoning provided')}")
            
            # 13. Execute trading decision if confidence is high enough
            if action in ['BUY', 'SELL'] and confidence >= 0.7:
                self.execute_trade(trade_recommendation)
            elif action == 'CLOSE' and confidence >= 0.6:
                self.close_trades(trade_recommendation)
            else:
                self.logger.info(f"No action taken: {action} with {confidence:.0%} confidence is below threshold")
                
            # 14. Update performance metrics periodically
            if datetime.now().hour == 0 and datetime.now().minute < 10:  # Around midnight
                self.performance_analyzer.analyze_performance()
                recommendations = self.performance_analyzer.get_optimization_recommendations()
                
                for rec in recommendations:
                    self.logger.info(f"Performance recommendation: {rec['aspect']} - {rec['recommendation']}")
                    
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
        
    def execute_trade(self, trade_recommendation: Dict[str, Any]) -> None:
        """
        Execute a trade based on Claude's recommendation
        
        Args:
            trade_recommendation: Trade recommendation from Claude
        """
        try:
            action = trade_recommendation.get('action', 'HOLD')
            entry_price = trade_recommendation.get('entry_price')
            stop_loss = trade_recommendation.get('stop_loss')
            take_profit = trade_recommendation.get('take_profit')
            
            # Skip if any required value is missing
            if not entry_price or not stop_loss or not take_profit:
                self.logger.warning("Missing required trade parameters, cannot execute")
                return
                
            # If take_profit is a list, use the first value
            if isinstance(take_profit, list) and take_profit:
                take_profit = take_profit[0]
                
            # Calculate position size based on risk management
            position_size = self.risk_manager.calculate_position_size(
                account_info=self.account_info,
                entry_price=entry_price,
                stop_loss=stop_loss,
                symbol_info=self.symbol_info,
                atr_value=trade_recommendation.get('atr')
            )
            
            # Override with Claude's position size if provided and within limits
            if trade_recommendation.get('position_size'):
                claude_size = float(trade_recommendation.get('position_size'))
                min_volume = self.symbol_info.get('volume_min', 0.01)
                max_volume = self.symbol_info.get('volume_max', 100.0)
                
                # Use the smaller of Claude's size and our calculated size
                position_size = min(claude_size, position_size)
                
                # Ensure within limits
                position_size = max(min_volume, min(position_size, max_volume))
                
            # Calculate risk amount
            price_distance = abs(entry_price - stop_loss)
            pip_value = self.order_executor.calculate_pip_value(self.symbol, position_size)
            pips_risked = price_distance / self.symbol_info.get('point', 0.0001) * 10
            risk_amount = pips_risked * pip_value
            
            # Log trade details
            self.logger.info(f"Executing {action} order for {self.symbol}:")
            self.logger.info(f"Entry: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            self.logger.info(f"Position Size: {position_size} lots, Risk: {risk_amount:.2f}")
            
            # Execute the order
            result = self.order_executor.open_position(
                symbol=self.symbol,
                order_type=action.lower(),
                volume=position_size,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"Claude Bot {datetime.now().strftime('%Y%m%d%H%M')}"
            )
            
            if result and result.get('result') == 'success':
                self.logger.info(f"Order executed successfully: ticket={result.get('ticket')}")
                
                # Add to trade manager
                ticket = result.get('ticket')
                self.trade_manager.add_trade(
                    ticket=ticket,
                    symbol=self.symbol,
                    order_type=action.lower(),
                    volume=position_size,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_amount=risk_amount
                )
                
                # Record trade with risk manager
                self.risk_manager.record_trade({
                    'ticket': ticket,
                    'symbol': self.symbol,
                    'type': action.lower(),
                    'volume': position_size,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_amount': risk_amount
                })
            else:
                self.logger.error(f"Order execution failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
        
    def close_trades(self, trade_recommendation: Dict[str, Any]) -> None:
        """
        Close trades based on Claude's recommendation
        
        Args:
            trade_recommendation: Trade recommendation from Claude
        """
        try:
            close_tickets = trade_recommendation.get('close_tickets', [])
            
            if not close_tickets:
                # If no specific tickets provided, check if we should close all for this symbol
                if trade_recommendation.get('close_all', False):
                    # Get all open positions for this symbol
                    open_positions = self.order_executor.get_open_positions(self.symbol)
                    close_tickets = [pos.get('ticket') for pos in open_positions]
                    
            if not close_tickets:
                self.logger.info("No tickets specified for closing")
                return
                
            self.logger.info(f"Closing trades: {close_tickets}")
            
            # Close each trade
            for ticket in close_tickets:
                result = self.order_executor.close_position(
                    ticket=ticket,
                    comment=f"Claude Bot Close {datetime.now().strftime('%Y%m%d%H%M')}"
                )
                
                if result and result.get('result') == 'success':
                    self.logger.info(f"Closed position {ticket} successfully")
                else:
                    self.logger.error(f"Failed to close position {ticket}: {result}")
                    
        except Exception as e:
            self.logger.error(f"Error closing trades: {e}")


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging()
    
    # Default symbol and timeframe
    symbol = os.getenv('SYMBOL', DEFAULT_SYMBOL)
    timeframe = os.getenv('TIMEFRAME', DEFAULT_TIMEFRAME)
    
    # Allow command line arguments to override
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 2:
        timeframe = sys.argv[2]
        
    # Create and start the trading bot
    bot = ForexTradingBot(symbol, timeframe)
    
    if bot.initialize_components():
        bot.start()
    else:
        logger.error("Failed to initialize trading bot components")
        sys.exit(1)