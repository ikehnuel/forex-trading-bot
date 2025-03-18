import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.strategies import BaseStrategy, StrategyFactory
from analysis.market_regime import MarketRegimeDetector
from performance.performance_analyzer import PerformanceAnalyzer

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data
    """
    def __init__(self, data_dir="./data/backtest"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        self.market_regime_detector = MarketRegimeDetector()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    initial_capital: float = 10000, position_size: float = 0.01, 
                    commission: float = 0.0, spread: float = 0.0002) -> Dict[str, Any]:
        """
        Run a backtest using the provided strategy and data
        
        Args:
            strategy: Strategy instance to test
            data: DataFrame with OHLC and indicator data
            initial_capital: Initial account balance
            position_size: Default position size in lots
            commission: Commission per trade in dollars
            spread: Spread in price units
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest with {strategy.name} strategy on {len(data)} bars")
        
        # Reset backtest state
        self.trades = []
        self.equity_curve = []
        
        # Make sure data is sorted by time
        if 'time' in data.columns:
            data = data.sort_values('time').reset_index(drop=True)
        
        # Prepare data and ensure all necessary columns
        backtest_data = self._prepare_data(data)
        
        # Initialize backtest state
        balance = initial_capital
        equity = initial_capital
        open_positions = []
        trade_id = 1
        
        # Set up initial equity curve point
        self.equity_curve.append({
            'time': backtest_data.iloc[0]['time'] if 'time' in backtest_data.columns else 0,
            'balance': balance,
            'equity': equity,
            'drawdown': 0,
            'positions': 0
        })
        
        # Calculate pip value (assuming standard forex lot size of 100,000)
        standard_lot_size = 100000
        contract_size = standard_lot_size * position_size
        
        # For JPY pairs, pip value is different
        point_value = 0.0001  # Standard forex point value
        pair_is_jpy = False
        
        if 'symbol' in backtest_data.columns:
            symbol = backtest_data.iloc[0]['symbol']
            pair_is_jpy = 'JPY' in symbol
            if pair_is_jpy:
                point_value = 0.01
        
        # Run the backtest
        for i in range(100, len(backtest_data)):  # Start after warmup period for indicators
            current_bar = backtest_data.iloc[i]
            next_bar = backtest_data.iloc[i+1] if i+1 < len(backtest_data) else None
            
            if next_bar is None:
                break
            
            # Update open positions
            for pos in list(open_positions):
                # Check if stop loss or take profit hit
                if pos['type'] == 'buy':
                    # Check if stop loss hit
                    if next_bar['low'] <= pos['stop_loss']:
                        # Close at stop loss
                        profit_pips = (pos['stop_loss'] - pos['entry_price']) / point_value
                        profit_amount = profit_pips * contract_size * point_value - commission
                        
                        # Update balance
                        balance += profit_amount
                        
                        # Record trade
                        self._record_trade(pos, 'stop_loss', pos['stop_loss'], 
                                         profit_pips, profit_amount, next_bar)
                        
                        # Remove from open positions
                        open_positions.remove(pos)
                        
                    # Check if take profit hit
                    elif next_bar['high'] >= pos['take_profit']:
                        # Close at take profit
                        profit_pips = (pos['take_profit'] - pos['entry_price']) / point_value
                        profit_amount = profit_pips * contract_size * point_value - commission
                        
                        # Update balance
                        balance += profit_amount
                        
                        # Record trade
                        self._record_trade(pos, 'take_profit', pos['take_profit'], 
                                         profit_pips, profit_amount, next_bar)
                        
                        # Remove from open positions
                        open_positions.remove(pos)
                        
                elif pos['type'] == 'sell':
                    # Check if stop loss hit
                    if next_bar['high'] >= pos['stop_loss']:
                        # Close at stop loss
                        profit_pips = (pos['entry_price'] - pos['stop_loss']) / point_value
                        profit_amount = profit_pips * contract_size * point_value - commission
                        
                        # Update balance
                        balance += profit_amount
                        
                        # Record trade
                        self._record_trade(pos, 'stop_loss', pos['stop_loss'], 
                                         profit_pips, profit_amount, next_bar)
                        
                        # Remove from open positions
                        open_positions.remove(pos)
                        
                    # Check if take profit hit
                    elif next_bar['low'] <= pos['take_profit']:
                        # Close at take profit
                        profit_pips = (pos['entry_price'] - pos['take_profit']) / point_value
                        profit_amount = profit_pips * contract_size * point_value - commission
                        
                        # Update balance
                        balance += profit_amount
                        
                        # Record trade
                        self._record_trade(pos, 'take_profit', pos['take_profit'], 
                                         profit_pips, profit_amount, next_bar)
                        
                        # Remove from open positions
                        open_positions.remove(pos)
            
            # Detect market regime
            look_back = 50
            if i >= look_back:
                regime_data = backtest_data.iloc[i-look_back:i+1]
                regime_info = self.market_regime_detector.detect_regime(regime_data)
            else:
                regime_info = {
                    'regime': 'unknown',
                    'trend_strength': 0,
                    'volatility': 'normal',
                    'range_bound': False
                }
            
            # Prepare market context
            market_context = {
                'market_regime': regime_info['regime'],
                'trend_strength': regime_info['trend_strength'],
                'volatility': regime_info['volatility'],
                'range_bound': regime_info['range_bound'],
                'is_breakout': regime_info.get('is_breakout', False),
                'is_breakout_up': regime_info.get('is_breakout_up', False),
                'is_breakout_down': regime_info.get('is_breakout_down', False)
            }
            
            # Generate trading signal
            lookback_window = 100
            analysis_data = backtest_data.iloc[max(0, i-lookback_window):i+1]
            
            signal = strategy.analyze(analysis_data, market_context)
            
            # Check for viable trading signal
            if signal['signal'] != 'NEUTRAL' and signal['confidence'] >= 0.5:
                if signal['entry'] is None or signal['stop_loss'] is None or signal['take_profit'] is None:
                    # Skip if missing key parameters
                    continue
                    
                # Create position with next bar open price + spread for buys / - spread for sells
                entry_price = next_bar['open']
                if signal['signal'] == 'BUY':
                    entry_price += spread
                elif signal['signal'] == 'SELL':
                    entry_price -= spread
                
                # Create position
                position = {
                    'id': trade_id,
                    'type': signal['signal'].lower(),
                    'entry_time': next_bar['time'] if 'time' in next_bar else i+1,
                    'entry_price': entry_price,
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'size': position_size,
                    'strategy': strategy.name,
                    'regime': regime_info['regime']
                }
                
                # Add to open positions
                open_positions.append(position)
                
                # Increment trade counter
                trade_id += 1
            
            # Calculate current equity
            unrealized_profit = 0
            for pos in open_positions:
                if pos['type'] == 'buy':
                    unrealized_profit += (current_bar['close'] - pos['entry_price']) / point_value * contract_size * point_value
                elif pos['type'] == 'sell':
                    unrealized_profit += (pos['entry_price'] - current_bar['close']) / point_value * contract_size * point_value
            
            equity = balance + unrealized_profit
            
            # Calculate drawdown
            peak_equity = max([point['equity'] for point in self.equity_curve])
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            
            # Record equity curve point
            self.equity_curve.append({
                'time': current_bar['time'] if 'time' in current_bar else i,
                'balance': balance,
                'equity': equity,
                'drawdown': drawdown,
                'positions': len(open_positions)
            })
        
        # Close any remaining open positions at the last bar
        if open_positions and len(backtest_data) > 0:
            last_bar = backtest_data.iloc[-1]
            for pos in list(open_positions):
                # Calculate profit based on last bar close
                if pos['type'] == 'buy':
                    profit_pips = (last_bar['close'] - pos['entry_price']) / point_value
                else:
                    profit_pips = (pos['entry_price'] - last_bar['close']) / point_value
                    
                profit_amount = profit_pips * contract_size * point_value - commission
                
                # Update balance
                balance += profit_amount
                
                # Record trade
                self._record_trade(pos, 'close', last_bar['close'], 
                                 profit_pips, profit_amount, last_bar)
                
                # Remove from open positions
                open_positions.remove(pos)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        # Create summary
        summary = {
            'strategy': strategy.name,
            'total_bars': len(backtest_data),
            'total_trades': len(self.trades),
            'initial_capital': initial_capital,
            'final_balance': balance,
            'net_profit': balance - initial_capital,
            'return_pct': (balance / initial_capital - 1) * 100,
            'performance_metrics': self.performance_metrics,
            'parameters': strategy.parameters
        }
        
        self.logger.info(f"Backtest completed: {summary['total_trades']} trades, net profit: {summary['net_profit']:.2f}, return: {summary['return_pct']:.2f}%")
        
        return summary
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting by ensuring required columns
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Processed DataFrame ready for backtesting
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Make sure we have required OHLC columns
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add a time column if not present
        if 'time' not in df.columns:
            df['time'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
        
        # Add a symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = 'EURUSD'  # Default symbol
        
        # Make sure all required columns are numeric
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _record_trade(self, position: Dict[str, Any], exit_reason: str, exit_price: float, 
                    profit_pips: float, profit_amount: float, exit_bar: pd.Series) -> None:
        """
        Record a completed trade
        
        Args:
            position: Position information
            exit_reason: Reason for exit
            exit_price: Exit price
            profit_pips: Profit in pips
            profit_amount: Profit in account currency
            exit_bar: Bar data at exit
        """
        # Create trade record
        trade = {
            'id': position['id'],
            'type': position['type'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': exit_bar['time'] if 'time' in exit_bar else None,
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'size': position['size'],
            'profit_pips': profit_pips,
            'profit_amount': profit_amount,
            'exit_reason': exit_reason,
            'strategy': position['strategy'],
            'regime': position['regime']
        }
        
        # Add to trades list
        self.trades.append(trade)
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades and equity curve
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_pips'] > 0]
        losing_trades = [t for t in self.trades if t['profit_pips'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_pips = sum(t['profit_pips'] for t in self.trades)
        total_profit = sum(t['profit_amount'] for t in self.trades)
        
        avg_win_pips = sum(t['profit_pips'] for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss_pips = sum(t['profit_pips'] for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        # Risk metrics
        profit_factor = abs(sum(t['profit_pips'] for t in winning_trades) / 
                          sum(t['profit_pips'] for t in losing_trades)) if sum(t['profit_pips'] for t in losing_trades) != 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win_pips) - ((1 - win_rate) * abs(avg_loss_pips))
        
        # Calculate max drawdown
        max_drawdown = max([point['drawdown'] for point in self.equity_curve]) if self.equity_curve else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            # Get equity changes
            equity_series = [point['equity'] for point in self.equity_curve]
            returns = [(equity_series[i] / equity_series[i-1] - 1) for i in range(1, len(equity_series))]
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Strategy analysis
        strategy_performance = {}
        for trade in self.trades:
            strategy = trade['strategy']
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pips': 0,
                    'win_rate': 0
                }
            
            strategy_performance[strategy]['trades'] += 1
            if trade['profit_pips'] > 0:
                strategy_performance[strategy]['wins'] += 1
            else:
                strategy_performance[strategy]['losses'] += 1
                
            strategy_performance[strategy]['total_pips'] += trade['profit_pips']
        
        # Calculate win rates
        for strategy, perf in strategy_performance.items():
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
        
        # Market regime analysis
        regime_performance = {}
        for trade in self.trades:
            regime = trade['regime']
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pips': 0,
                    'win_rate': 0
                }
            
            regime_performance[regime]['trades'] += 1
            if trade['profit_pips'] > 0:
                regime_performance[regime]['wins'] += 1
            else:
                regime_performance[regime]['losses'] += 1
                
            regime_performance[regime]['total_pips'] += trade['profit_pips']
        
        # Calculate win rates
        for regime, perf in regime_performance.items():
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
        
        # Full metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'total_profit': total_profit,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'avg_pips_per_trade': total_pips / total_trades if total_trades > 0 else 0,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'strategy_performance': strategy_performance,
            'regime_performance': regime_performance
        }
        
        return metrics
    
    def save_results(self, filename: str) -> None:
        """
        Save backtest results to file
        
        Args:
            filename: Output filename
        """
        if not self.trades:
            self.logger.warning("No trades to save")
            return
        
        # Create output path
        output_path = os.path.join(self.data_dir, filename)
        
        # Prepare data to save
        save_data = {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=self._json_serializer)
            
        self.logger.info(f"Backtest results saved to {output_path}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load backtest results from file
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary with loaded results
        """
        # Create input path
        input_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(input_path):
            self.logger.error(f"File not found: {input_path}")
            return {}
        
        try:
            # Load from JSON
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            # Update instance variables
            self.trades = data.get('trades', [])
            self.equity_curve = data.get('equity_curve', [])
            self.performance_metrics = data.get('performance_metrics', {})
            
            self.logger.info(f"Loaded backtest results from {input_path}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading backtest results: {e}")
            return {}
    
    def plot_equity_curve(self, show_drawdown=True, save_path=None) -> None:
        """
        Plot equity curve from backtest
        
        Args:
            show_drawdown: Whether to include drawdown in the plot
            save_path: Path to save the plot, or None to display
        """
        if not self.equity_curve:
            self.logger.warning("No equity curve to plot")
            return
        
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            
            # Create figure with two subplots if showing drawdown
            if show_drawdown:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot equity curve
            ax1.plot(equity_df['time'], equity_df['equity'], label='Equity', color='blue')
            ax1.plot(equity_df['time'], equity_df['balance'], label='Balance', color='green', alpha=0.7)
            
            ax1.set_title('Backtest Equity Curve')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Account Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add drawdown subplot if requested
            if show_drawdown:
                ax2.fill_between(equity_df['time'], 0, equity_df['drawdown'] * 100, color='red', alpha=0.3)
                ax2.set_title('Drawdown %')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Drawdown %')
                ax2.grid(True, alpha=0.3)
                
                # Set y-axis to start from 0
                ax2.set_ylim(bottom=0)
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Equity curve plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
    
    def plot_trade_analysis(self, save_path=None) -> None:
        """
        Plot trade analysis charts
        
        Args:
            save_path: Path to save the plot, or None to display
        """
        if not self.trades:
            self.logger.warning("No trades to analyze")
            return
        
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Profit distribution
            sns.histplot(trades_df['profit_pips'], kde=True, ax=axes[0, 0], color='blue')
            axes[0, 0].set_title('Profit Distribution (Pips)')
            axes[0, 0].set_xlabel('Profit (Pips)')
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            
            # Plot 2: Cumulative profit
            trades_df['cumulative_profit'] = trades_df['profit_amount'].cumsum()
            axes[0, 1].plot(trades_df.index, trades_df['cumulative_profit'], color='green')
            axes[0, 1].set_title('Cumulative Profit')
            axes[0, 1].set_xlabel('Trade #')
            axes[0, 1].set_ylabel('Profit')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Win rate by regime
            if 'regime' in trades_df.columns:
                regime_stats = trades_df.groupby('regime').agg({
                    'profit_pips': ['count', 'mean', 'sum'],
                    'id': lambda x: (x > 0).mean()  # Approximation of win rate
                })
                
                # Flatten multi-index
                regime_stats.columns = ['count', 'avg_pips', 'total_pips', 'win_rate']
                
                # Calculate win rate properly
                for regime in regime_stats.index:
                    regime_trades = trades_df[trades_df['regime'] == regime]
                    regime_stats.loc[regime, 'win_rate'] = len(regime_trades[regime_trades['profit_pips'] > 0]) / len(regime_trades)
                
                # Plot
                regime_stats['win_rate'].plot(kind='bar', ax=axes[1, 0], color='purple')
                axes[1, 0].set_title('Win Rate by Market Regime')
                axes[1, 0].set_xlabel('Market Regime')
                axes[1, 0].set_ylabel('Win Rate')
                axes[1, 0].set_ylim(0, 1)
                
                # Add value labels
                for i, v in enumerate(regime_stats['win_rate']):
                    axes[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
            # Plot 4: Average profit by exit reason
            exit_stats = trades_df.groupby('exit_reason').agg({
                'profit_pips': ['count', 'mean', 'sum']
            })
            
            # Flatten multi-index
            exit_stats.columns = ['count', 'avg_pips', 'total_pips']
            
            # Plot
            exit_stats['avg_pips'].plot(kind='bar', ax=axes[1, 1], color='orange')
            axes[1, 1].set_title('Average Profit by Exit Reason')
            axes[1, 1].set_xlabel('Exit Reason')
            axes[1, 1].set_ylabel('Average Profit (Pips)')
            
            # Add value labels
            for i, v in enumerate(exit_stats['avg_pips']):
                axes[1, 1].text(i, v + (1 if v >= 0 else -1), f"{v:.1f}", ha='center')
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Trade analysis plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting trade analysis: {e}")
    
    def optimize_strategy(self, strategy_class, data: pd.DataFrame, 
                         param_grid: Dict[str, List], 
                         initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_class: Strategy class to optimize
            data: DataFrame with OHLC and indicator data
            param_grid: Dictionary with parameter names and lists of values to test
            initial_capital: Initial account balance
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting parameter optimization for {strategy_class.__name__}")
        
        # Prepare data for backtesting
        backtest_data = self._prepare_data(data)
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run backtests with each parameter combination
        results = []
        
        for params in param_combinations:
            # Create strategy instance with these parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            result = self.run_backtest(strategy, backtest_data, initial_capital=initial_capital)
            
            # Save parameters and key metrics
            optimization_result = {
                'parameters': params,
                'net_profit': result['net_profit'],
                'return_pct': result['return_pct'],
                'total_trades': result['total_trades'],
                'win_rate': result['performance_metrics'].get('win_rate', 0),
                'profit_factor': result['performance_metrics'].get('profit_factor', 0),
                'max_drawdown': result['performance_metrics'].get('max_drawdown', 0),
                'sharpe_ratio': result['performance_metrics'].get('sharpe_ratio', 0)
            }
            
            results.append(optimization_result)
        
        # Sort results by net profit
        results.sort(key=lambda x: x['net_profit'], reverse=True)
        
        # Return best results
        best_result = results[0] if results else {}
        
        # Create summary
        summary = {
            'strategy': strategy_class.__name__,
            'total_combinations': len(param_combinations),
            'best_parameters': best_result.get('parameters', {}),
            'best_metrics': {
                'net_profit': best_result.get('net_profit', 0),
                'return_pct': best_result.get('return_pct', 0),
                'total_trades': best_result.get('total_trades', 0),
                'win_rate': best_result.get('win_rate', 0),
                'profit_factor': best_result.get('profit_factor', 0),
                'max_drawdown': best_result.get('max_drawdown', 0),
                'sharpe_ratio': best_result.get('sharpe_ratio', 0)
            },
            'all_results': results
        }
        
        self.logger.info(f"Optimization completed. Best return: {best_result.get('return_pct', 0):.2f}%")
        
        return summary
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters from a parameter grid
        
        Args:
            param_grid: Dictionary with parameter names and lists of values
            
        Returns:
            List of parameter dictionaries
        """
        # Base case: empty grid
        if not param_grid:
            return [{}]
        
        # Get all keys
        keys = list(param_grid.keys())
        
        # Generate combinations recursively
        if len(keys) == 1:
            # Base case: one parameter
            key = keys[0]
            return [{key: value} for value in param_grid[key]]
        else:
            # Recursive case: multiple parameters
            key = keys[0]
            rest_grid = {k: param_grid[k] for k in keys[1:]}
            
            # Generate combinations for the rest of the parameters
            rest_combinations = self._generate_param_combinations(rest_grid)
            
            # Combine with current parameter
            combinations = []
            for value in param_grid[key]:
                for rest_combo in rest_combinations:
                    combo = {key: value, **rest_combo}
                    combinations.append(combo)
            
            return combinations
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")


# src/backtesting/backtest_runner.py
import argparse
import pandas as pd
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_logging
from mt5_connector.data_collector import DataCollector
from mt5_connector.connection import MT5Connector
from analysis.strategies import StrategyFactory, TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy
from backtesting.backtest_engine import BacktestEngine

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Symbol to backtest')
    
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Timeframe to backtest')
    
    parser.add_argument('--strategy', type=str, default='trend_following',
                        choices=['trend_following', 'mean_reversion', 'breakout', 'all'],
                        help='Strategy to backtest')
    
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtest')
    
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size in lots')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization')
    
    parser.add_argument('--save-results', action='store_true',
                        help='Save backtest results')
    
    parser.add_argument('--plot', action='store_true',
                        help='Plot backtest results')
    
    return parser.parse_args()

def get_strategy(strategy_name, **kwargs):
    """Get strategy instance based on name"""
    return StrategyFactory.create_strategy(strategy_name, **kwargs)

def load_data(symbol, timeframe, start_date, end_date=None):
    """Load historical data for backtesting"""
    logger = logging.getLogger(__name__)
    
    try:
        # Try to connect to MT5 for data
        mt5 = MT5Connector()
        if mt5.check_connection():
            data_collector = DataCollector(mt5)
            
            # Convert dates to datetime
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
            
            # Calculate number of bars needed (approximate)
            days_diff = (end_datetime - start_datetime).days
            bars_needed = days_diff * 24 if timeframe == 'H1' else days_diff
            
            # Get data
            data = data_collector.get_ohlc_data(symbol, timeframe, bars_needed, True)
            
            if data is not None and not data.empty:
                # Filter by date range
                data = data[(data['time'] >= start_datetime) & (data['time'] <= end_datetime)]
                
                # Add symbol column
                data['symbol'] = symbol
                
                logger.info(f"Loaded {len(data)} bars from MT5 for {symbol} {timeframe}")
                
                # Disconnect from MT5
                mt5.disconnect()
                
                return data
    except Exception as e:
        logger.error(f"Error loading data from MT5: {e}")
    
    # Fallback: Try to load from CSV
    try:
        # Try to find CSV in data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data')
        
        # Look for matching files
        for filename in os.listdir(data_dir):
            if symbol in filename and timeframe in filename and filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                
                # Load CSV
                data = pd.read_csv(filepath)
                
                # Convert time to datetime
                if 'time' in data.columns:
                    data['time'] = pd.to_datetime(data['time'])
                    
                    # Filter by date range
                    start_datetime = pd.to_datetime(start_date)
                    end_datetime = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
                    
                    data = data[(data['time'] >= start_datetime) & (data['time'] <= end_datetime)]
                
                # Add symbol column if not present
                if 'symbol' not in data.columns:
                    data['symbol'] = symbol
                
                logger.info(f"Loaded {len(data)} bars from CSV for {symbol} {timeframe}")
                
                return data
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
    
    # If all else fails, generate random data for testing
    logger.warning("Using randomly generated data for testing")
    
    # Generate random data
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
    
    days_diff = (end_datetime - start_datetime).days
    bars_count = days_diff * 24 if timeframe == 'H1' else days_diff
    
    import numpy as np
    
    # Create DataFrame with random data
    data = pd.DataFrame({
        'time': pd.date_range(start=start_date, periods=bars_count, freq='H' if timeframe == 'H1' else 'D'),
        'open': np.random.normal(1.2000, 0.0010, bars_count),
        'high': np.random.normal(1.2010, 0.0010, bars_count),
        'low': np.random.normal(1.1990, 0.0010, bars_count),
        'close': np.random.normal(1.2005, 0.0010, bars_count),
        'volume': np.random.randint(100, 1000, bars_count),
        'symbol': symbol
    })
    
    # Make sure high is highest and low is lowest
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])
    
    logger.info(f"Generated {len(data)} random bars for testing")
    
    return data

def run_backtest(args):
    """Run backtest with specified parameters"""
    logger = logging.getLogger(__name__)
    
    # Load data
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data is None or data.empty:
        logger.error("No data available for backtesting")
        return
    
    # Create backtest engine
    engine = BacktestEngine()
    
    # Run backtest for each strategy
    if args.strategy == 'all':
        strategies = ['trend_following', 'mean_reversion', 'breakout']
    else:
        strategies = [args.strategy]
    
    for strategy_name in strategies:
        logger.info(f"Running backtest for {strategy_name} strategy")
        
        # Create strategy
        strategy = get_strategy(strategy_name)
        
        if args.optimize:
            # Define parameter grid for optimization
            if strategy_name == 'trend_following':
                param_grid = {
                    'ma_fast': [10, 20, 30],
                    'ma_slow': [40, 50, 60],
                    'atr_multiplier': [1.5, 2.0, 2.5],
                    'profit_target_multiplier': [1.5, 2.0, 2.5]
                }
                strategy_class = TrendFollowingStrategy
                
            elif strategy_name == 'mean_reversion':
                param_grid = {
                    'rsi_period': [9, 14, 21],
                    'rsi_oversold': [20, 30],
                    'rsi_overbought': [70, 80],
                    'atr_multiplier': [1.0, 1.5, 2.0]
                }
                strategy_class = MeanReversionStrategy
                
            elif strategy_name == 'breakout':
                param_grid = {
                    'lookback_period': [10, 20, 30],
                    'atr_multiplier': [1.0, 1.5, 2.0],
                    'profit_target_multiplier': [1.5, 2.0, 2.5]
                }
                strategy_class = BreakoutStrategy
            
            # Run optimization
            optimization_results = engine.optimize_strategy(
                strategy_class=strategy_class,
                data=data,
                param_grid=param_grid,
                initial_capital=args.initial_capital
            )
            
            # Print optimization results
            logger.info(f"Optimization results for {strategy_name}:")
            logger.info(f"Best parameters: {optimization_results['best_parameters']}")
            logger.info(f"Best return: {optimization_results['best_metrics']['return_pct']:.2f}%")
            
            # Create strategy with optimized parameters
            strategy = get_strategy(strategy_name, **optimization_results['best_parameters'])
            
            # Save optimization results
            if args.save_results:
                # Create filename
                filename = f"{args.symbol}_{args.timeframe}_{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d')}.json"
                
                # Create output directory
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'backtest')
                os.makedirs(output_dir, exist_ok=True)
                
                # Save optimization results
                import json
                with open(os.path.join(output_dir, filename), 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                
                logger.info(f"Optimization results saved to {filename}")
        
        # Run backtest
        backtest_results = engine.run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=args.initial_capital,
            position_size=args.position_size
        )
        
        # Print backtest results
        logger.info(f"Backtest results for {strategy_name}:")
        logger.info(f"Net profit: {backtest_results['net_profit']:.2f}")
        logger.info(f"Return: {backtest_results['return_pct']:.2f}%")
        logger.info(f"Total trades: {backtest_results['total_trades']}")
        logger.info(f"Win rate: {backtest_results['performance_metrics'].get('win_rate', 0):.2f}")
        
        # Save backtest results
        if args.save_results:
            # Create filename
            filename = f"{args.symbol}_{args.timeframe}_{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Save results
            engine.save_results(filename)
        
        # Plot results
        if args.plot:
            # Plot equity curve
            engine.plot_equity_curve()
            
            # Plot trade analysis
            engine.plot_trade_analysis()

if __name__ == '__main__':
    # Configure logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Run backtest
    run_backtest(args)