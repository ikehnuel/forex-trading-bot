import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, logger

class PerformanceAnalyzer:
    """
    Analyzes trading performance, identifies strengths and weaknesses,
    and provides recommendations for strategy optimization.
    """
    def __init__(self, data_path=os.path.join(DATA_DIR, "trade_history")):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.trade_history = []
        self.performance_metrics = {}
        self.load_trade_history()
        
    def load_trade_history(self) -> None:
        """Load trade history from JSON files"""
        try:
            if not os.path.exists(self.data_path):
                self.logger.warning(f"Trade history directory not found: {self.data_path}")
                return
                
            # Find all history files
            history_files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
            
            if not history_files:
                self.logger.warning("No trade history files found")
                return
                
            # Sort by date
            history_files.sort(reverse=True)
            latest_file = os.path.join(self.data_path, history_files[0])
            
            # Load latest history
            with open(latest_file, 'r') as f:
                history_data = json.load(f)
                
            if isinstance(history_data, dict):
                self.trade_history = history_data.get('trade_history', [])
                
                # Also get active trades
                active_trades = history_data.get('active_trades', [])
                
                # Add active trades to history for analysis
                self.trade_history.extend(active_trades)
            else:
                self.trade_history = history_data
                
            self.logger.info(f"Loaded {len(self.trade_history)} trades from history")
            
        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
            self.trade_history = []
            
    def save_trade_history(self) -> None:
        """Save trade history to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.data_path, exist_ok=True)
            
            # Create filename with current date
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.data_path, filename)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=self._json_serializer)
                
            self.logger.info(f"Saved trade history to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")
            
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a completed trade to history
        
        Args:
            trade_data: Trade data dictionary
        """
        try:
            # Ensure trade has all required fields
            required_fields = ['ticket', 'symbol', 'type', 'entry_price', 'exit_price', 
                           'entry_time', 'exit_time', 'profit_pips', 'profit_amount', 
                           'volume', 'initial_stop_loss', 'initial_take_profit']
                           
            # Add missing fields with default values
            for field in required_fields:
                if field not in trade_data:
                    trade_data[field] = None
                    
            # Add trade to history
            self.trade_history.append(trade_data)
            self.save_trade_history()
            
        except Exception as e:
            self.logger.error(f"Error adding trade to history: {e}")
            
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.trade_history:
                return {}
                
            # Convert trade history to DataFrame for analysis
            df = pd.DataFrame(self.trade_history)
            
            # Ensure we have numeric profit values
            if 'profit_pips' in df.columns:
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
            else:
                df['profit_pips'] = 0
                
            if 'profit_amount' in df.columns:
                df['profit_amount'] = pd.to_numeric(df['profit_amount'], errors='coerce')
            else:
                df['profit_amount'] = 0
                
            # Basic metrics
            total_trades = len(df)
            winning_trades = df[df['profit_pips'] > 0]
            losing_trades = df[df['profit_pips'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            total_pips = df['profit_pips'].sum()
            total_profit = df['profit_amount'].sum()
            
            avg_win_pips = winning_trades['profit_pips'].mean() if win_count > 0 else 0
            avg_loss_pips = losing_trades['profit_pips'].mean() if loss_count > 0 else 0
            
            # Converting entry_time and exit_time to datetime if they're strings
            if 'entry_time' in df.columns and df['entry_time'].dtype == 'object':
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                
            if 'exit_time' in df.columns and df['exit_time'].dtype == 'object':
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                
            # Calculate trade duration
            if 'entry_time' in df.columns and 'exit_time' in df.columns:
                df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # in hours
                
                avg_duration = df['duration'].mean()
                avg_win_duration = winning_trades['duration'].mean() if win_count > 0 else 0
                avg_loss_duration = losing_trades['duration'].mean() if loss_count > 0 else 0
            else:
                avg_duration = 0
                avg_win_duration = 0
                avg_loss_duration = 0
                
            # Risk metrics
            if 'initial_stop_loss' in df.columns and 'entry_price' in df.columns and 'exit_price' in df.columns:
                df['risk_reward_ratio'] = df.apply(
                    lambda row: abs((row['exit_price'] - row['entry_price']) / 
                                  (row['entry_price'] - row['initial_stop_loss'])) 
                                  if row['initial_stop_loss'] is not None and row['entry_price'] != row['initial_stop_loss'] else 0, 
                    axis=1
                )
                
                avg_risk_reward = df['risk_reward_ratio'].mean()
            else:
                avg_risk_reward = 0
                
            # Profit factor
            gross_profit = winning_trades['profit_pips'].sum()
            gross_loss = abs(losing_trades['profit_pips'].sum()) if loss_count > 0 else 1  # Avoid division by zero
            profit_factor = gross_profit / gross_loss
            
            # Expectancy
            expectancy = (win_rate * avg_win_pips) - ((1 - win_rate) * abs(avg_loss_pips))
            
            # Sharpe ratio (simplified)
            if 'profit_amount' in df.columns and 'exit_time' in df.columns:
                daily_returns = self._calculate_daily_returns(df)
                if len(daily_returns) > 1:
                    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
                else:
                    sharpe = 0
            else:
                sharpe = 0
                
            # Maximum drawdown
            max_drawdown, drawdown_periods = self._calculate_drawdown(df)
            
            # Streak analysis
            max_win_streak, max_loss_streak, current_streak = self._analyze_streaks(df)
            
            # Time-based analysis
            if 'entry_time' in df.columns:
                hourly_performance = self._analyze_performance_by_hour(df)
                daily_performance = self._analyze_performance_by_day(df)
            else:
                hourly_performance = {}
                daily_performance = {}
                
            # Symbol performance
            if 'symbol' in df.columns:
                symbol_performance = self._analyze_performance_by_symbol(df)
            else:
                symbol_performance = {}
                
            # Store all metrics
            self.performance_metrics = {
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
                'avg_risk_reward': avg_risk_reward,
                'avg_duration': avg_duration,
                'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration,
                'max_drawdown': max_drawdown,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'current_streak': current_streak,
                'sharpe_ratio': sharpe,
                'hourly_performance': hourly_performance,
                'daily_performance': daily_performance,
                'symbol_performance': symbol_performance,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {}
            
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate recommendations to improve trading performance
        
        Returns:
            List of recommendation dictionaries
        """
        try:
            if not self.performance_metrics:
                self.analyze_performance()
                
            if not self.performance_metrics:
                return []
                
            recommendations = []
            
            # Check win rate
            if self.performance_metrics['win_rate'] < 0.4:
                recommendations.append({
                    'aspect': 'Entry Criteria',
                    'issue': 'Low win rate',
                    'recommendation': 'Strengthen entry filters. Consider waiting for confirmation signals across multiple timeframes.'
                })
                
            # Check profit factor
            if self.performance_metrics['profit_factor'] < 1.2:
                recommendations.append({
                    'aspect': 'Profit Management',
                    'issue': 'Low profit factor',
                    'recommendation': 'Increase profit targets or use trailing stops to capture larger moves in winning trades.'
                })
                
            # Check risk-reward ratio
            if self.performance_metrics['avg_risk_reward'] < 1.5:
                recommendations.append({
                    'aspect': 'Risk Management',
                    'issue': 'Suboptimal risk-reward ratio',
                    'recommendation': 'Aim for trades with at least 1:2 risk-reward ratio. Consider tighter stop losses or higher profit targets.'
                })
                
            # Check drawdown
            if self.performance_metrics['max_drawdown'] > 0.15:  # 15% drawdown
                recommendations.append({
                    'aspect': 'Capital Preservation',
                    'issue': 'High maximum drawdown',
                    'recommendation': 'Reduce position sizes or implement stricter risk management rules during losing streaks.'
                })
                
            # Check winning vs losing duration
            if (self.performance_metrics['avg_loss_duration'] > 
                self.performance_metrics['avg_win_duration'] * 1.5):
                recommendations.append({
                    'aspect': 'Trade Management',
                    'issue': 'Holding losing trades too long',
                    'recommendation': 'Cut losing trades faster. Implement time-based stops for trades that don\'t move in your favor quickly.'
                })
                
            # Check time-based performance
            hourly_perf = self.performance_metrics['hourly_performance']
            if hourly_perf:
                worst_hours = [hour for hour, metrics in hourly_perf.items() 
                              if metrics['win_rate'] < 0.3 and metrics['total_trades'] > 5]
                if worst_hours:
                    hour_list = ', '.join([f"{h}:00" for h in worst_hours])
                    recommendations.append({
                        'aspect': 'Trading Hours',
                        'issue': f'Poor performance during hours: {hour_list}',
                        'recommendation': 'Avoid trading during these hours or use more stringent criteria.'
                    })
                    
            # Check day-based performance
            daily_perf = self.performance_metrics['daily_performance']
            if daily_perf:
                worst_days = [day for day, metrics in daily_perf.items() 
                             if metrics['win_rate'] < 0.3 and metrics['total_trades'] > 5]
                if worst_days:
                    day_list = ', '.join(worst_days)
                    recommendations.append({
                        'aspect': 'Trading Days',
                        'issue': f'Poor performance on days: {day_list}',
                        'recommendation': 'Avoid trading on these days or use more stringent criteria.'
                    })
                    
            # Check symbol performance
            symbol_perf = self.performance_metrics['symbol_performance']
            if symbol_perf:
                worst_symbols = [symbol for symbol, metrics in symbol_perf.items() 
                               if metrics['win_rate'] < 0.3 and metrics['total_trades'] > 5]
                if worst_symbols:
                    symbol_list = ', '.join(worst_symbols)
                    recommendations.append({
                        'aspect': 'Currency Pairs',
                        'issue': f'Poor performance with pairs: {symbol_list}',
                        'recommendation': 'Consider removing these pairs from your trading plan or developing specific strategies for them.'
                    })
                    
            # Check for losing streaks
            if self.performance_metrics['max_loss_streak'] > 5:
                recommendations.append({
                    'aspect': 'Mental Management',
                    'issue': 'Long losing streaks',
                    'recommendation': 'Implement a rule to reduce position size after 2-3 consecutive losses, or take a short break from trading.'
                })
                
            # Add Claude-specific recommendations
            recommendations.append({
                'aspect': 'AI Strategy Adjustment',
                'issue': 'Optimizing Claude API usage',
                'recommendation': 'Adjust analysis frequency based on timeframe - use more frequent checks during active trading hours and less during quiet periods.'
            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []
            
    def identify_best_trading_conditions(self) -> Dict[str, Any]:
        """
        Identify the most profitable trading conditions
        
        Returns:
            Dictionary with best trading conditions
        """
        try:
            if not self.trade_history:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(self.trade_history)
            
            # Skip if empty or missing key columns
            if df.empty or 'profit_pips' not in df.columns:
                return {}
                
            # Ensure profit values are numeric
            df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
            
            # Best trading hours
            if 'entry_time' in df.columns:
                # Convert entry_time to datetime if it's a string
                if df['entry_time'].dtype == 'object':
                    df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                    
                # Extract hour
                df['hour'] = df['entry_time'].dt.hour
                
                hour_performance = df.groupby('hour').agg({
                    'profit_pips': ['sum', 'mean'],
                    'ticket': 'count'
                })
                
                hour_performance.columns = ['total_pips', 'avg_pips', 'trade_count']
                
                # Calculate win rate by hour
                win_rates = df[df['profit_pips'] > 0].groupby('hour').size() / hour_performance['trade_count']
                hour_performance['win_rate'] = win_rates
                
                best_hours = hour_performance[hour_performance['trade_count'] >= 5].sort_values('avg_pips', ascending=False).head(3)
                best_hours_dict = best_hours.to_dict()
            else:
                best_hours_dict = {}
                
            # Best trading days
            if 'entry_time' in df.columns:
                df['day_of_week'] = df['entry_time'].dt.day_name()
                
                day_performance = df.groupby('day_of_week').agg({
                    'profit_pips': ['sum', 'mean'],
                    'ticket': 'count'
                })
                
                day_performance.columns = ['total_pips', 'avg_pips', 'trade_count']
                
                # Calculate win rate by day
                win_rates = df[df['profit_pips'] > 0].groupby('day_of_week').size() / day_performance['trade_count']
                day_performance['win_rate'] = win_rates
                
                best_days = day_performance[day_performance['trade_count'] >= 5].sort_values('avg_pips', ascending=False).head(3)
                best_days_dict = best_days.to_dict()
            else:
                best_days_dict = {}
                
            # Best currency pairs
            if 'symbol' in df.columns:
                symbol_performance = df.groupby('symbol').agg({
                    'profit_pips': ['sum', 'mean'],
                    'ticket': 'count'
                })
                
                symbol_performance.columns = ['total_pips', 'avg_pips', 'trade_count']
                
                # Calculate win rate by symbol
                win_rates = df[df['profit_pips'] > 0].groupby('symbol').size() / symbol_performance['trade_count']
                symbol_performance['win_rate'] = win_rates
                
                best_symbols = symbol_performance[symbol_performance['trade_count'] >= 5].sort_values('avg_pips', ascending=False).head(3)
                best_symbols_dict = best_symbols.to_dict()
            else:
                best_symbols_dict = {}
                
            # Best trade setups (if we have a 'setup_type' field)
            best_setups_dict = {}
            if 'setup_type' in df.columns:
                setup_performance = df.groupby('setup_type').agg({
                    'profit_pips': ['sum', 'mean'],
                    'ticket': 'count'
                })
                
                setup_performance.columns = ['total_pips', 'avg_pips', 'trade_count']
                
                # Calculate win rate by setup
                win_rates = df[df['profit_pips'] > 0].groupby('setup_type').size() / setup_performance['trade_count']
                setup_performance['win_rate'] = win_rates
                
                best_setups = setup_performance[setup_performance['trade_count'] >= 5].sort_values('avg_pips', ascending=False).head(3)
                best_setups_dict = best_setups.to_dict()
                
            return {
                'best_hours': best_hours_dict,
                'best_days': best_days_dict,
                'best_symbols': best_symbols_dict,
                'best_setups': best_setups_dict
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying best trading conditions: {e}")
            return {}
            
    def create_performance_summary(self) -> str:
        """
        Create a JSON summary of performance for Claude to analyze
        
        Returns:
            JSON string with performance summary
        """
        try:
            if not self.performance_metrics:
                self.analyze_performance()
                
            # Create a summary dictionary
            summary = {
                'performance_metrics': self.performance_metrics,
                'recommendations': self.get_optimization_recommendations(),
                'best_conditions': self.identify_best_trading_conditions(),
                'trade_sample': self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
            }
            
            # Convert to JSON string
            return json.dumps(summary, indent=2, default=self._json_serializer)
            
        except Exception as e:
            self.logger.error(f"Error creating performance summary: {e}")
            return json.dumps({'error': str(e)})
            
    def generate_performance_report(self, output_dir=None) -> str:
        """
        Generate a detailed performance report with charts
        
        Args:
            output_dir: Directory to save the report, or None for default
            
        Returns:
            Path to the generated report
        """
        try:
            # Use default directory if none provided
            if output_dir is None:
                output_dir = os.path.join(DATA_DIR, "reports")
                
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure we have performance metrics
            if not self.performance_metrics:
                self.analyze_performance()
                
            if not self.performance_metrics:
                return "No performance data available"
                
            # Create report filename
            report_file = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d')}.html")
            
            # Convert trade history to DataFrame
            df = pd.DataFrame(self.trade_history)
            
            # Skip if empty
            if df.empty:
                return "No trade data available"
                
            # Format dates if needed
            if 'entry_time' in df.columns and df['entry_time'].dtype == 'object':
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                
            if 'exit_time' in df.columns and df['exit_time'].dtype == 'object':
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                
            # Generate HTML report
            html_content = []
            
            # Add header
            html_content.append("<html><head>")
            html_content.append("<title>Forex Trading Performance Report</title>")
            html_content.append("<style>")
            html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
            html_content.append("h1, h2, h3 { color: #2c3e50; }")
            html_content.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
            html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_content.append("th { background-color: #f2f2f2; }")
            html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
            html_content.append(".good { color: green; }")
            html_content.append(".bad { color: red; }")
            html_content.append(".neutral { color: orange; }")
            html_content.append(".chart { width: 100%; height: 300px; margin-bottom: 20px; }")
            html_content.append("</style>")
            html_content.append("</head><body>")
            
            # Add title
            html_content.append("<h1>Forex Trading Performance Report</h1>")
            html_content.append(f"<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Add summary section
            html_content.append("<h2>Performance Summary</h2>")
            html_content.append("<table>")
            html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            # Add key metrics
            metrics = [
                ("Total Trades", f"{self.performance_metrics['total_trades']}"),
                ("Win Rate", f"{self.performance_metrics['win_rate']:.2%}"),
                ("Profit Factor", f"{self.performance_metrics['profit_factor']:.2f}"),
                ("Expectancy (pips/trade)", f"{self.performance_metrics['expectancy']:.2f}"),
                ("Total Profit (pips)", f"{self.performance_metrics['total_pips']:.2f}"),
                ("Average Win (pips)", f"{self.performance_metrics['avg_win_pips']:.2f}"),
                ("Average Loss (pips)", f"{self.performance_metrics['avg_loss_pips']:.2f}"),
                ("Risk-Reward Ratio", f"{self.performance_metrics['avg_risk_reward']:.2f}"),
                ("Max Drawdown", f"{self.performance_metrics['max_drawdown']:.2%}"),
                ("Sharpe Ratio", f"{self.performance_metrics['sharpe_ratio']:.2f}"),
                ("Max Win Streak", f"{self.performance_metrics['max_win_streak']}"),
                ("Max Loss Streak", f"{self.performance_metrics['max_loss_streak']}")
            ]
            
            for metric, value in metrics:
                html_content.append(f"<tr><td>{metric}</td><td>{value}</td></tr>")
                
            html_content.append("</table>")
            
            # Add recommendations section
            recommendations = self.get_optimization_recommendations()
            if recommendations:
                html_content.append("<h2>Optimization Recommendations</h2>")
                html_content.append("<table>")
                html_content.append("<tr><th>Aspect</th><th>Issue</th><th>Recommendation</th></tr>")
                
                for rec in recommendations:
                    html_content.append(f"<tr><td>{rec['aspect']}</td><td>{rec['issue']}</td><td>{rec['recommendation']}</td></tr>")
                    
                html_content.append("</table>")
                
            # Add best conditions section
            best_conditions = self.identify_best_trading_conditions()
            if best_conditions:
                html_content.append("<h2>Best Trading Conditions</h2>")
                
                # Best hours
                if best_conditions.get('best_hours'):
                    html_content.append("<h3>Best Hours to Trade</h3>")
                    html_content.append("<table>")
                    html_content.append("<tr><th>Hour</th><th>Win Rate</th><th>Avg Pips</th><th>Total Trades</th></tr>")
                    
                    try:
                        for hour in best_conditions['best_hours'].get('avg_pips', {}).keys():
                            win_rate = best_conditions['best_hours'].get('win_rate', {}).get(hour, 0)
                            avg_pips = best_conditions['best_hours'].get('avg_pips', {}).get(hour, 0)
                            trade_count = best_conditions['best_hours'].get('trade_count', {}).get(hour, 0)
                            
                            html_content.append(f"<tr><td>{hour}:00</td><td>{win_rate:.2%}</td><td>{avg_pips:.2f}</td><td>{trade_count}</td></tr>")
                    except:
                        pass
                        
                    html_content.append("</table>")
                    
                # Best days
                if best_conditions.get('best_days'):
                    html_content.append("<h3>Best Days to Trade</h3>")
                    html_content.append("<table>")
                    html_content.append("<tr><th>Day</th><th>Win Rate</th><th>Avg Pips</th><th>Total Trades</th></tr>")
                    
                    try:
                        for day in best_conditions['best_days'].get('avg_pips', {}).keys():
                            win_rate = best_conditions['best_days'].get('win_rate', {}).get(day, 0)
                            avg_pips = best_conditions['best_days'].get('avg_pips', {}).get(day, 0)
                            trade_count = best_conditions['best_days'].get('trade_count', {}).get(day, 0)
                            
                            html_content.append(f"<tr><td>{day}</td><td>{win_rate:.2%}</td><td>{avg_pips:.2f}</td><td>{trade_count}</td></tr>")
                    except:
                        pass
                        
                    html_content.append("</table>")
                    
                # Best symbols
                if best_conditions.get('best_symbols'):
                    html_content.append("<h3>Best Currency Pairs</h3>")
                    html_content.append("<table>")
                    html_content.append("<tr><th>Symbol</th><th>Win Rate</th><th>Avg Pips</th><th>Total Trades</th></tr>")
                    
                    try:
                        for symbol in best_conditions['best_symbols'].get('avg_pips', {}).keys():
                            win_rate = best_conditions['best_symbols'].get('win_rate', {}).get(symbol, 0)
                            avg_pips = best_conditions['best_symbols'].get('avg_pips', {}).get(symbol, 0)
                            trade_count = best_conditions['best_symbols'].get('trade_count', {}).get(symbol, 0)
                            
                            html_content.append(f"<tr><td>{symbol}</td><td>{win_rate:.2%}</td><td>{avg_pips:.2f}</td><td>{trade_count}</td></tr>")
                    except:
                        pass
                        
                    html_content.append("</table>")
                    
            # Add recent trades section
            recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
            if recent_trades:
                html_content.append("<h2>Recent Trades</h2>")
                html_content.append("<table>")
                html_content.append("<tr><th>Ticket</th><th>Symbol</th><th>Type</th><th>Entry Time</th><th>Exit Time</th><th>Profit (pips)</th></tr>")
                
                for trade in reversed(recent_trades):
                    # Format entry and exit times
                    entry_time = trade.get('entry_time', '')
                    if isinstance(entry_time, str):
                        entry_time_str = entry_time
                    elif isinstance(entry_time, datetime):
                        entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        entry_time_str = str(entry_time)
                        
                    exit_time = trade.get('exit_time', '')
                    if isinstance(exit_time, str):
                        exit_time_str = exit_time
                    elif isinstance(exit_time, datetime):
                        exit_time_str = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        exit_time_str = str(exit_time)
                        
                    # Determine profit class
                    profit_pips = trade.get('profit_pips', 0)
                    if profit_pips > 0:
                        profit_class = "good"
                    elif profit_pips < 0:
                        profit_class = "bad"
                    else:
                        profit_class = "neutral"
                        
                    html_content.append(f"<tr>")
                    html_content.append(f"<td>{trade.get('ticket', '')}</td>")
                    html_content.append(f"<td>{trade.get('symbol', '')}</td>")
                    html_content.append(f"<td>{trade.get('type', '')}</td>")
                    html_content.append(f"<td>{entry_time_str}</td>")
                    html_content.append(f"<td>{exit_time_str}</td>")
                    html_content.append(f"<td class='{profit_class}'>{profit_pips:.2f}</td>")
                    html_content.append(f"</tr>")
                    
                html_content.append("</table>")
                
            # Close HTML
            html_content.append("</body></html>")
            
            # Write HTML to file
            with open(report_file, 'w') as f:
                f.write('\n'.join(html_content))
                
            self.logger.info(f"Generated performance report: {report_file}")
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {e}"
            
    def _calculate_daily_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from trade history
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Series with daily returns
        """
        try:
            if df.empty or 'exit_time' not in df.columns or 'profit_amount' not in df.columns:
                return pd.Series()
                
            # Convert exit_time to datetime if it's not already
            if df['exit_time'].dtype == 'object':
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                
            # Convert profit_amount to numeric if needed
            if df['profit_amount'].dtype == 'object':
                df['profit_amount'] = pd.to_numeric(df['profit_amount'], errors='coerce')
                
            # Group trades by exit date and sum profits
            daily_profits = df.groupby(df['exit_time'].dt.date)['profit_amount'].sum()
            
            # Convert to returns (assuming a fixed account size of 10,000 for simplicity)
            account_size = 10000
            daily_returns = daily_profits / account_size
            
            return daily_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating daily returns: {e}")
            return pd.Series()
            
    def _calculate_drawdown(self, df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate maximum drawdown from trade history
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Tuple of (max_drawdown, drawdown_periods)
        """
        try:
            if df.empty or 'exit_time' not in df.columns or 'profit_amount' not in df.columns:
                return 0, []
                
            # Convert exit_time to datetime if it's not already
            if df['exit_time'].dtype == 'object':
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                
            # Convert profit_amount to numeric if needed
            if df['profit_amount'].dtype == 'object':
                df['profit_amount'] = pd.to_numeric(df['profit_amount'], errors='coerce')
                
            # Sort by exit_time
            df = df.sort_values('exit_time')
            
            # Calculate cumulative profit
            df['cumulative_profit'] = df['profit_amount'].cumsum()
            
            # Calculate running maximum
            df['running_max'] = df['cumulative_profit'].cummax()
            
            # Calculate drawdown
            df['drawdown'] = df['running_max'] - df['cumulative_profit']
            
            # Find maximum drawdown
            max_drawdown = df['drawdown'].max()
            
            # Account size (assuming starting with 10,000)
            account_size = 10000
            max_drawdown_pct = max_drawdown / account_size
            
            # Find drawdown periods
            threshold = max_drawdown * 0.5  # Periods with at least 50% of max drawdown
            
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, row in df.iterrows():
                if not in_drawdown and row['drawdown'] > threshold:
                    in_drawdown = True
                    start_idx = i
                elif in_drawdown and row['drawdown'] <= threshold:
                    in_drawdown = False
                    
                    # Create drawdown period info
                    period = {
                        'start': df.loc[start_idx, 'exit_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(df.loc[start_idx, 'exit_time'], datetime) else str(df.loc[start_idx, 'exit_time']),
                        'end': row['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['exit_time'], datetime) else str(row['exit_time']),
                        'depth': df.loc[start_idx:i, 'drawdown'].max()
                    }
                    
                    drawdown_periods.append(period)
                    
            return max_drawdown_pct, drawdown_periods
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return 0, []
            
    def _analyze_streaks(self, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Analyze winning and losing streaks
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Tuple of (max_win_streak, max_loss_streak, current_streak)
        """
        try:
            if df.empty or 'profit_pips' not in df.columns:
                return 0, 0, 0
                
            # Convert profit_pips to numeric if needed
            if df['profit_pips'].dtype == 'object':
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
                
            # Sort by time if available
            if 'exit_time' in df.columns:
                if df['exit_time'].dtype == 'object':
                    df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                df = df.sort_values('exit_time')
                
            # Create a column indicating win/loss
            df['result'] = df['profit_pips'].apply(lambda x: 'win' if x > 0 else 'loss')
            
            # Calculate streaks
            df['streak_change'] = (df['result'] != df['result'].shift()).astype(int)
            df['streak_id'] = df['streak_change'].cumsum()
            
            # Group by streak and count
            streak_counts = df.groupby(['streak_id', 'result']).size().reset_index(name='count')
            
            # Find maximum win and loss streaks
            max_win_streak = streak_counts[streak_counts['result'] == 'win']['count'].max() if not streak_counts[streak_counts['result'] == 'win'].empty else 0
            max_loss_streak = streak_counts[streak_counts['result'] == 'loss']['count'].max() if not streak_counts[streak_counts['result'] == 'loss'].empty else 0
            
            # Find current streak
            latest_streak = df.iloc[-1]['streak_id']
            latest_result = df.iloc[-1]['result']
            current_streak = len(df[df['streak_id'] == latest_streak])
            
            if latest_result == 'loss':
                current_streak = -current_streak
                
            return max_win_streak, max_loss_streak, current_streak
            
        except Exception as e:
            self.logger.error(f"Error analyzing streaks: {e}")
            return 0, 0, 0
            
    def _analyze_performance_by_hour(self, df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Analyze performance by hour of day
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Dictionary with hour performance stats
        """
        try:
            if df.empty or 'entry_time' not in df.columns or 'profit_pips' not in df.columns:
                return {}
                
            # Convert entry_time to datetime if it's not already
            if df['entry_time'].dtype == 'object':
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                
            # Convert profit_pips to numeric if needed
            if df['profit_pips'].dtype == 'object':
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
                
            # Add hour column
            df['hour'] = df['entry_time'].dt.hour
            
            # Group by hour
            hours = {}
            
            for hour, group in df.groupby('hour'):
                total_trades = len(group)
                winning_trades = len(group[group['profit_pips'] > 0])
                
                if total_trades > 0:
                    hours[hour] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': winning_trades / total_trades,
                        'total_pips': group['profit_pips'].sum(),
                        'avg_pips': group['profit_pips'].mean()
                    }
                    
            return hours
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by hour: {e}")
            return {}
            
    def _analyze_performance_by_day(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by day of week
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Dictionary with day performance stats
        """
        try:
            if df.empty or 'entry_time' not in df.columns or 'profit_pips' not in df.columns:
                return {}
                
            # Convert entry_time to datetime if it's not already
            if df['entry_time'].dtype == 'object':
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                
            # Convert profit_pips to numeric if needed
            if df['profit_pips'].dtype == 'object':
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
                
            # Add day column
            df['day'] = df['entry_time'].dt.day_name()
            
            # Group by day
            days = {}
            
            for day, group in df.groupby('day'):
                total_trades = len(group)
                winning_trades = len(group[group['profit_pips'] > 0])
                
                if total_trades > 0:
                    days[day] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': winning_trades / total_trades,
                        'total_pips': group['profit_pips'].sum(),
                        'avg_pips': group['profit_pips'].mean()
                    }
                    
            return days
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by day: {e}")
            return {}
            
    def _analyze_performance_by_symbol(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by symbol
        
        Args:
            df: DataFrame with trade history
            
        Returns:
            Dictionary with symbol performance stats
        """
        try:
            if df.empty or 'symbol' not in df.columns or 'profit_pips' not in df.columns:
                return {}
                
            # Convert profit_pips to numeric if needed
            if df['profit_pips'].dtype == 'object':
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
                
            # Group by symbol
            symbols = {}
            
            for symbol, group in df.groupby('symbol'):
                total_trades = len(group)
                winning_trades = len(group[group['profit_pips'] > 0])
                
                if total_trades > 0:
                    symbols[symbol] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': winning_trades / total_trades,
                        'total_pips': group['profit_pips'].sum(),
                        'avg_pips': group['profit_pips'].mean()
                    }
                    
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by symbol: {e}")
            return {}
            
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")


class StrategyOptimizer:
    """
    Optimizes trading strategies based on performance analysis
    and provides recommendations for Claude API prompts.
    """
    def __init__(self, performance_analyzer):
        self.logger = logging.getLogger(__name__)
        self.performance_analyzer = performance_analyzer
        self.optimization_params = {}
        
    def set_optimization_params(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for optimization
        
        Args:
            params: Dictionary of optimization parameters
        """
        self.optimization_params = params
        
    def recommend_claude_prompts(self) -> Dict[str, Any]:
        """
        Generate optimized prompts for Claude based on performance data
        
        Returns:
            Dictionary with prompt recommendations
        """
        try:
            performance = self.performance_analyzer.performance_metrics
            if not performance:
                return {"status": "error", "message": "Insufficient data for optimization"}
                
            # Extract key metrics
            win_rate = performance.get('win_rate', 0)
            profit_factor = performance.get('profit_factor', 0)
            max_drawdown = performance.get('max_drawdown', 0)
            
            # Best trading conditions
            best_conditions = self.performance_analyzer.identify_best_trading_conditions()
            
            # Optimize based on performance
            prompt_recommendations = []
            
            # Base prompt template
            base_prompt = """
            Analyze the following forex data for {symbol} on {timeframe} timeframe:
            
            {data_summary}
            
            Based on this analysis:
            1. Identify the current market regime (trending, ranging, volatile)
            2. Determine potential entry points with precise stop loss and take profit levels
            3. Calculate optimal position size based on current volatility
            4. Provide confidence level for any trade recommendation
            """
            
            # Add emphasis based on performance
            if win_rate < 0.4:
                prompt_recommendations.append({
                    'focus': 'Entry Quality',
                    'prompt_addition': """
                    Focus on high-probability setups only. Look for:
                    - Strong confluence of multiple indicators
                    - Clear support/resistance levels
                    - Alignment across at least 2 higher timeframes
                    - Rejection patterns at key levels
                    
                    Only provide trade recommendations with 80%+ confidence.
                    """
                })
                
            if profit_factor < 1.5:
                prompt_recommendations.append({
                    'focus': 'Profit Maximization',
                    'prompt_addition': """
                    For winning trades, provide:
                    - Multiple take-profit targets (partial exit points)
                    - Trailing stop strategy based on ATR
                    - Key exit indicators to watch for trend exhaustion
                    - Scale-out levels to maximize profits on strong moves
                    """
                })
                
            if max_drawdown > 0.15:
                prompt_recommendations.append({
                    'focus': 'Risk Management',
                    'prompt_addition': """
                    Emphasize strict risk management:
                    - Limit position size to maximum 1.5% risk per trade
                    - Ensure minimum 1:2 risk-reward ratio
                    - Identify nearby liquidity levels that could cause stop hunts
                    - Provide market invalidation points beyond stop loss
                    """
                })
                
            # Time-specific recommendations
            best_hours = best_conditions.get('best_hours', {})
            if best_hours:
                hours_list = list(best_hours.get('avg_pips', {}).keys())
                prompt_recommendations.append({
                    'focus': 'Time Optimization',
                    'prompt_addition': f"""
                    Pay special attention to setups forming during the following hours:
                    {', '.join([f"{h}:00" for h in hours_list])} UTC
                    
                    These times have historically shown higher profitability.
                    """
                })
                
            # Add currency-specific guidance
            best_symbols = best_conditions.get('best_symbols', {})
            if best_symbols:
                symbols_list = list(best_symbols.get('avg_pips', {}).keys())
                prompt_recommendations.append({
                    'focus': 'Currency Specialization',
                    'prompt_addition': f"""
                    Use specialized analysis for these currency pairs:
                    {', '.join(symbols_list)}
                    
                    These pairs have shown the highest profitability in past trading.
                    """
                })
                
            # Build final optimized prompt
            optimized_prompt = base_prompt
            for rec in prompt_recommendations:
                optimized_prompt += f"\n\n# {rec['focus']}\n{rec['prompt_addition']}"
                
            return {
                'status': 'success',
                'base_prompt': base_prompt,
                'recommendations': prompt_recommendations,
                'optimized_prompt': optimized_prompt
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending Claude prompts: {e}")
            return {"status": "error", "message": str(e)}
        
    def optimize_trading_parameters(self) -> Dict[str, Any]:
        """
        Recommend optimized trading parameters based on performance
        
        Returns:
            Dictionary with optimized parameters
        """
        try:
            performance = self.performance_analyzer.performance_metrics
            if not performance:
                return {}
                
            # Extract trade history
            trade_history = self.performance_analyzer.trade_history
            if not trade_history:
                return {}
                
            # Convert to DataFrame
            df = pd.DataFrame(trade_history)
            
            # Return empty if no trades
            if df.empty:
                return {}
                
            # Ensure we have numeric values
            if 'profit_pips' in df.columns:
                df['profit_pips'] = pd.to_numeric(df['profit_pips'], errors='coerce')
                
            if 'profit_amount' in df.columns:
                df['profit_amount'] = pd.to_numeric(df['profit_amount'], errors='coerce')
                
            # Optimize parameters
            optimized_params = {}
            
            # 1. Optimize stop loss distance based on ATR
            if 'atr' in df.columns and 'initial_stop_loss' in df.columns and 'entry_price' in df.columns:
                # Calculate stop distance in ATR units
                df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
                df['initial_stop_loss'] = pd.to_numeric(df['initial_stop_loss'], errors='coerce')
                df['atr'] = pd.to_numeric(df['atr'], errors='coerce')
                
                df['stop_distance_atr'] = abs(df['entry_price'] - df['initial_stop_loss']) / df['atr']
                
                # Find optimal ATR multiplier for stop loss
                winning_stops = df[df['profit_pips'] > 0]['stop_distance_atr'].mean()
                
                optimized_params['stop_loss_atr_multiplier'] = winning_stops
                
            # 2. Optimize take profit distance
            if 'initial_take_profit' in df.columns and 'initial_stop_loss' in df.columns and 'entry_price' in df.columns:
                # Calculate risk reward ratio
                df['take_profit_distance'] = abs(df['initial_take_profit'] - df['entry_price'])
                df['stop_loss_distance'] = abs(df['initial_stop_loss'] - df['entry_price'])
                
                # Calculate RR ratio where stop loss distance > 0
                mask = df['stop_loss_distance'] > 0
                df.loc[mask, 'rr_ratio'] = df.loc[mask, 'take_profit_distance'] / df.loc[mask, 'stop_loss_distance']
                
                # Find optimal RR ratio for winning trades
                winning_mask = (df['profit_pips'] > 0) & mask
                if winning_mask.any():
                    optimal_rr = df.loc[winning_mask, 'rr_ratio'].mean()
                    optimized_params['optimal_risk_reward'] = optimal_rr
                    
            # 3. Optimize trailing stop parameters
            if 'trailing_activated' in df.columns:
                trailing_trades = df[df['trailing_activated'] == True]
                non_trailing_trades = df[df['trailing_activated'] == False]
                
                if len(trailing_trades) > 5 and len(non_trailing_trades) > 5:
                    trailing_profit = trailing_trades['profit_pips'].mean()
                    non_trailing_profit = non_trailing_trades['profit_pips'].mean()
                    
                    trailing_effectiveness = trailing_profit / non_trailing_profit if non_trailing_profit > 0 else 1
                    
                    optimized_params['use_trailing_stops'] = trailing_effectiveness > 1.2
                    
            # 4. Optimize entry confirmation
            if 'timeframes_aligned' in df.columns:
                # Group by alignment level
                df['timeframes_aligned'] = pd.to_numeric(df['timeframes_aligned'], errors='coerce')
                
                # Round to nearest 0.1 for binning
                df['alignment_bin'] = (df['timeframes_aligned'] * 10).round() / 10
                
                # Group by alignment level
                alignment_performance = df.groupby('alignment_bin').agg({
                    'profit_pips': 'mean',
                    'ticket': 'count'
                })
                
                # Find optimal alignment threshold with at least 5 trades
                if not alignment_performance.empty:
                    valid_alignments = alignment_performance[alignment_performance['ticket'] >= 5]
                    if not valid_alignments.empty:
                        optimal_alignment = valid_alignments['profit_pips'].idxmax()
                        optimized_params['min_timeframe_alignment'] = optimal_alignment
                        
            # 5. Optimize position sizing
            if 'risk_amount' in df.columns and 'profit_amount' in df.columns:
                # Calculate return on risk
                mask = df['risk_amount'] > 0
                df.loc[mask, 'return_on_risk'] = df.loc[mask, 'profit_amount'] / df.loc[mask, 'risk_amount']
                
                # Find optimal risk per trade
                valid_returns = df['return_on_risk'].dropna()
                if not valid_returns.empty:
                    avg_return_on_risk = valid_returns.mean()
                    
                    if avg_return_on_risk > 0:
                        optimal_risk = min(0.02, 0.01 * avg_return_on_risk)  # Cap at 2%
                    else:
                        optimal_risk = 0.01  # Default to 1%
                        
                    optimized_params['optimal_risk_per_trade'] = optimal_risk
                    
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing trading parameters: {e}")
            return {}
        
    def generate_claude_optimization_report(self) -> str:
        """
        Generate a comprehensive optimization report for Claude to analyze
        
        Returns:
            JSON string with optimization report
        """
        try:
            # Get performance metrics
            performance = self.performance_analyzer.analyze_performance()
            
            # Get optimization recommendations
            recommendations = self.performance_analyzer.get_optimization_recommendations()
            
            # Get optimized parameters
            optimized_params = self.optimize_trading_parameters()
            
            # Get optimal prompt recommendations
            prompt_recs = self.recommend_claude_prompts()
            
            # Build the report
            report = {
                'performance_summary': {
                    'win_rate': performance.get('win_rate', 0),
                    'profit_factor': performance.get('profit_factor', 0),
                    'expectancy': performance.get('expectancy', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0)
                },
                'optimization_recommendations': recommendations,
                'optimized_parameters': optimized_params,
                'claude_prompt_optimization': prompt_recs,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Convert to JSON string
            return json.dumps(report, indent=2, default=self._json_serializer)
            
        except Exception as e:
            self.logger.error(f"Error generating Claude optimization report: {e}")
            return json.dumps({"error": str(e)})
        
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")