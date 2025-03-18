import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector.connection import MT5Connector

class TestMT5Connector(unittest.TestCase):
    @patch('src.mt5_connector.connection.mt5')
    def test_connect_success(self, mock_mt5):
        # Configure the mock
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock()
        
        # Create connector and test connection
        connector = MT5Connector()
        result = connector.connect()
        
        # Assert that the connection was successful
        self.assertTrue(result)
        self.assertTrue(connector.initialized)
        mock_mt5.initialize.assert_called_once()
    
    @patch('src.mt5_connector.connection.mt5')
    def test_connect_failure(self, mock_mt5):
        # Configure the mock to simulate connection failure
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = "Connection error"
        
        # Create connector and test connection
        connector = MT5Connector()
        result = connector.connect()
        
        # Assert that the connection failed
        self.assertFalse(result)
        self.assertFalse(connector.initialized)
        mock_mt5.initialize.assert_called_once()
    
    @patch('src.mt5_connector.connection.mt5')
    def test_get_account_info(self, mock_mt5):
        # Configure the mock
        mock_account = MagicMock()
        mock_account.balance = 10000
        mock_account.equity = 10100
        mock_account.margin = 1000
        mock_account.margin_free = 9100
        mock_account.profit = 100
        mock_mt5.account_info.return_value = mock_account
        
        # Create connector
        connector = MT5Connector()
        connector.initialized = True
        
        # Get account info
        account_info = connector.get_account_info()
        
        # Assert account info was returned correctly
        self.assertEqual(account_info["balance"], 10000)
        self.assertEqual(account_info["equity"], 10100)
        self.assertEqual(account_info["profit"], 100)
        mock_mt5.account_info.assert_called_once()
    
    @patch('src.mt5_connector.connection.mt5')
    def test_is_market_open(self, mock_mt5):
        # Configure the mock
        mock_symbol_info = MagicMock()
        mock_symbol_info.trade_mode = 1  # Not SYMBOL_TRADE_MODE_DISABLED (0)
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        # Create connector
        connector = MT5Connector()
        connector.initialized = True
        
        # Check if market is open
        result = connector.is_market_open("EURUSD")
        
        # Assert market is open
        self.assertTrue(result)
        mock_mt5.symbol_info.assert_called_once_with("EURUSD")


# tests/test_market_regime.py
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.market_regime import MarketRegimeDetector

class TestMarketRegimeDetector(unittest.TestCase):
    def setUp(self):
        # Create sample OHLC data
        self.data = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=50, freq='H'),
            'open': np.random.normal(1.2000, 0.0010, 50),
            'high': np.random.normal(1.2010, 0.0010, 50),
            'low': np.random.normal(1.1990, 0.0010, 50),
            'close': np.random.normal(1.2005, 0.0010, 50),
            'volume': np.random.randint(100, 1000, 50)
        })
        
        # Add indicators
        self.data['sma20'] = self.data['close'].rolling(window=20).mean()
        self.data['sma50'] = self.data['close'].rolling(window=50).mean()
        
        # Create uptrend data
        self.uptrend_data = self.data.copy()
        self.uptrend_data['close'] = np.linspace(1.1990, 1.2100, 50)
        self.uptrend_data['sma20'] = self.uptrend_data['close'].rolling(window=20).mean()
        
        # Create downtrend data
        self.downtrend_data = self.data.copy()
        self.downtrend_data['close'] = np.linspace(1.2100, 1.1990, 50)
        self.downtrend_data['sma20'] = self.downtrend_data['close'].rolling(window=20).mean()
        
        # Create ranging data
        self.ranging_data = self.data.copy()
        self.ranging_data['close'] = np.concatenate([
            np.linspace(1.2000, 1.2020, 10),
            np.linspace(1.2020, 1.2000, 10),
            np.linspace(1.2000, 1.2020, 10),
            np.linspace(1.2020, 1.2000, 10),
            np.linspace(1.2000, 1.2020, 10)
        ])
        self.ranging_data['sma20'] = self.ranging_data['close'].rolling(window=20).mean()
        
        # Calculate ATR
        tr1 = self.data['high'] - self.data['low']
        tr2 = (self.data['high'] - self.data['close'].shift()).abs()
        tr3 = (self.data['low'] - self.data['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['atr14'] = tr.rolling(window=14).mean()
        
        # Create detector
        self.detector = MarketRegimeDetector()
    
    def test_detect_regime_with_uptrend(self):
        # Test uptrend detection
        uptrend_result = self.detector.detect_regime(self.uptrend_data)
        
        self.assertIsNotNone(uptrend_result)
        self.assertTrue('regime' in uptrend_result)
        self.assertTrue(uptrend_result['trend_strength'] > 0)
        self.assertEqual(uptrend_result['regime'], 'trending_up')
    
    def test_detect_regime_with_downtrend(self):
        # Test downtrend detection
        downtrend_result = self.detector.detect_regime(self.downtrend_data)
        
        self.assertIsNotNone(downtrend_result)
        self.assertTrue('regime' in downtrend_result)
        self.assertTrue(downtrend_result['trend_strength'] < 0)
        self.assertEqual(downtrend_result['regime'], 'trending_down')
    
    def test_detect_regime_with_ranging(self):
        # Test ranging detection
        ranging_result = self.detector.detect_regime(self.ranging_data)
        
        self.assertIsNotNone(ranging_result)
        self.assertTrue('regime' in ranging_result)
        self.assertTrue(abs(ranging_result['trend_strength']) < 0.5)  # Low trend strength for ranging
    
    def test_get_optimal_strategy(self):
        # Test strategy recommendations
        for regime in ['trending_up', 'trending_down', 'ranging', 'breakout_up', 'reversal', 'choppy']:
            regime_info = {'regime': regime, 'volatility': 'normal'}
            strategy = self.detector.get_optimal_strategy(regime_info)
            
            self.assertIsNotNone(strategy)
            self.assertTrue('strategy' in strategy)
            self.assertTrue('confidence' in strategy)
            
            # Verify strategy types match regimes
            if regime == 'trending_up' or regime == 'trending_down':
                self.assertEqual(strategy['strategy'], 'trend_following')
            elif regime == 'ranging':
                self.assertEqual(strategy['strategy'], 'mean_reversion')
            elif 'breakout' in regime:
                self.assertEqual(strategy['strategy'], 'breakout')
            elif regime == 'reversal':
                self.assertEqual(strategy['strategy'], 'reversal')
            else:
                self.assertEqual(strategy['strategy'], 'selective')


# tests/test_risk_manager.py
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trade_management.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager(
            max_risk_per_trade=0.02,  # 2% risk per trade
            max_daily_risk=0.05,      # 5% daily risk
            max_drawdown_threshold=0.15  # 15% drawdown threshold
        )
        
        # Sample account info
        self.account_info = {
            'balance': 10000,
            'equity': 10000,
            'margin': 1000,
            'free_margin': 9000
        }
        
        # Sample symbol info
        self.symbol_info = {
            'name': 'EURUSD',
            'point': 0.0001,
            'volume_min': 0.01,
            'volume_max': 10.0,
            'volume_step': 0.01,
            'trade_contract_size': 100000,
            'trade_tick_size': 0.00001,
            'trade_tick_value': 1.0,
            'is_market_open': True,
            'trade_mode': 0  # TRADE_MODE_FULL
        }
    
    def test_can_place_trade_normal_conditions(self):
        # Test normal trading conditions
        can_trade, reason = self.risk_manager.can_place_trade(self.account_info, self.symbol_info)
        
        self.assertTrue(can_trade)
        self.assertEqual(reason, "Trade allowed")
    
    def test_can_place_trade_daily_risk_exceeded(self):
        # Test daily risk limit exceeded
        self.risk_manager.daily_risk_used = 0.06  # Above 5% daily limit
        
        can_trade, reason = self.risk_manager.can_place_trade(self.account_info, self.symbol_info)
        
        self.assertFalse(can_trade)
        self.assertTrue("Daily risk limit" in reason)
    
    def test_can_place_trade_excessive_drawdown(self):
        # Test excessive drawdown
        self.risk_manager.peak_balance = 12000  # Previous peak
        self.risk_manager.update_drawdown(10000)  # Current balance, 16.7% drawdown
        
        can_trade, reason = self.risk_manager.can_place_trade(self.account_info, self.symbol_info)
        
        self.assertFalse(can_trade)
        self.assertTrue("drawdown threshold exceeded" in reason)
    
    def test_calculate_position_size_fixed(self):
        # Test fixed percentage position sizing
        position_size = self.risk_manager.calculate_position_size(
            account_info=self.account_info,
            entry_price=1.2000,
            stop_loss=1.1950,  # 50 pip stop
            symbol_info=self.symbol_info
        )
        
        # Expected calculation:
        # Risk amount = 10000 * 0.02 = 200
        # Risk pips = 50
        # Pip value for 1 lot = 10
        # Position size = 200 / (50 * 10) = 0.4 lots
        
        self.assertAlmostEqual(position_size, 0.4, delta=0.01)
    
    def test_calculate_position_size_with_drawdown(self):
        # Test position sizing with drawdown scaling
        self.risk_manager.peak_balance = 11000
        self.risk_manager.update_drawdown(10000)  # 9.1% drawdown
        
        position_size = self.risk_manager.calculate_position_size(
            account_info=self.account_info,
            entry_price=1.2000,
            stop_loss=1.1950,  # 50 pip stop
            symbol_info=self.symbol_info
        )
        
        # Expected to be reduced from normal position size
        normal_size = 0.4
        self.assertLess(position_size, normal_size)
    
    def test_consecutive_losses_impact(self):
        # Test impact of consecutive losses on position sizing
        self.risk_manager.consecutive_losses = 2
        
        position_size = self.risk_manager.calculate_position_size(
            account_info=self.account_info,
            entry_price=1.2000,
            stop_loss=1.1950,
            symbol_info=self.symbol_info
        )
        
        # Expected to be reduced from normal position size
        normal_size = 0.4
        self.assertLess(position_size, normal_size)
    
    def test_consecutive_wins_impact(self):
        # Test impact of consecutive wins on position sizing
        self.risk_manager.consecutive_wins = 3
        
        position_size = self.risk_manager.calculate_position_size(
            account_info=self.account_info,
            entry_price=1.2000,
            stop_loss=1.1950,
            symbol_info=self.symbol_info
        )
        
        # Expected to be increased from normal position size
        normal_size = 0.4
        self.assertGreater(position_size, normal_size)


# Run all tests
if __name__ == '__main__':
    unittest.main()