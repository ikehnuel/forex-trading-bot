import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# API Keys
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20240229')

# Trading Parameters
DEFAULT_SYMBOL = os.getenv('SYMBOL', 'EURUSD')
DEFAULT_TIMEFRAME = os.getenv('TIMEFRAME', 'H1')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '3600'))  # Default 1 hour in seconds
MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))  # 2% risk per trade
MAX_DAILY_RISK = float(os.getenv('MAX_DAILY_RISK', '0.05'))  # 5% max daily risk
MAX_DRAWDOWN_THRESHOLD = float(os.getenv('MAX_DRAWDOWN_THRESHOLD', '0.15'))  # 15% max drawdown threshold

# MT5 Connection Parameters
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')
MT5_PATH = os.getenv('MT5_PATH', '')

# Timeframe mappings
TIMEFRAME_MAP = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 60,
    'H4': 240,
    'D1': 1440,
    'W1': 10080,
    'MN1': 43200
}

# Timeframe relationships for analysis
TIMEFRAME_RELATIONSHIPS = {
    'M1': ['M5', 'M15', 'H1'],
    'M5': ['M15', 'M30', 'H1'],
    'M15': ['M5', 'M30', 'H1'],
    'M30': ['M15', 'H1', 'H4'],
    'H1': ['M30', 'H4', 'D1'],
    'H4': ['H1', 'D1', 'W1'],
    'D1': ['H4', 'W1', 'MN1'],
    'W1': ['D1', 'MN1']
}

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRADE_HISTORY_DIR = os.path.join(DATA_DIR, 'trade_history')
MARKET_DATA_DIR = os.path.join(DATA_DIR, 'market_data')

# Create directories if they don't exist
for directory in [DATA_DIR, LOG_DIR, TRADE_HISTORY_DIR, MARKET_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
def setup_logging():
    log_file = os.path.join(LOG_DIR, f"forex_bot_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger('forex_bot')

# Initialize logger
logger = setup_logging()

# Trading parameters
TRADING_CONFIG = {
    'position_sizing_method': os.getenv('POSITION_SIZING_METHOD', 'fixed'),  # fixed, atr, volatility
    'trailing_stop': {
        'enable': os.getenv('TRAILING_STOP_ENABLE', 'True').lower() == 'true',
        'activation_percent': float(os.getenv('TRAILING_ACTIVATION_PERCENT', '0.5')),
        'step_percent': float(os.getenv('TRAILING_STEP_PERCENT', '0.2')),
        'lock_percent': float(os.getenv('TRAILING_LOCK_PERCENT', '0.6'))
    },
    'partial_exit': {
        'enable': os.getenv('PARTIAL_EXIT_ENABLE', 'True').lower() == 'true',
        'first_target_percent': float(os.getenv('FIRST_TARGET_PERCENT', '0.4')),
        'first_exit_percent': float(os.getenv('FIRST_EXIT_PERCENT', '0.3')),
        'second_target_percent': float(os.getenv('SECOND_TARGET_PERCENT', '0.7')),
        'second_exit_percent': float(os.getenv('SECOND_EXIT_PERCENT', '0.3'))
    },
    'breakeven': {
        'enable': os.getenv('BREAKEVEN_ENABLE', 'True').lower() == 'true',
        'activation_percent': float(os.getenv('BREAKEVEN_ACTIVATION_PERCENT', '0.3')),
        'buffer_pips': float(os.getenv('BREAKEVEN_BUFFER_PIPS', '5'))
    }
}

# API usage optimization
API_OPTIMIZATION = {
    'enable': os.getenv('API_OPTIMIZATION_ENABLE', 'True').lower() == 'true',
    'max_candles': int(os.getenv('MAX_CANDLES', '20')),
    'daily_token_limit': int(os.getenv('DAILY_TOKEN_LIMIT', '100000')),
    'batch_requests': os.getenv('BATCH_REQUESTS', 'True').lower() == 'true'
}