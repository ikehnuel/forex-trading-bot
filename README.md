# Forex Trading Bot with Claude AI

A sophisticated forex trading bot that leverages Claude AI for market analysis and trading decisions. The bot connects to MetaTrader 5 for execution and uses advanced risk management and multi-timeframe analysis.

## Features

- **AI-Powered Analysis**: Uses Claude AI to analyze market conditions and make trading decisions
- **Multi-Timeframe Analysis**: Analyzes multiple timeframes to find confluent signals
- **Market Regime Detection**: Identifies market conditions (trending, ranging, volatile)
- **Advanced Risk Management**: Dynamic position sizing and risk controls
- **Trade Management**: Trailing stops, partial exits, and breakeven strategies
- **Performance Analytics**: Tracks and analyzes trading performance

## Requirements

- Python 3.10+
- Docker and Docker Compose
- MetaTrader 5 account
- Claude API key from Anthropic

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/ikehnuel/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Create and configure your environment file:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your API keys and trading parameters.

3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ikehnuel/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your API keys and trading parameters.

5. Run the bot:
   ```bash
   python src/main.py
   ```

## Configuration

The bot can be configured through environment variables or the `.env` file:

- `CLAUDE_API_KEY`: Your Claude API key
- `SYMBOL`: Trading symbol (default: EURUSD)
- `TIMEFRAME`: Trading timeframe (default: H1)
- `CHECK_INTERVAL`: Time between analyses in seconds (default: 3600)
- `MAX_RISK_PER_TRADE`: Maximum risk per trade as decimal (default: 0.02)
- `MAX_DAILY_RISK`: Maximum daily risk as decimal (default: 0.05)
- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`: MT5 credentials

Additional configuration options are available in the `.env.example` file.

## Architecture

The bot is built with a modular architecture:

- **MT5 Connector**: Handles communication with MetaTrader 5
- **Analysis System**: Market regime detection and multi-timeframe analysis
- **Claude Analyzer**: Communicates with Claude AI API
- **Trade Management**: Manages trade lifecycle and risk
- **Performance Analytics**: Tracks and optimizes performance

## Performance Optimization

The system includes performance analytics that:

1. Tracks all trade results and performance metrics
2. Analyzes win rates by time, currency pair, and conditions
3. Provides optimization recommendations
4. Adapts Claude prompts based on historical performance
5. Optimizes position sizing based on volatility and results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading forex involves significant risk of loss. Past performance is not indicative of future results.