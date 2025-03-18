#!/bin/bash
# Startup script for the forex trading bot

# Start Xvfb
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

# Check if we need to wait for MT5 installation
if [ ! -f "/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe" ]; then
    echo "Waiting for MT5 installation to complete..."
    sleep 30
fi

# Create directories if they don't exist
mkdir -p /app/data/trade_history
mkdir -p /app/data/market_data
mkdir -p /app/data/reports
mkdir -p /app/logs

# Check if config exists
if [ ! -f "/app/config/.env" ]; then
    echo "Configuration file not found. Creating from template..."
    cp /app/.env.example /app/config/.env
    echo "Please update the configuration file with your credentials."
fi

# Load configuration
if [ -f "/app/config/.env" ]; then
    echo "Loading configuration from /app/config/.env"
    export $(grep -v '^#' /app/config/.env | xargs)
fi

# Start the trading bot
echo "Starting Forex Trading Bot..."
cd /app
python -u src/main.py
