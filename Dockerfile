# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Install system dependencies including minimal required libraries
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    xvfb \
    xauth \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    wine \
    xdg-utils \
    x11-xserver-utils \
    dbus \
    ca-certificates \
    build-essential \
    autoconf \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set up Xvfb for Wine
ENV DISPLAY=:99

# Install MT5 with proper Xvfb setup
RUN mkdir -p /mt5 && \
    wget -q -O /tmp/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe && \
    (Xvfb :99 -screen 0 1024x768x16 & \
    sleep 3 && \
    wine /tmp/mt5setup.exe /auto && \
    sleep 10 && \
    killall Xvfb) || true && \
    rm /tmp/mt5setup.exe

# Copy requirements first for better caching
COPY requirements.txt .

# Install pip tools for better reliability
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core numerical packages first
RUN pip install --no-cache-dir numpy==1.23.5 pandas==2.1.0 scipy==1.11.3

# Install TA-Lib-Precompiled instead of TA-Lib
RUN pip install --no-cache-dir TA-Lib-Precompiled

# Install MetaTrader5 package
RUN pip install --no-cache-dir MetaTrader5

# Install remaining packages
RUN pip install --no-cache-dir \
    colorama \
    anthropic==0.15.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    matplotlib==3.7.3 \
    seaborn==0.13.0 \
    statsmodels==0.14.0 \
    pytz==2023.3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/trade_history /app/data/market_data /app/data/reports /app/logs

# Set environment variables
ENV MT5_PATH="/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Add startup script
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

# Start virtual display and application
ENTRYPOINT ["/start.sh"]
