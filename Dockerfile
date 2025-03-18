FROM python:3.10-slim

# Install system dependencies
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
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Wine for running MT5
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wine \
    wine32 \
    && rm -rf /var/lib/apt/lists/*

# Install MetaTrader 5
RUN mkdir -p /mt5 && \
    wget -q -O /tmp/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe && \
    wine /tmp/mt5setup.exe /auto && \
    rm /tmp/mt5setup.exe

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/trade_history /app/data/market_data /app/data/reports /app/logs

# Set environment variables
ENV DISPLAY=:99
ENV MT5_PATH="/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Add startup script
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

# Start virtual display and application
ENTRYPOINT ["/start.sh"]
