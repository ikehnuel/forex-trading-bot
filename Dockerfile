FROM python:3.10-slim

# Install system dependencies with proper cleanup
RUN apt-get update && apt-get install -y \
    wget unzip xvfb xauth libglib2.0-0 libnss3 libgconf-2-4 \
    libfontconfig1 libxrender1 libxtst6 wine xdg-utils \
    x11-xserver-utils dbus ca-certificates build-essential autoconf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV DISPLAY=:99

# Install MT5 with proper cleanup
RUN mkdir -p /mt5 && \
    wget -q -O /tmp/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe && \
    (Xvfb :99 -screen 0 1024x768x16 & \
    sleep 3 && \
    wine /tmp/mt5setup.exe /auto && \
    sleep 10 && \
    killall Xvfb || true) && \
    rm -f /tmp/mt5setup.exe

# Python package installation in a single layer to save space
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.23.5 pandas==2.1.0 scipy==1.11.3 && \
    pip install --no-cache-dir TA-Lib-Precompiled && \
    pip install --no-cache-dir MetaTrader5 && \
    pip install --no-cache-dir colorama anthropic==0.15.0 python-dotenv==1.0.0 \
    requests==2.31.0 matplotlib==3.7.3 seaborn==0.13.0 statsmodels==0.14.0 pytz==2023.3

# Directory setup
RUN mkdir -p /app/data/trade_history /app/data/market_data /app/data/reports /app/logs

# Copy application code
COPY . .

# Environment configuration
ENV MT5_PATH="/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy and setup entrypoint
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
