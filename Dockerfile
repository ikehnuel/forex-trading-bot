# Use a Windows Server Core image with Python installed
FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Set working directory
WORKDIR C:/app

# Install Python using PowerShell
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe -OutFile python-3.10.0.exe ; \
    Start-Process python-3.10.0.exe -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1' -NoNewWindow -Wait ; \
    Remove-Item -Force python-3.10.0.exe

# Refresh environment variables to recognize Python
RUN refreshenv || setx /M PATH "%PATH%;C:\Program Files\Python310;C:\Program Files\Python310\Scripts"

# Download and install MetaTrader 5 directly (no need for Wine/Xvfb on Windows)
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe -OutFile mt5setup.exe ; \
    Start-Process -FilePath mt5setup.exe -ArgumentList '/auto' -NoNewWindow -Wait ; \
    Remove-Item -Force mt5setup.exe

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.23.5 pandas==2.1.0 scipy==1.11.3 && \
    pip install --no-cache-dir TA-Lib-Precompiled && \
    pip install --no-cache-dir MetaTrader5 && \
    pip install --no-cache-dir colorama anthropic==0.15.0 python-dotenv==1.0.0 \
    requests==2.31.0 matplotlib==3.7.3 seaborn==0.13.0 statsmodels==0.14.0 pytz==2023.3

# Create directories (Windows path format)
RUN mkdir C:\app\data\trade_history C:\app\data\market_data C:\app\data\reports C:\app\logs

# Copy application code
COPY . .

# Set environment variables (Windows path format)
ENV MT5_PATH="C:\\Program Files\\MetaTrader 5\\terminal64.exe"
ENV PYTHONPATH="C:\\app;${PYTHONPATH}"

# Convert start.sh to start.ps1 or start.cmd
# For simplicity, we'll just create a basic CMD file
RUN echo @echo off > start.cmd && \
    echo python main.py >> start.cmd

# Use CMD file as entrypoint
ENTRYPOINT ["C:\\app\\start.cmd"]
