# src/web_monitor/app.py
import os
import json
import logging
import time
import pandas as pd
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, setup_logging, TRADING_CONFIG
)
from mt5_connector.connection import MT5Connector
from mt5_connector.data_collector import DataCollector
from mt5_connector.order_executor import OrderExecutor
from performance.performance_analyzer import PerformanceAnalyzer

# Initialize logger
logger = setup_logging()

# Create Flask app
app = Flask(__name__)

# Global variables
mt5_connector = None
data_collector = None
order_executor = None
performance_analyzer = None
bot_status = {
    "status": "stopped",
    "last_check": None,
    "started_at": None,
    "uptime": 0,
    "api_calls": 0,
    "api_cost": 0.0,
    "trades_executed": 0,
    "trades_active": 0
}

# Initialize components
def initialize_components():
    global mt5_connector, data_collector, order_executor, performance_analyzer
    
    try:
        # Initialize MT5 connection
        mt5_connector = MT5Connector()
        if not mt5_connector.check_connection():
            logger.error("Failed to connect to MT5")
            return False
            
        # Initialize data collection
        data_collector = DataCollector(mt5_connector)
        
        # Initialize order execution
        order_executor = OrderExecutor(mt5_connector)
        
        # Initialize performance analysis
        performance_analyzer = PerformanceAnalyzer()
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False

# Background thread for status updates
def status_updater():
    global bot_status
    
    while True:
        try:
            # Update bot status
            if bot_status["status"] == "running" and bot_status["started_at"] is not None:
                # Calculate uptime
                uptime_seconds = (datetime.now() - datetime.fromisoformat(bot_status["started_at"])).total_seconds()
                bot_status["uptime"] = uptime_seconds
            
            # Update active trades count if order executor is available
            if order_executor is not None:
                try:
                    active_positions = order_executor.get_open_positions()
                    bot_status["trades_active"] = len(active_positions)
                except:
                    pass
                
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in status updater: {e}")
            time.sleep(10)  # Longer delay on error

# Start background thread
status_thread = threading.Thread(target=status_updater, daemon=True)
status_thread.start()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', bot_status=bot_status)

@app.route('/api/status')
def api_status():
    """Get current bot status"""
    return jsonify(bot_status)

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_status
    
    if bot_status["status"] == "running":
        return jsonify({"success": False, "message": "Bot is already running"})
    
    # Initialize components if needed
    if mt5_connector is None:
        success = initialize_components()
        if not success:
            return jsonify({"success": False, "message": "Failed to initialize components"})
    
    # Update status
    bot_status["status"] = "running"
    bot_status["started_at"] = datetime.now().isoformat()
    bot_status["last_check"] = datetime.now().isoformat()
    
    return jsonify({"success": True, "message": "Bot started successfully"})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_status
    
    if bot_status["status"] == "stopped":
        return jsonify({"success": False, "message": "Bot is already stopped"})
    
    # Update status
    bot_status["status"] = "stopped"
    
    return jsonify({"success": True, "message": "Bot stopped successfully"})

@app.route('/api/account_info')
def account_info():
    """Get MT5 account information"""
    if mt5_connector is None:
        return jsonify({"success": False, "message": "MT5 not connected"})
    
    try:
        account_info = mt5_connector.get_account_info()
        if account_info:
            return jsonify({"success": True, "data": account_info})
        else:
            return jsonify({"success": False, "message": "Failed to get account info"})
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/active_trades')
def active_trades():
    """Get active trades"""
    if order_executor is None:
        return jsonify({"success": False, "message": "Order executor not initialized"})
    
    try:
        positions = order_executor.get_open_positions()
        return jsonify({"success": True, "data": positions})
    except Exception as e:
        logger.error(f"Error getting active trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/trade_history')
def trade_history():
    """Get trade history"""
    if order_executor is None:
        return jsonify({"success": False, "message": "Order executor not initialized"})
    
    try:
        # Get days parameter or default to 7
        days = int(request.args.get('days', '7'))
        
        history = order_executor.get_deal_history(days)
        return jsonify({"success": True, "data": history})
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/performance')
def performance():
    """Get performance metrics"""
    if performance_analyzer is None:
        return jsonify({"success": False, "message": "Performance analyzer not initialized"})
    
    try:
        # Analyze performance
        metrics = performance_analyzer.analyze_performance()
        if metrics:
            return jsonify({"success": True, "data": metrics})
        else:
            return jsonify({"success": False, "message": "No performance data available"})
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/market_data')
def market_data():
    """Get current market data"""
    if data_collector is None:
        return jsonify({"success": False, "message": "Data collector not initialized"})
    
    try:
        # Get parameters
        symbol = request.args.get('symbol', 'EURUSD')
        timeframe = request.args.get('timeframe', 'H1')
        bars = int(request.args.get('bars', '20'))
        
        # Get data
        data = data_collector.get_ohlc_data(symbol, timeframe, bars)
        
        if data is not None:
            # Convert to list of dictionaries
            data_list = data.to_dict('records')
            
            # Format datetime objects
            for item in data_list:
                if 'time' in item and isinstance(item['time'], pd.Timestamp):
                    item['time'] = item['time'].isoformat()
            
            return jsonify({"success": True, "data": data_list})
        else:
            return jsonify({"success": False, "message": "Failed to get market data"})
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/close_trade', methods=['POST'])
def close_trade():
    """Close a specific trade"""
    if order_executor is None:
        return jsonify({"success": False, "message": "Order executor not initialized"})
    
    try:
        # Get ticket from request
        data = request.get_json()
        ticket = data.get('ticket')
        
        if not ticket:
            return jsonify({"success": False, "message": "No ticket provided"})
        
        # Close position
        result = order_executor.close_position(ticket, "Closed from web interface")
        
        if result and result.get('result') == 'success':
            return jsonify({"success": True, "message": f"Position {ticket} closed successfully"})
        else:
            return jsonify({"success": False, "message": f"Failed to close position: {result}"})
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/close_all_trades', methods=['POST'])
def close_all_trades():
    """Close all active trades"""
    if order_executor is None:
        return jsonify({"success": False, "message": "Order executor not initialized"})
    
    try:
        # Get all open positions
        positions = order_executor.get_open_positions()
        
        if not positions:
            return jsonify({"success": True, "message": "No open positions to close"})
        
        # Close each position
        results = []
        success_count = 0
        
        for position in positions:
            ticket = position.get('ticket')
            result = order_executor.close_position(ticket, "Closed all from web interface")
            
            if result and result.get('result') == 'success':
                success_count += 1
                results.append({"ticket": ticket, "status": "closed"})
            else:
                results.append({"ticket": ticket, "status": "failed", "message": str(result)})
        
        return jsonify({
            "success": True, 
            "message": f"Closed {success_count} of {len(positions)} positions",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error closing all trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/trading_config')
def trading_config():
    """Get current trading configuration"""
    return jsonify({"success": True, "data": TRADING_CONFIG})

@app.route('/api/update_config', methods=['POST'])
def update_config():
    """Update trading configuration"""
    try:
        # Get new config from request
        new_config = request.get_json()
        
        if not new_config:
            return jsonify({"success": False, "message": "No configuration provided"})
        
        # Update config
        for key, value in new_config.items():
            if key in TRADING_CONFIG:
                TRADING_CONFIG[key] = value
        
        # Save to file
        config_path = os.path.join(DATA_DIR, "config", "trading_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(TRADING_CONFIG, f, indent=2)
        
        return jsonify({"success": True, "message": "Configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({"success": False, "message": str(e)})

# HTML templates
@app.route('/templates/index.html')
def get_index_template():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Trading Bot Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .dashboard-card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        .stat-card {
            text-align: center;
            padding: 15px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        .stat-label {
            color: #6c757d;
            font-size: 14px;
        }
        .status-running {
            color: #198754;
        }
        .status-stopped {
            color: #dc3545;
        }
        .nav-pills .nav-link.active {
            background-color: #0d6efd;
        }
        .table-responsive {
            max-height: 400px;
            overflow-y: auto;
        }
        #performanceChart, #equityChart {
            width: 100%;
            height: 300px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="bi bi-currency-exchange me-2"></i>
                Forex Trading Bot
            </a>
            <div class="ms-auto d-flex align-items-center">
                <div class="me-3">
                    <span class="text-light">Status:</span>
                    <span id="botStatusBadge" class="badge ms-2">Unknown</span>
                </div>
                <button id="startStopBtn" class="btn btn-success">
                    <i class="bi bi-play-fill"></i> Start
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid py-4">
        <div class="row g-3 mb-4">
            <!-- Stats Cards -->
            <div class="col-md-3">
                <div class="card dashboard-card stat-card">
                    <div class="card-body">
                        <div class="stat-value" id="balanceValue">$0.00</div>
                        <div class="stat-label">Account Balance</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card stat-card">
                    <div class="card-body">
                        <div class="stat-value" id="equityValue">$0.00</div>
                        <div class="stat-label">Equity</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card stat-card">
                    <div class="card-body">
                        <div class="stat-value" id="activeTradesValue">0</div>
                        <div class="stat-label">Active Trades</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card stat-card">
                    <div class="card-body">
                        <div class="stat-value" id="profitValue">$0.00</div>
                        <div class="stat-label">Current Profit</div>
                    </div>
                </div>
            </div>
        </div>

        <ul class="nav nav-pills mb-3" id="dashboard-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="active-trades-tab" data-bs-toggle="pill" data-bs-target="#active-trades" type="button" role="tab">Active Trades</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="trade-history-tab" data-bs-toggle="pill" data-bs-target="#trade-history" type="button" role="tab">Trade History</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="performance-tab" data-bs-toggle="pill" data-bs-target="#performance" type="button" role="tab">Performance</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="settings-tab" data-bs-toggle="pill" data-bs-target="#settings" type="button" role="tab">Settings</button>
            </li>
        </ul>

        <div class="tab-content" id="dashboard-tabContent">
            <!-- Active Trades Tab -->
            <div class="tab-pane fade show active" id="active-trades" role="tabpanel">
                <div class="card dashboard-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="card-title mb-0">Active Trades</h5>
                        <button id="closeAllBtn" class="btn btn-danger btn-sm">
                            <i class="bi bi-x-circle"></i> Close All
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Ticket</th>
                                        <th>Symbol</th>
                                        <th>Type</th>
                                        <th>Open Price</th>
                                        <th>Current Price</th>
                                        <th>SL</th>
                                        <th>TP</th>
                                        <th>Profit</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="activeTradesTable">
                                    <tr>
                                        <td colspan="9" class="text-center">No active trades</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Trade History Tab -->
            <div class="tab-pane fade" id="trade-history" role="tabpanel">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Trade History</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Ticket</th>
                                        <th>Symbol</th>
                                        <th>Type</th>
                                        <th>Open Time</th>
                                        <th>Close Time</th>
                                        <th>Profit</th>
                                    </tr>
                                </thead>
                                <tbody id="tradeHistoryTable">
                                    <tr>
                                        <td colspan="6" class="text-center">No trade history</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Tab -->
            <div class="tab-pane fade" id="performance" role="tabpanel">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Performance Metrics</h5>
                            </div>
                            <div class="card-body">
                                <div id="performanceMetrics" class="row g-3">
                                    <div class="col-6">
                                        <div class="stat-card border rounded">
                                            <div class="stat-value" id="winRateValue">0%</div>
                                            <div class="stat-label">Win Rate</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="stat-card border rounded">
                                            <div class="stat-value" id="profitFactorValue">0</div>
                                            <div class="stat-label">Profit Factor</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="stat-card border rounded">
                                            <div class="stat-value" id="totalTradesValue">0</div>
                                            <div class="stat-label">Total Trades</div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="stat-card border rounded">
                                            <div class="stat-value" id="maxDrawdownValue">0%</div>
                                            <div class="stat-label">Max Drawdown</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Equity Curve</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="equityChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">Performance by Symbol</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="symbolPerformanceChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Settings Tab -->
            <div class="tab-pane fade" id="settings" role="tabpanel">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Trading Settings</h5>
                    </div>
                    <div class="card-body">
                        <form id="settingsForm">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <h6>Risk Parameters</h6>
                                    <div class="mb-3">
                                        <label for="maxRiskPerTrade" class="form-label">Max Risk Per Trade</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="maxRiskPerTrade" step="0.01" min="0.01" max="0.10">
                                            <span class="input-group-text">%</span>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label for="maxDailyRisk" class="form-label">Max Daily Risk</label>
                                        <div class="input-group">
                                            <input type="number" class="form-control" id="maxDailyRisk" step="0.01" min="0.01" max="0.20">
                                            <span class="input-group-text">%</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>Trade Management</h6>
                                    <div class="mb-3 form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="trailingStopEnable">
                                        <label class="form-check-label" for="trailingStopEnable">Enable Trailing Stops</label>
                                    </div>
                                    <div class="mb-3 form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="partialExitEnable">
                                        <label class="form-check-label" for="partialExitEnable">Enable Partial Exits</label>
                                    </div>
                                    <div class="mb-3 form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="breakEvenEnable">
                                        <label class="form-check-label" for="breakEvenEnable">Enable Break-Even Stops</label>
                                    </div>
                                </div>
                            </div>
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">Save Settings</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="card dashboard-card mt-4">
                    <div class="card-header">
                        <h5 class="card-title mb-0">System Information</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Bot Status:</strong> <span id="botStatusInfo">Unknown</span></p>
                                <p><strong>Uptime:</strong> <span id="uptimeValue">0:00:00</span></p>
                                <p><strong>Last Check:</strong> <span id="lastCheckValue">Never</span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>API Calls:</strong> <span id="apiCallsValue">0</span></p>
                                <p><strong>API Cost:</strong> $<span id="apiCostValue">0.00</span></p>
                                <p><strong>Trades Executed:</strong> <span id="tradesExecutedValue">0</span></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Global variables
        let equityChart = null;
        let symbolPerformanceChart = null;
        
        // Update dashboard function
        function updateDashboard() {
            // Get bot status
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status badge
                    const statusBadge = document.getElementById('botStatusBadge');
                    const statusInfo = document.getElementById('botStatusInfo');
                    
                    if (data.status === 'running') {
                        statusBadge.className = 'badge bg-success ms-2';
                        statusBadge.textContent = 'Running';
                        statusInfo.className = 'status-running';
                        statusInfo.textContent = 'Running';
                        
                        // Update start/stop button
                        const startStopBtn = document.getElementById('startStopBtn');
                        startStopBtn.className = 'btn btn-danger';
                        startStopBtn.innerHTML = '<i class="bi bi-stop-fill"></i> Stop';
                    } else {
                        statusBadge.className = 'badge bg-danger ms-2';
                        statusBadge.textContent = 'Stopped';
                        statusInfo.className = 'status-stopped';
                        statusInfo.textContent = 'Stopped';
                        
                        // Update start/stop button
                        const startStopBtn = document.getElementById('startStopBtn');
                        startStopBtn.className = 'btn btn-success';
                        startStopBtn.innerHTML = '<i class="bi bi-play-fill"></i> Start';
                    }
                    
                    // Update system info
                    document.getElementById('apiCallsValue').textContent = data.api_calls;
                    document.getElementById('apiCostValue').textContent = data.api_cost.toFixed(2);
                    document.getElementById('tradesExecutedValue').textContent = data.trades_executed;
                    document.getElementById('activeTradesValue').textContent = data.trades_active;
                    
                    // Update uptime
                    if (data.uptime > 0) {
                        const uptime = new Date(data.uptime * 1000).toISOString().substr(11, 8);
                        document.getElementById('uptimeValue').textContent = uptime;
                    }
                    
                    // Update last check
                    if (data.last_check) {
                        const lastCheck = new Date(data.last_check);
                        document.getElementById('lastCheckValue').textContent = lastCheck.toLocaleString();
                    }
                })
                .catch(error => console.error('Error fetching bot status:', error));
                
            // Get account info
            fetch('/api/account_info')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const accountInfo = data.data;
                        document.getElementById('balanceValue').textContent = '$' + accountInfo.balance.toFixed(2);
                        document.getElementById('equityValue').textContent = '$' + accountInfo.equity.toFixed(2);
                        document.getElementById('profitValue').textContent = '$' + accountInfo.profit.toFixed(2);
                        
                        // Set text color for profit
                        const profitValue = document.getElementById('profitValue');
                        if (accountInfo.profit > 0) {
                            profitValue.className = 'stat-value text-success';
                        } else if (accountInfo.profit < 0) {
                            profitValue.className = 'stat-value text-danger';
                        } else {
                            profitValue.className = 'stat-value';
                        }
                    }
                })
                .catch(error => console.error('Error fetching account info:', error));
                
            // Get active trades
            fetch('/api/active_trades')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const trades = data.data;
                        const tableBody = document.getElementById('activeTradesTable');
                        
                        if (trades.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="9" class="text-center">No active trades</td></tr>';
                            return;
                        }
                        
                        tableBody.innerHTML = '';
                        
                        trades.forEach(trade => {
                            const row = document.createElement('tr');
                            
                            // Format profit with color
                            const profitClass = trade.profit > 0 ? 'text-success' : (trade.profit < 0 ? 'text-danger' : '');
                            
                            row.innerHTML = `
                                <td>${trade.ticket}</td>
                                <td>${trade.symbol}</td>
                                <td>${trade.type.toUpperCase()}</td>
                                <td>${trade.open_price.toFixed(5)}</td>
                                <td>${trade.current_price.toFixed(5)}</td>
                                <td>${trade.stop_loss > 0 ? trade.stop_loss.toFixed(5) : '-'}</td>
                                <td>${trade.take_profit > 0 ? trade.take_profit.toFixed(5) : '-'}</td>
                                <td class="${profitClass}">${trade.profit.toFixed(2)}</td>
                                <td>
                                    <button class="btn btn-sm btn-danger close-trade-btn" data-ticket="${trade.ticket}">
                                        <i class="bi bi-x"></i>
                                    </button>
                                </td>
                            `;
                            
                            tableBody.appendChild(row);
                        });
                        
                        // Add event listeners for close buttons
                        document.querySelectorAll('.close-trade-btn').forEach(button => {
                            button.addEventListener('click', function() {
                                const ticket = this.getAttribute('data-ticket');
                                closeTrade(ticket);
                            });
                        });
                    }
                })
                .catch(error => console.error('Error fetching active trades:', error));
        }
        
        // Function to load trade history
        function loadTradeHistory() {
            fetch('/api/trade_history')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const history = data.data;
                        const tableBody = document.getElementById('tradeHistoryTable');
                        
                        if (history.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="6" class="text-center">No trade history</td></tr>';
                            return;
                        }
                        
                        tableBody.innerHTML = '';
                        
                        history.forEach(trade => {
                            const row = document.createElement('tr');
                            
                            // Format profit with color
                            const profitClass = trade.profit > 0 ? 'text-success' : (trade.profit < 0 ? 'text-danger' : '');
                            
                            row.innerHTML = `
                                <td>${trade.ticket}</td>
                                <td>${trade.symbol}</td>
                                <td>${trade.type.toUpperCase()}</td>
                                <td>${trade.time}</td>
                                <td>${trade.time}</td>
                                <td class="${profitClass}">${trade.profit.toFixed(2)}</td>
                            `;
                            
                            tableBody.appendChild(row);
                        });
                    }
                })
                .catch(error => console.error('Error fetching trade history:', error));
        }
        
        // Function to load performance data
        function loadPerformanceData() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const metrics = data.data;
                        
                        // Update performance metrics
                        document.getElementById('winRateValue').textContent = (metrics.win_rate * 100).toFixed(1) + '%';
                        document.getElementById('profitFactorValue').textContent = metrics.profit_factor.toFixed(2);
                        document.getElementById('totalTradesValue').textContent = metrics.total_trades;
                        document.getElementById('maxDrawdownValue').textContent = (metrics.max_drawdown * 100).toFixed(1) + '%';
                        
                        // Update equity chart
                        updateEquityChart(metrics.equity_curve);
                        
                        // Update symbol performance chart
                        updateSymbolPerformanceChart(metrics.symbol_performance);
                    }
                })
                .catch(error => console.error('Error fetching performance data:', error));
        }
        
        // Function to update equity chart
        function updateEquityChart(equityData) {
            if (!equityData || equityData.length === 0) return;
            
            const ctx = document.getElementById('equityChart').getContext('2d');
            
            // Prepare data
            const labels = equityData.map(point => new Date(point.time).toLocaleDateString());
            const equityValues = equityData.map(point => point.equity);
            const balanceValues = equityData.map(point => point.balance);
            
            // Destroy existing chart if it exists
            if (equityChart) {
                equityChart.destroy();
            }
            
            // Create new chart
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Equity',
                            data: equityValues,
                            borderColor: 'blue',
                            backgroundColor: 'rgba(0, 0, 255, 0.1)',
                            tension: 0.1,
                            fill: false
                        },
                        {
                            label: 'Balance',
                            data: balanceValues,
                            borderColor: 'green',
                            backgroundColor: 'rgba(0, 128, 0, 0.1)',
                            tension: 0.1,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Equity Curve'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Function to update symbol performance chart
        function updateSymbolPerformanceChart(symbolData) {
            if (!symbolData || Object.keys(symbolData).length === 0) return;
            
            const ctx = document.getElementById('symbolPerformanceChart').getContext('2d');
            
            // Prepare data
            const symbols = Object.keys(symbolData);
            const totalPips = symbols.map(symbol => symbolData[symbol].total_pips);
            const winRates = symbols.map(symbol => symbolData[symbol].win_rate * 100);
            
            // Destroy existing chart if it exists
            if (symbolPerformanceChart) {
                symbolPerformanceChart.destroy();
            }
            
            // Create new chart
            symbolPerformanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: symbols,
                    datasets: [
                        {
                            label: 'Total Pips',
                            data: totalPips,
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Win Rate (%)',
                            data: winRates,
                            type: 'line',
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Performance by Symbol'
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Total Pips'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            min: 0,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Win Rate (%)'
                            }
                        }
                    }
                }
            });
        }
        
        // Function to load trading configuration
        function loadTradingConfig() {
            fetch('/api/trading_config')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const config = data.data;
                        
                        // Set form values
                        document.getElementById('maxRiskPerTrade').value = config.max_risk_per_trade * 100;
                        document.getElementById('maxDailyRisk').value = config.max_daily_risk * 100;
                        
                        // Set checkboxes
                        document.getElementById('trailingStopEnable').checked = config.trailing_stop?.enable ?? false;
                        document.getElementById('partialExitEnable').checked = config.partial_exit?.enable ?? false;
                        document.getElementById('breakEvenEnable').checked = config.breakeven?.enable ?? false;
                    }
                })
                .catch(error => console.error('Error fetching trading config:', error));
        }
        
        // Function to start or stop the bot
        function toggleBotStatus() {
            const statusBadge = document.getElementById('botStatusBadge');
            const isRunning = statusBadge.textContent === 'Running';
            
            const endpoint = isRunning ? '/api/stop_bot' : '/api/start_bot';
            
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateDashboard();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => console.error('Error toggling bot status:', error));
        }
        
        // Function to close a specific trade
        function closeTrade(ticket) {
            if (!confirm(`Are you sure you want to close trade #${ticket}?`)) {
                return;
            }
            
            fetch('/api/close_trade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticket: ticket })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Trade closed successfully');
                        updateDashboard();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => console.error('Error closing trade:', error));
        }
        
        // Function to close all trades
        function closeAllTrades() {
            if (!confirm('Are you sure you want to close all open trades?')) {
                return;
            }
            
            fetch('/api/close_all_trades', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(data.message);
                        updateDashboard();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => console.error('Error closing all trades:', error));
        }
        
        // Function to save settings
        function saveSettings(event) {
            event.preventDefault();
            
            const config = {
                max_risk_per_trade: document.getElementById('maxRiskPerTrade').value / 100,
                max_daily_risk: document.getElementById('maxDailyRisk').value / 100,
                trailing_stop: {
                    enable: document.getElementById('trailingStopEnable').checked
                },
                partial_exit: {
                    enable: document.getElementById('partialExitEnable').checked
                },
                breakeven: {
                    enable: document.getElementById('breakEvenEnable').checked
                }
            };
            
            fetch('/api/update_config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Settings saved successfully');
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => console.error('Error saving settings:', error));
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            // Initial update
            updateDashboard();
            
            // Set up event handlers
            const startStopBtn = document.getElementById('startStopBtn');
            startStopBtn.addEventListener('click', toggleBotStatus);
            
            const closeAllBtn = document.getElementById('closeAllBtn');
            closeAllBtn.addEventListener('click', closeAllTrades);
            
            const settingsForm = document.getElementById('settingsForm');
            settingsForm.addEventListener('submit', saveSettings);
            
            // Set up tab event handlers
            document.getElementById('trade-history-tab').addEventListener('click', loadTradeHistory);
            document.getElementById('performance-tab').addEventListener('click', loadPerformanceData);
            document.getElementById('settings-tab').addEventListener('click', loadTradingConfig);
            
            // Set up automatic refresh
            setInterval(updateDashboard, 5000);
        });
    </script>
</body>
</html>
    """

# Run the app
if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)