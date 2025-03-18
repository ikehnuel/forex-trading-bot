import json
import logging
import time
import anthropic
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLAUDE_API_KEY, CLAUDE_MODEL, API_OPTIMIZATION, logger

class ClaudeAnalyzer:
    def __init__(self, api_key: str = CLAUDE_API_KEY, model: str = CLAUDE_MODEL):
        """
        Initialize the Claude Analyzer with API credentials and settings
        
        Args:
            api_key: Claude API key
            model: Claude model to use (default from config)
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.base_prompt_template = self._get_base_prompt_template()
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        }
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.credit_optimization_enabled = API_OPTIMIZATION['enable']
        self.max_candles = API_OPTIMIZATION['max_candles']
        
    def _get_base_prompt_template(self) -> str:
        """
        Define the base system prompt template for Claude
        
        Returns:
            str: System prompt template
        """
        return """
        You are a forex trading assistant that analyzes market data and provides trading recommendations.
        Your task is to analyze the provided OHLC data, indicators, and market context for {symbol} on the {timeframe} timeframe.
        
        # Current Market Context
        - Account Balance: {account_balance}
        - Open Positions: {open_positions}
        - Current Market Regime: {market_regime}
        - Timeframe Alignment Score: {timeframe_alignment}
        
        # Trading Rules
        1. Only recommend trades when there is strong confluence across multiple timeframes
        2. Adapt strategy based on detected market regime
        3. Include precise stop loss and take profit levels with each recommendation
        4. Consider existing positions and overall exposure
        5. Recommend position size based on volatility and risk parameters
        6. It's completely acceptable to recommend no action if conditions aren't optimal
        
        # Response Format
        Provide your analysis and trading recommendation in strict JSON format:
        ```
        {
            "market_analysis": {
                "trend": "bullish|bearish|neutral",
                "strength": 0-100,
                "support_levels": [level1, level2],
                "resistance_levels": [level1, level2],
                "key_observations": ["observation1", "observation2"]
            },
            "trade_recommendation": {
                "action": "BUY|SELL|HOLD|CLOSE",
                "confidence": 0-100,
                "reasoning": "brief explanation",
                "entry_price": numeric or null,
                "stop_loss": numeric or null,
                "take_profit": [level1, level2],
                "position_size": numeric or null,
                "close_tickets": [list of position tickets] or null,
                "risk_reward_ratio": numeric
            }
        }
        ```
        
        Provide only the JSON response with no additional text.
        """
    
    def analyze_market_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           ohlc_data: List[Dict], 
                           indicators: Dict, 
                           market_context: Dict,
                           existing_positions: List[Dict] = None) -> Dict:
        """
        Send market data to Claude for analysis and trading recommendations
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            timeframe: Chart timeframe (e.g., 'H1')
            ohlc_data: List of OHLC candles
            indicators: Dictionary of technical indicators
            market_context: Additional market context information
            existing_positions: List of currently open positions
        
        Returns:
            Dict: Claude's analysis and trade recommendation
        """
        try:
            # Prepare data for Claude (optimize token usage)
            data = self._prepare_data_for_claude(
                symbol, timeframe, ohlc_data, indicators, market_context, existing_positions
            )
            
            # Format prompt with template
            system_prompt = self._format_system_prompt(symbol, timeframe, market_context)
            user_message = f"Here is the current forex market data to analyze:\n\n{json.dumps(data, indent=2)}\n\nBased on this data, please provide your trading recommendation in the required JSON format."
            
            # Rate limiting and token optimization
            self._apply_rate_limiting()
            
            # Send request to Claude
            start_time = time.time()
            self.last_request_time = datetime.now()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Track usage
            self._update_token_usage(response)
            self.request_count += 1
            
            # Process and validate response
            content = response.content[0].text
            trading_advice = self._parse_claude_response(content)
            
            # Log request details
            self.logger.info(f"Claude analysis for {symbol}/{timeframe} completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"Token usage: {response.usage.input_tokens} input, {response.usage.output_tokens} output")
            
            return trading_advice
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error getting response from Claude API: {e}")
            return {
                "error": str(e),
                "market_analysis": {"trend": "unknown", "strength": 0},
                "trade_recommendation": {"action": "HOLD", "confidence": 0, "reasoning": "API error"}
            }
    
    def _prepare_data_for_claude(self, 
                               symbol: str, 
                               timeframe: str, 
                               ohlc_data: List[Dict], 
                               indicators: Dict, 
                               market_context: Dict,
                               existing_positions: List[Dict] = None) -> Dict:
        """
        Optimize data for Claude to reduce token usage
        
        Returns:
            Dict: Formatted data for Claude
        """
        # Apply token optimization if enabled
        if self.credit_optimization_enabled:
            # Limit number of candles (focus on most recent)
            max_candles = self.max_candles
            recent_candles = ohlc_data[-max_candles:] if len(ohlc_data) > max_candles else ohlc_data
            
            # Simplify candle data (round values, remove unnecessary fields)
            simplified_candles = []
            for candle in recent_candles:
                simplified_candles.append({
                    "time": candle["time"],
                    "open": round(candle["open"], 5),
                    "high": round(candle["high"], 5),
                    "low": round(candle["low"], 5),
                    "close": round(candle["close"], 5),
                    "volume": int(candle["volume"]) if "volume" in candle else 0
                })
            
            # Only include key indicators
            key_indicators = {}
            for indicator, values in indicators.items():
                # Only include latest values for most indicators
                if isinstance(values, list) and len(values) > 5:
                    key_indicators[indicator] = values[-5:]
                else:
                    key_indicators[indicator] = values
        else:
            simplified_candles = ohlc_data
            key_indicators = indicators
            
        # Format positions data
        position_data = []
        if existing_positions:
            for pos in existing_positions:
                position_data.append({
                    "ticket": pos.get("ticket"),
                    "type": pos.get("type"),
                    "volume": pos.get("volume"),
                    "open_price": pos.get("open_price"),
                    "current_price": pos.get("current_price"),
                    "profit": pos.get("profit"),
                    "swap": pos.get("swap", 0)
                })
        
        # Create the final data structure
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ohlc_data": simplified_candles,
            "indicators": key_indicators,
            "market_context": market_context,
            "positions": position_data
        }
        
        return data
    
    def _format_system_prompt(self, symbol: str, timeframe: str, market_context: Dict) -> str:
        """
        Format the system prompt with current context
        
        Returns:
            str: Formatted system prompt
        """
        return self.base_prompt_template.format(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=market_context.get("account_balance", "Unknown"),
            open_positions=len(market_context.get("open_positions", [])),
            market_regime=market_context.get("market_regime", "Unknown"),
            timeframe_alignment=market_context.get("timeframe_alignment", "Unknown")
        )
    
    def _parse_claude_response(self, response_text: str) -> Dict:
        """
        Parse and validate Claude's JSON response
        
        Args:
            response_text: Raw text response from Claude
            
        Returns:
            Dict: Parsed and validated response
        """
        try:
            # Extract JSON from response (handle cases where Claude adds markdown formatting)
            json_str = response_text
            if "```" in response_text:
                parts = response_text.split("```")
                for part in parts:
                    if part.strip().startswith("{") or part.strip().startswith("json"):
                        json_str = part.strip()
                        if json_str.startswith("json"):
                            json_str = json_str[4:].strip()
                        break
            
            # Parse JSON
            trading_advice = json.loads(json_str)
            
            # Validate required fields
            required_sections = ["market_analysis", "trade_recommendation"]
            for section in required_sections:
                if section not in trading_advice:
                    trading_advice[section] = {}
            
            # Ensure recommendation has required fields
            if "action" not in trading_advice["trade_recommendation"]:
                trading_advice["trade_recommendation"]["action"] = "HOLD"
                trading_advice["trade_recommendation"]["confidence"] = 0
                trading_advice["trade_recommendation"]["reasoning"] = "Incomplete analysis"
                
            return trading_advice
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Claude response: {e}\nResponse: {response_text}")
            return {
                "error": "Failed to parse response",
                "market_analysis": {"trend": "unknown", "strength": 0},
                "trade_recommendation": {"action": "HOLD", "confidence": 0, "reasoning": "Failed to parse response"}
            }
    
    def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to avoid API limits"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 1.0:  # Minimum 1 second between requests
                time.sleep(1.0 - elapsed)
    
    def _update_token_usage(self, response) -> None:
        """Update token usage tracking"""
        # Claude API pricing (approximate)
        input_cost_per_1k = 0.015  # $0.015 per 1K input tokens
        output_cost_per_1k = 0.060  # $0.060 per 1K output tokens
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        self.token_usage["prompt_tokens"] += input_tokens
        self.token_usage["completion_tokens"] += output_tokens
        self.token_usage["total_tokens"] += input_tokens + output_tokens
        
        # Calculate estimated cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        self.token_usage["estimated_cost"] += input_cost + output_cost
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            "token_usage": self.token_usage,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_request_time": self.last_request_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_request_time else None
        }
    
    def update_base_prompt(self, new_template: str) -> None:
        """Update the base prompt template"""
        self.base_prompt_template = new_template
        self.logger.info("Base prompt template updated")
    
    def set_credit_optimization(self, enabled: bool) -> None:
        """Enable or disable credit optimization"""
        self.credit_optimization_enabled = enabled
        self.logger.info(f"Credit optimization {'enabled' if enabled else 'disabled'}")