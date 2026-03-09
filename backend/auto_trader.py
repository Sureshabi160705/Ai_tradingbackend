"""
Automatic Trading Mode - Analyzes indices and recommends best options for trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data_fetcher import get_index_data, get_option_chain, get_live_price
from indicators import add_indicators
from candlestick_patterns import detect_patterns
from option_chain import process_option_chain
import json
import threading
from functools import lru_cache

class AutoTrader:
    def __init__(self):
        self.last_update = None
        self.ist = pytz.timezone('Asia/Kolkata')
        self.cached_predictions = None
        self.cache_time = None
        self.cache_duration = 30  # Cache for 30 seconds
        self.lock = threading.Lock()
        
    def get_auto_predictions(self):
        """
        Get automatic predictions for all major indices
        Returns trading recommendations with best options
        Uses caching to avoid repeated API calls
        """
        # Check if cache is still valid
        now = datetime.now()
        if self.cached_predictions and self.cache_time:
            age = (now - self.cache_time).total_seconds()
            if age < self.cache_duration:
                return self.cached_predictions
        
        # Fetch fresh predictions
        with self.lock:
            indices = {
                'nifty': '^NSEI',
                'banknifty': '^NSEBANK',
                'sensex': '^BSESN'
            }
            
            predictions = []
            
            for index_name, symbol in indices.items():
                try:
                    prediction = self.analyze_index(index_name, symbol)
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    print(f"Error analyzing {index_name}: {e}")
                    # Return cached version if analysis fails
                    if self.cached_predictions:
                        return self.cached_predictions
                    continue
            
            # Cache the new predictions
            self.cached_predictions = predictions
            self.cache_time = now
            
            return predictions
    
    def analyze_index(self, index_name, symbol):
        """
        Analyze a single index and return recommendation
        """
        try:
            # Get candlestick data
            data = get_index_data(symbol, interval='5m', period='10d')
            
            if data.empty:
                return None
            
            # Filter to market hours
            ist = pytz.timezone('Asia/Kolkata')
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            if data['Datetime'].dt.tz is None:
                data['Datetime'] = data['Datetime'].dt.tz_localize('Asia/Kolkata')
            
            market_start = data['Datetime'].dt.time >= pd.Timestamp('09:15:00').time()
            market_end = data['Datetime'].dt.time <= pd.Timestamp('15:30:00').time()
            data = data[market_start & market_end]
            
            if data.empty:
                return None
            
            # Add technical indicators to all historical data (need at least 50 candles for indicators)
            data = add_indicators(data)
            
            # Detect candlestick patterns
            data = detect_patterns(data)
            
            # Get latest trading day data
            latest_date = data['Datetime'].dt.date.max()
            today_data = data[data['Datetime'].dt.date == latest_date].copy()
            
            if today_data.empty:
                return None
            
            # Get the latest candle
            latest = today_data.iloc[-1]
            current_price = latest['Close']
            
            # Analyze trend and patterns using all data (with indicators)
            trend, confidence = self.analyze_trend(data)
            
            # Get current live price
            try:
                live_price = get_live_price(symbol)
            except:
                live_price = current_price
            
            # Get best options
            best_options = self.get_best_options_for_index(
                index_name, symbol, live_price, trend
            )
            
            # Determine recommended trade
            trade_recommendation = self.get_trade_recommendation(trend, data)
            
            # Get best strike price
            best_strike = self.get_best_strike(best_options)
            
            result = {
                'index': index_name.upper(),
                'current_price': round(live_price, 2),
                'prediction': trend,
                'confidence': round(confidence, 1),
                'recommended_trade': trade_recommendation,
                'best_strike': best_strike,
                'best_options': best_options[:5],  # Top 5 options
                'indicators': {
                    'rsi': round(latest.get('RSI', 50), 2),
                    'macd': round(latest.get('MACD', 0), 4),
                    'bb_position': self.get_bb_position(latest),
                    'moving_averages_trend': self.get_ma_trend(latest)
                },
                'patterns': self.get_recent_patterns(data),
                'timestamp': datetime.now(ist).isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing index {index_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_trend(self, data):
        """
        Analyze trend direction and confidence
        """
        if len(data) < 5:
            return "HOLD", 50.0
        
        latest = data.iloc[-1]
        
        # Analyze recent candles
        recent = data.tail(5)
        close_trend = recent['Close'].diff().sum()
        
        # Technical indicators
        rsi = latest.get('RSI', 50)
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_SIGNAL', 0)
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        
        trend_score = 0
        confidence = 50
        
        # Price above moving averages - bullish
        if latest['Close'] > sma_20 > sma_50:
            trend_score += 3
            confidence += 20
        elif latest['Close'] < sma_20 < sma_50:
            trend_score -= 3
            confidence += 20
        elif latest['Close'] > sma_20:
            trend_score += 1
            confidence += 10
        elif latest['Close'] < sma_20:
            trend_score -= 1
            confidence += 10
        
        # RSI analysis
        if rsi > 70:
            trend_score -= 2  # Overbought - potential reversal
            confidence += 15
        elif rsi < 30:
            trend_score += 2  # Oversold - potential reversal
            confidence += 15
        elif rsi > 60:
            trend_score += 1  # Strong
        elif rsi < 40:
            trend_score -= 1  # Weak
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            trend_score += 2
            confidence += 15
        elif macd < macd_signal and macd < 0:
            trend_score -= 2
            confidence += 15
        elif macd > macd_signal:
            trend_score += 1
            confidence += 10
        elif macd < macd_signal:
            trend_score -= 1
            confidence += 10
        
        # Candle pattern analysis
        if latest.get('ENGULFING') == 100 or latest.get('HAMMER') == 100:
            trend_score += 2
            confidence += 15
        elif latest.get('MORNING_STAR') == 100:
            trend_score += 3
            confidence += 20
        elif latest.get('ENGULFING') == -100 or latest.get('SHOOTING_STAR') == -100:
            trend_score -= 2
            confidence += 15
        elif latest.get('EVENING_STAR') == -100:
            trend_score -= 3
            confidence += 20
        
        # Close trend
        if close_trend > 0:
            trend_score += 1
            confidence += 5
        elif close_trend < 0:
            trend_score -= 1
            confidence += 5
        
        # Recent volatility
        volatility = (latest['High'] - latest['Low']) / latest['Close'] * 100
        if volatility > 2:  # High volatility
            confidence += 5  # More conviction in trend
        
        confidence = min(confidence, 95)
        confidence = max(confidence, 50)
        
        if trend_score > 2:
            trend = "BULLISH"
        elif trend_score < -2:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        return trend, confidence
    
    def get_trade_recommendation(self, trend, data):
        """
        Get specific trade recommendation based on trend
        """
        latest = data.iloc[-1]
        current_price = latest['Close']
        atr = latest.get('ATR', (latest['High'] - latest['Low']) / 2)
        
        if trend == "BULLISH":
            # Find ATM CALL strike
            base_strike = int((current_price // 100)) * 100
            call_strike = base_strike + 100 if current_price > base_strike + 50 else base_strike
            return f"BUY CALL @ {call_strike}"
        elif trend == "BEARISH":
            # Find ATM PUT strike
            base_strike = int((current_price // 100)) * 100
            put_strike = base_strike if current_price < base_strike + 50 else base_strike + 100
            return f"BUY PUT @ {put_strike}"
        else:
            return "HOLD / STRADDLE"
    
    def get_best_options_for_index(self, index_name, symbol, current_price, trend):
        """
        Get best options for trading based on current price and trend
        """
        try:
            # Get option chain
            option_data = get_option_chain(index_name.upper().replace('SENSEX', 'SENSEX'))
            
            if not option_data or 'records' not in option_data:
                print(f"No option data for {index_name}")
                return []
            
            # Process option chain
            processed = process_option_chain(option_data)
            
            if not processed or 'strikes' not in processed:
                print(f"No processed strikes for {index_name}")
                return []
            
            best_options = []
            
            # Analyze strikes based on trend
            for strike_info in processed['strikes'][:30]:  # Check top 30 strikes
                strike_price = strike_info.get('strikePrice', 0)
                
                # Get CE and PE prices
                ce_data = strike_info.get('CE', {})
                pe_data = strike_info.get('PE', {})
                
                ce_price = ce_data.get('lastPrice', 0) or ce_data.get('bidPrice', 0) or 0
                pe_price = pe_data.get('lastPrice', 0) or pe_data.get('bidPrice', 0) or 0
                
                ce_oi = ce_data.get('openInterest', 0) or 0
                pe_oi = pe_data.get('openInterest', 0) or 0
                
                # For BULLISH trend: prioritize CALL options
                if trend == "BULLISH" and ce_price > 0:
                    # Find strikes ATM and slightly above
                    if strike_price >= current_price:
                        profit_potential = (current_price - strike_price) + ce_price
                        if profit_potential >= 10:  # Min 10 points profit
                            best_options.append({
                                'option_type': 'CE',
                                'strike': int(strike_price),
                                'current_price': round(ce_price, 2),
                                'profit_potential': round(profit_potential, 2),
                                'oi': int(ce_oi),
                                'option_name': f"{index_name.upper()} {strike_price} CE",
                                'implied_volatility': ce_data.get('impliedVolatility', 0) or 0
                            })
                
                # For BEARISH trend: prioritize PUT options
                elif trend == "BEARISH" and pe_price > 0:
                    # Find strikes ATM and slightly below
                    if strike_price <= current_price:
                        profit_potential = (strike_price - current_price) + pe_price
                        if profit_potential >= 10:  # Min 10 points profit
                            best_options.append({
                                'option_type': 'PE',
                                'strike': int(strike_price),
                                'current_price': round(pe_price, 2),
                                'profit_potential': round(profit_potential, 2),
                                'oi': int(pe_oi),
                                'option_name': f"{index_name.upper()} {strike_price} PE",
                                'implied_volatility': pe_data.get('impliedVolatility', 0) or 0
                            })
                
                # For NEUTRAL: both options
                elif trend == "NEUTRAL":
                    if ce_price > 0 and strike_price >= current_price:
                        profit_potential = (current_price - strike_price) + ce_price
                        if profit_potential >= 10:
                            best_options.append({
                                'option_type': 'CE',
                                'strike': int(strike_price),
                                'current_price': round(ce_price, 2),
                                'profit_potential': round(profit_potential, 2),
                                'oi': int(ce_oi),
                                'option_name': f"{index_name.upper()} {strike_price} CE",
                                'implied_volatility': ce_data.get('impliedVolatility', 0) or 0
                            })
                    
                    if pe_price > 0 and strike_price <= current_price:
                        profit_potential = (strike_price - current_price) + pe_price
                        if profit_potential >= 10:
                            best_options.append({
                                'option_type': 'PE',
                                'strike': int(strike_price),
                                'current_price': round(pe_price, 2),
                                'profit_potential': round(profit_potential, 2),
                                'oi': int(pe_oi),
                                'option_name': f"{index_name.upper()} {strike_price} PE",
                                'implied_volatility': pe_data.get('impliedVolatility', 0) or 0
                            })
            
            # Sort by profit potential
            best_options.sort(key=lambda x: x['profit_potential'], reverse=True)
            
            return best_options
        
        except Exception as e:
            print(f"Error getting best options for {index_name}: {e}")
            return []
    
    def get_best_strike(self, options):
        """
        Get the best strike from options list
        """
        if not options:
            return "-"
        
        # Prefer high profit potential with good OI
        best = max(options, key=lambda x: (x['profit_potential'], x['oi']))
        return f"{best['strike']} {best['option_type']}"
    
    def get_bb_position(self, row):
        """
        Determine position relative to Bollinger Bands
        """
        close = row['Close']
        bb_upper = row.get('BB_UPPER', close)
        bb_lower = row.get('BB_LOWER', close)
        bb_middle = row.get('SMA_20', close)
        
        if close > bb_upper:
            return "Above Upper"
        elif close < bb_lower:
            return "Below Lower"
        elif close > bb_middle:
            return "Upper Half"
        else:
            return "Lower Half"
    
    def get_ma_trend(self, row):
        """
        Determine moving average trend
        """
        close = row['Close']
        sma_20 = row.get('SMA_20', close)
        sma_50 = row.get('SMA_50', close)
        ema_12 = row.get('EMA_12', close)
        
        if sma_20 > sma_50 > ema_12:
            return "Strongly Bullish"
        elif sma_20 > sma_50:
            return "Bullish"
        elif sma_20 < sma_50 < ema_12:
            return "Strongly Bearish"
        elif sma_20 < sma_50:
            return "Bearish"
        else:
            return "Mixed"
    
    def get_recent_patterns(self, data):
        """
        Get recent candlestick patterns detected
        """
        patterns = []
        
        for idx in range(max(0, len(data) - 3), len(data)):
            row = data.iloc[idx]
            
            pattern_map = {
                'ENGULFING': ('Engulfing', 100),
                'HAMMER': ('Hammer', 100),
                'SHOOTING_STAR': ('Shooting Star', -100),
                'MORNING_STAR': ('Morning Star', 100),
                'EVENING_STAR': ('Evening Star', -100),
                'THREE_WHITE_SOLDIERS': ('Three White Soldiers', 100),
                'THREE_BLACK_CROWS': ('Three Black Crows', -100),
            }
            
            for key, (name, signal_value) in pattern_map.items():
                if key in row and row[key] == signal_value:
                    patterns.append({
                        'name': name,
                        'type': 'Bullish' if signal_value > 0 else 'Bearish',
                        'time': row['Datetime'].isoformat() if 'Datetime' in row else None
                    })
        
        return patterns


# Singleton instance
auto_trader = AutoTrader()

def get_auto_predictions():
    """Get auto predictions"""
    return auto_trader.get_auto_predictions()
