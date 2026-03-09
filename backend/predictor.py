import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime, timedelta

class OptionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = 'models/option_predictor.pkl'
        self.scaler_path = 'models/scaler.pkl'
        os.makedirs('models', exist_ok=True)

    def prepare_features(self, data):
        """
        Prepare features for ML model
        """
        features = []

        for i in range(len(data)):
            if i < 50:  # Need enough historical data
                features.append([0] * 20)  # Placeholder
                continue

            row = data.iloc[i]

            # Technical indicators
            rsi = row.get('RSI', 50)
            macd = row.get('MACD', 0)
            macd_signal = row.get('MACD_SIGNAL', 0)
            bb_upper = row.get('BB_UPPER', row['Close'])
            bb_lower = row.get('BB_LOWER', row['Close'])
            sma_20 = row.get('SMA_20', row['Close'])
            sma_50 = row.get('SMA_50', row['Close'])
            ema_12 = row.get('EMA_12', row['Close'])
            ema_26 = row.get('EMA_26', row['Close'])
            stoch_k = row.get('STOCH_K', 50)
            stoch_d = row.get('STOCH_D', 50)
            atr = row.get('ATR', 0)
            willr = row.get('WILLR', -50)
            cci = row.get('CCI', 0)
            adx = row.get('ADX', 25)

            # Price action
            close = row['Close']
            high = row['High']
            low = row['Low']
            open_price = row['Open']

            # Recent price changes
            prev_close = data.iloc[i-1]['Close'] if i > 0 else close
            price_change = (close - prev_close) / prev_close * 100

            # Volatility
            volatility = (high - low) / close * 100

            # Volume
            volume = row.get('Volume', 0)

            # Candlestick patterns (simplified)
            pattern_score = 0
            if 'ENGULFING' in row and row['ENGULFING'] == 100:
                pattern_score += 1
            if 'HAMMER' in row and row['HAMMER'] == 100:
                pattern_score += 1
            if 'MORNING_STAR' in row and row['MORNING_STAR'] == 100:
                pattern_score += 1
            if 'ENGULFING' in row and row['ENGULFING'] == -100:
                pattern_score -= 1
            if 'SHOOTING_STAR' in row and row['SHOOTING_STAR'] == -100:
                pattern_score -= 1
            if 'EVENING_STAR' in row and row['EVENING_STAR'] == -100:
                pattern_score -= 1

            feature_vector = [
                rsi, macd, macd_signal,
                (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0,
                (close - sma_20) / sma_20 * 100,
                (close - sma_50) / sma_50 * 100,
                (ema_12 - ema_26) / ema_26 * 100,
                stoch_k, stoch_d,
                atr / close * 100,
                willr, cci, adx,
                price_change, volatility,
                np.log(volume + 1),
                pattern_score
            ]

            features.append(feature_vector)

        return np.array(features)

    def create_labels(self, data, future_periods=5):
        """
        Create labels based on future price movement
        """
        labels = []

        for i in range(len(data)):
            if i >= len(data) - future_periods:
                labels.append(0)  # Neutral for last few rows
                continue

            current_price = data.iloc[i]['Close']
            future_prices = data.iloc[i:i+future_periods]['Close']
            max_future = future_prices.max()
            min_future = future_prices.min()

            # Define thresholds
            upside = (max_future - current_price) / current_price * 100
            downside = (current_price - min_future) / current_price * 100

            if upside > 2 and upside > downside:
                label = 1  # CALL (bullish)
            elif downside > 2 and downside > upside:
                label = -1  # PUT (bearish)
            else:
                label = 0  # HOLD

            labels.append(label)

        return np.array(labels)

    def train_model(self, data):
        """
        Train the ML model
        """
        features = self.prepare_features(data)
        labels = self.create_labels(data)

        # Remove neutral labels for binary classification
        mask = labels != 0
        X = features[mask]
        y = (labels[mask] + 1) // 2  # Convert -1,1 to 0,1

        if len(X) < 100:
            print("Not enough data for training")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

        # Save model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """
        Load trained model
        """
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict_signal(self, data):
        """
        Predict CALL/PUT signal
        """
        if self.model is None:
            if not self.load_model():
                return "HOLD"

        features = self.prepare_features(data)
        if len(features) == 0:
            return "HOLD"

        latest_features = features[-1].reshape(1, -1)
        scaled_features = self.scaler.transform(latest_features)

        prediction = self.model.predict(scaled_features)[0]
        probability = self.model.predict_proba(scaled_features)[0]

        if prediction == 1:
            confidence = probability[1]
            return f"BUY CALL ({confidence:.2f})"
        else:
            confidence = probability[0]
            return f"BUY PUT ({confidence:.2f})"

def generate_signal(data):
    """
    Legacy function for backward compatibility
    """
    predictor = OptionPredictor()
    return predictor.predict_signal(data)

def get_best_strike(processed_option_data, signal, underlying_price):
    """
    Recommend best strike price based on signal and option data
    """
    if not processed_option_data:
        return None

    strikes = processed_option_data['strikes']

    if signal.startswith("BUY CALL"):
        # For CALL, find strike slightly above current price
        candidates = [s for s in strikes if s['strikePrice'] > underlying_price]
        if candidates:
            # Sort by open interest (liquidity)
            candidates.sort(key=lambda x: x['CE']['openInterest'], reverse=True)
            return candidates[0]['strikePrice']

    elif signal.startswith("BUY PUT"):
        # For PUT, find strike slightly below current price
        candidates = [s for s in strikes if s['strikePrice'] < underlying_price]
        if candidates:
            candidates.sort(key=lambda x: x['PE']['openInterest'], reverse=True)
            return candidates[-1]['strikePrice']  # Closest below

    return underlying_price  # ATM if no clear signal

def calculate_targets(entry_price, signal, stop_loss_pct=0.5, target_pct=2.0):
    """
    Calculate entry, stop loss, and target prices
    """
    if signal.startswith("BUY CALL") or signal.startswith("BUY PUT"):
        stop_loss = entry_price * (1 - stop_loss_pct / 100)
        target = entry_price * (1 + target_pct / 100)
    else:
        stop_loss = entry_price
        target = entry_price

    return {
        'entry': round(entry_price, 2),
        'stop_loss': round(stop_loss, 2),
        'target': round(target, 2)
    }