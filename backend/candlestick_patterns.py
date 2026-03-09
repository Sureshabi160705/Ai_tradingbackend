import talib
import pandas as pd

def detect_patterns(data):
    """
    Detect candlestick patterns using TA-Lib
    """
    open = data['Open']
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Single candlestick patterns
    data['DOJI'] = talib.CDLDOJI(open, high, low, close)
    data['HAMMER'] = talib.CDLHAMMER(open, high, low, close)
    data['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open, high, low, close)
    data['SPINNING_TOP'] = talib.CDLSPINNINGTOP(open, high, low, close)
    data['MARUBOZU'] = talib.CDLMARUBOZU(open, high, low, close)

    # Double candlestick patterns
    data['ENGULFING'] = talib.CDLENGULFING(open, high, low, close)
    data['HARAMI'] = talib.CDLHARAMI(open, high, low, close)
    data['PIERCING'] = talib.CDLPIERCING(open, high, low, close)
    data['DARK_CLOUD_COVER'] = talib.CDLDARKCLOUDCOVER(open, high, low, close)

    # Triple candlestick patterns
    data['MORNING_STAR'] = talib.CDLMORNINGSTAR(open, high, low, close)
    data['EVENING_STAR'] = talib.CDLEVENINGSTAR(open, high, low, close)
    data['THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(open, high, low, close)
    data['THREE_BLACK_CROWS'] = talib.CDL3BLACKCROWS(open, high, low, close)
    data['ABANDONED_BABY'] = talib.CDLABANDONEDBABY(open, high, low, close)

    # Other patterns
    data['RISING_THREE_METHODS'] = talib.CDLRISEFALL3METHODS(open, high, low, close)
    data['FALLING_THREE_METHODS'] = talib.CDLRISEFALL3METHODS(open, high, low, close)
    data['SEPARATING_LINES'] = talib.CDLSEPARATINGLINES(open, high, low, close)
    data['BREAKAWAY'] = talib.CDLBREAKAWAY(open, high, low, close)

    return data

def get_pattern_signals(data):
    """
    Convert pattern values to buy/sell signals
    """
    signals = []

    for idx, row in data.iterrows():
        signal = "HOLD"

        # Bullish patterns
        if row['ENGULFING'] == 100 or row['HAMMER'] == 100 or row['MORNING_STAR'] == 100:
            signal = "BUY"
        # Bearish patterns
        elif row['ENGULFING'] == -100 or row['SHOOTING_STAR'] == -100 or row['EVENING_STAR'] == -100:
            signal = "SELL"

        signals.append(signal)

    return signals