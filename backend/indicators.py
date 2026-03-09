import ta
import pandas as pd
import numpy as np

def add_indicators(data):
    """
    Add technical indicators to the dataframe
    """
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_SIGNAL'] = macd.macd_signal()
    data['MACD_DIFF'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_UPPER'] = bb.bollinger_hband()
    data['BB_LOWER'] = bb.bollinger_lband()
    data['BB_MIDDLE'] = bb.bollinger_mavg()

    # VWAP
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close'])/3).cumsum() / data['Volume'].cumsum()

    # Moving Averages
    data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['STOCH_K'] = stoch.stoch()
    data['STOCH_D'] = stoch.stoch_signal()

    # ATR (Average True Range)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

    # Williams %R
    data['WILLR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()

    # CCI (Commodity Channel Index)
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()
    data['ADX_POS'] = adx.adx_pos()
    data['ADX_NEG'] = adx.adx_neg()

    return data