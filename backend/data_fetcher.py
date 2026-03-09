import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import time

def get_index_data(symbol, interval="5m", period="5d"):
    """
    Fetch historical data for an index
    """
    data = yf.download(symbol, interval=interval, period=period, progress=False)
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    return data

def get_nifty_data():
    return get_index_data("^NSEI")

def get_banknifty_data():
    return get_index_data("^NSEBANK")

def get_sensex_data():
    return get_index_data("^BSESN")

def get_option_chain(symbol="NIFTY"):
    """
    Fetch option chain data from NSE
    """
    url_map = {
        "NIFTY": "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
        "BANKNIFTY": "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY%20BANK",
        "SENSEX": "https://www.nseindia.com/api/option-chain-indices?symbol=SENSEX"
    }

    url = url_map.get(symbol, url_map["NIFTY"])

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive"
    }

    try:
        session = requests.Session()
        # First visit the main page to set cookies
        session.get("https://www.nseindia.com", headers=headers)
        time.sleep(2)
        response = session.get(url, headers=headers)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None

def get_live_price(symbol):
    """
    Get current price of an index
    """
    ticker = yf.Ticker(symbol)
    return ticker.info.get('regularMarketPrice', 0)

def get_expiry_dates(symbol="NIFTY"):
    """
    Get available expiry dates for options
    """
    data = get_option_chain(symbol)
    if data and 'records' in data:
        return data['records'].get('expiryDates', [])
    return []

def get_strikes(symbol="NIFTY"):
    """
    Get available strike prices
    """
    data = get_option_chain(symbol)
    if data and 'records' in data:
        return [strike['strikePrice'] for strike in data['records']['data']]
    return []