import requests
import pandas as pd
from datetime import datetime
import json

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    }

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None

def process_option_chain(data):
    """
    Process raw option chain data into structured format
    """
    if not data or 'records' not in data:
        return None

    records = data['records']
    underlying_value = records.get('underlyingValue', 0)
    timestamp = records.get('timestamp', '')

    # Get expiry dates
    expiry_dates = records.get('expiryDates', [])

    # Process strikes
    strikes_data = []
    for strike_data in records.get('data', []):
        strike_info = {
            'strikePrice': strike_data.get('strikePrice', 0),
            'expiryDate': strike_data.get('expiryDate', ''),
            'CE': {
                'openInterest': strike_data.get('CE', {}).get('openInterest', 0),
                'changeinOpenInterest': strike_data.get('CE', {}).get('changeinOpenInterest', 0),
                'totalTradedVolume': strike_data.get('CE', {}).get('totalTradedVolume', 0),
                'impliedVolatility': strike_data.get('CE', {}).get('impliedVolatility', 0),
                'lastPrice': strike_data.get('CE', {}).get('lastPrice', 0),
                'change': strike_data.get('CE', {}).get('change', 0),
                'bidQty': strike_data.get('CE', {}).get('bidQty', 0),
                'bidprice': strike_data.get('CE', {}).get('bidprice', 0),
                'askQty': strike_data.get('CE', {}).get('askQty', 0),
                'askPrice': strike_data.get('CE', {}).get('askPrice', 0)
            },
            'PE': {
                'openInterest': strike_data.get('PE', {}).get('openInterest', 0),
                'changeinOpenInterest': strike_data.get('PE', {}).get('changeinOpenInterest', 0),
                'totalTradedVolume': strike_data.get('PE', {}).get('totalTradedVolume', 0),
                'impliedVolatility': strike_data.get('PE', {}).get('impliedVolatility', 0),
                'lastPrice': strike_data.get('PE', {}).get('lastPrice', 0),
                'change': strike_data.get('PE', {}).get('change', 0),
                'bidQty': strike_data.get('PE', {}).get('bidQty', 0),
                'bidprice': strike_data.get('PE', {}).get('bidprice', 0),
                'askQty': strike_data.get('PE', {}).get('askQty', 0),
                'askPrice': strike_data.get('PE', {}).get('askPrice', 0)
            }
        }
        strikes_data.append(strike_info)

    return {
        'underlying_value': underlying_value,
        'timestamp': timestamp,
        'expiry_dates': expiry_dates,
        'strikes': strikes_data
    }

def get_oi_analysis(processed_data):
    """
    Analyze open interest data for insights
    """
    if not processed_data:
        return {}

    strikes = processed_data['strikes']
    underlying = processed_data['underlying_value']

    # Find max OI for CE and PE
    max_ce_oi = max(strikes, key=lambda x: x['CE']['openInterest'])
    max_pe_oi = max(strikes, key=lambda x: x['PE']['openInterest'])

    # Calculate PCR (Put-Call Ratio)
    total_ce_oi = sum(s['CE']['openInterest'] for s in strikes)
    total_pe_oi = sum(s['PE']['openInterest'] for s in strikes)
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

    return {
        'max_ce_oi_strike': max_ce_oi['strikePrice'],
        'max_pe_oi_strike': max_pe_oi['strikePrice'],
        'pcr': round(pcr, 2),
        'underlying': underlying
    }

def get_atm_strikes(processed_data, tolerance=100):
    """
    Get at-the-money strikes
    """
    if not processed_data:
        return []

    underlying = processed_data['underlying_value']
    strikes = [s['strikePrice'] for s in processed_data['strikes']]

    atm_strikes = [s for s in strikes if abs(s - underlying) <= tolerance]
    return atm_strikes