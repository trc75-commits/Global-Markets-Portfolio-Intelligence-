import pandas as pd
import yfinance as yf

US_INDEXES = ['^DJI', '^GSPC', '^IXIC']
TREASURY_YIELDS = ['^IRX', '^TNX', '^TYX']
GLOBAL_INDEXES = ['^FTSE', '^N225', '^GDAXI', '^HSI', '^SSEC']

def fetch_price_change(tickers):
    """
    Fetch current level and % change for given tickers.
    Returns DataFrame with columns: Index, Level, Change %
    """
    try:
        df = yf.download(tickers, period="2d", interval="1d", progress=False)
        
        if df.empty:
            return pd.DataFrame({
                'Index': tickers,
                'Level': [0.0] * len(tickers),
                'Change %': [0.0] * len(tickers)
            })
        
        # Handle both single and multiple ticker downloads
        # yfinance returns MultiIndex columns for multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close']
        else:
            close = df[['Close']]
        
        # Get last and previous close
        last = close.iloc[-1]
        prev = close.iloc[-2] if len(close) > 1 else last
        
        # Calculate percentage change
        pct = ((last - prev) / prev * 100).fillna(0)
        
        # Ensure we return proper arrays for DataFrame construction
        if isinstance(last, pd.Series):
            level_vals = last.values
            pct_vals = pct.values
        else:
            level_vals = [last]
            pct_vals = [pct]
        
        return pd.DataFrame({
            'Index': tickers,
            'Level': level_vals,
            'Change %': pct_vals
        })
        
    except Exception as e:
        # Return safe fallback on any error
        return pd.DataFrame({
            'Index': tickers,
            'Level': [0.0] * len(tickers),
            'Change %': [0.0] * len(tickers)
        })

def get_us_indexes():
    return fetch_price_change(US_INDEXES)

def get_treasury_yields():
    return fetch_price_change(TREASURY_YIELDS)

def get_global_indexes():
    return fetch_price_change(GLOBAL_INDEXES)
