import os
import pandas as pd
import yfinance as yf
import datetime as dt

CACHE_FILE = "cache_prices.parquet"
REFRESH_DAYS = 3
MAX_AGE_DAYS = 1

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_parquet(CACHE_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _save_cache(df):
    try:
        df.to_parquet(CACHE_FILE)
    except Exception:
        pass

def clear_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

def _extract_close_prices(df, tickers):
    """
    Extract close prices from yfinance DataFrame and return properly formatted DataFrame.
    Handles both single and multiple ticker cases.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Case 1: Multiple tickers (MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            result = df["Adj Close"]
        elif "Close" in df.columns.get_level_values(0):
            result = df["Close"]
        else:
            result = df
        return result
    
    # Case 2: Single ticker
    if len(tickers) == 1:
        ticker = tickers[0]
        # yfinance returns columns: Open, High, Low, Close, Volume, etc.
        if "Adj Close" in df.columns:
            return pd.DataFrame({ticker: df["Adj Close"]})
        elif "Close" in df.columns:
            return pd.DataFrame({ticker: df["Close"]})
        else:
            # Fallback: use the entire series if it's already a Series
            if isinstance(df, pd.Series):
                return pd.DataFrame({ticker: df})
            return df
    
    # Case 3: Already properly formatted
    return df

def get_prices_smart(tickers, period="1y", interval="1d"):
    tickers = [t.upper() for t in tickers if t]
    if not tickers:
        return pd.DataFrame()
    
    df_cache = _load_cache()
    if not df_cache.empty:
        df_cache = df_cache.loc[:, [c for c in df_cache.columns if c in tickers]]
    
    today = dt.date.today()
    
    # Check if cache needs refresh
    try:
        cache_age = (today - df_cache.index.max().date()).days
        needs_refresh = df_cache.empty or cache_age > MAX_AGE_DAYS
    except (AttributeError, TypeError, ValueError):
        needs_refresh = True
    
    if needs_refresh:
        # Full refresh - download all data
        df_new = yf.download(tickers, period=period, interval=interval, progress=False)
        df_new = _extract_close_prices(df_new, tickers)
        _save_cache(df_new)
        return df_new
    
    # Partial refresh - get recent data only
    start = df_cache.index.max() - dt.timedelta(days=REFRESH_DAYS)
    df_recent = yf.download(tickers, start=start.strftime("%Y-%m-%d"), interval=interval, progress=False)
    df_recent = _extract_close_prices(df_recent, tickers)
    
    # Combine and deduplicate
    df_comb = pd.concat([df_cache, df_recent])
    df_comb = df_comb[~df_comb.index.duplicated(keep="last")].sort_index()
    _save_cache(df_comb)
    return df_comb
