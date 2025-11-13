
import requests, pandas as pd, yfinance as yf
from cache_manager import get_prices_smart

YF_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

def search_tickers(query,count=10):
    if not query: return []
    r=requests.get(YF_SEARCH_URL,params={'q':query,'quotesCount':count,'newsCount':0},timeout=8)
    if r.status_code!=200: return []
    data=r.json().get('quotes',[])[:count]
    return [{'symbol':q.get('symbol'),'shortname':q.get('shortname',''),'exch':q.get('exchDisp',''),'type':q.get('quoteType','')} for q in data]

def get_prices(tickers,period="1y",interval="1d"):
    return get_prices_smart(tickers,period=period,interval=interval)

def get_macro_headlines(max_items=8):
    syms=['^GSPC','^IXIC','^DJI','^TNX','DX-Y.NYB','GC=F']
    seen=set(); items=[]
    for s in syms:
        try: news=yf.Ticker(s).news or []
        except: news=[]
        for n in news:
            t=n.get('title'); l=n.get('link'); p=n.get('publisher','')
            if t and l and (t,l) not in seen:
                seen.add((t,l)); items.append((t,l,p))
            if len(items)>=max_items: return items
    return items
