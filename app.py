# Updated app.py
# - Require a matched suggestion to add a custom ticker (no longer falls back to raw typed value).
# - Build-only suggestions from search_tickers; if none found the Add button will not add.
# - When no historical price data exists, plot a minimal time series (start -> now) so the portfolio growth chart shows a line immediately.
# - Show a "Data last updated" box on all three tabs reflecting cache_prices.parquet modification time.

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date, timedelta

from data_manager import (
    init_db,
    get_portfolios,
    load_portfolio,
    save_portfolio,
    delete_portfolio,
)
from cache_manager import get_prices_smart, clear_cache
from market_overview import get_us_indexes, get_treasury_yields, get_global_indexes
from ai_interface import get_ai_feedback
from utils_marketdata import search_tickers  # used for live suggestions

# Reasonable, pre-populated choices for each asset class.
ASSET_CHOICES = {
    "Equities": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "SPY", "VTI", "QQQ", "BRK-B"
    ],
    "Fixed Income": [
        "IEF", "AGG", "BND", "LQD", "TIP", "TLT", "HYG"
    ],
    "Alternatives": [
        "GLD", "IAU", "SLV", "GDX", "VNQ", "GSG", "PDBC", "BTC-USD"
    ],
}

# Basic expected returns by asset class used for "Projected Annual" metric.
DEFAULT_PROJECTED_RETURNS = {
    "Equities": 0.07,      # 7% expected annual
    "Fixed Income": 0.03,  # 3% expected annual
    "Alternatives": 0.02,  # 2% expected annual
}

# Initialize session state for AI suggestions
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = None
if "suggested_tickers" not in st.session_state:
    st.session_state.suggested_tickers = []

# ---------------------------------------------------------------------
# Initialise database and ensure a default portfolio exists
# ---------------------------------------------------------------------
DB_PATH = "portfolios.db"
init_db(DB_PATH)

if "selected_portfolio" not in st.session_state:
    st.session_state.selected_portfolio = "My Portfolio"

# create a default portfolio if none exist
if not get_portfolios(DB_PATH):
    default_alloc = {
        "Equities": {"SPY": 0.6},
        "Fixed Income": {"IEF": 0.3},
        "Alternatives": {"GLD": 0.1},
    }
    save_portfolio(
        DB_PATH,
        "My Portfolio",
        {
            "allocations": default_alloc,
            "aum": 100_000,
            "risk_profile": "Balanced",
            "description": "Default diversified portfolio.",
            "created_at": datetime.utcnow().date().isoformat(),
        },
    )

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Global Markets Portfolio Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("💼 Portfolio Manager")

# ---------------------------------------------------------------------
# Helper: determine last data update time from cache file
# ---------------------------------------------------------------------
CACHE_FILE = "cache_prices.parquet"

def get_data_last_updated():
    try:
        if os.path.exists(CACHE_FILE):
            mtime = datetime.utcfromtimestamp(os.path.getmtime(CACHE_FILE))
            return mtime.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass
    return "Unknown"

# ---------------------------------------------------------------------
# Portfolio selection and creation
# ---------------------------------------------------------------------
_ports = get_portfolios(DB_PATH) or []
ports = [p.get("name", "Unnamed") for p in _ports]
if not ports:
    ports = ["My Portfolio"]

sel_index = ports.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in ports else 0
sel = st.sidebar.selectbox("Select Portfolio", ports, index=sel_index, key="sel_port")
st.session_state.selected_portfolio = sel

# add new portfolio
new_name = st.sidebar.text_input("New Portfolio Name", key="new_port_name")
if st.sidebar.button("➕ Add", key="add_portfolio_btn") and new_name.strip():
    default_alloc = {
        "Equities": {"SPY": 0.6},
        "Fixed Income": {"IEF": 0.3},
        "Alternatives": {"GLD": 0.1},
    }
    save_portfolio(
        DB_PATH,
        new_name.strip(),
        {
            "allocations": default_alloc,
            "aum": 0.0,
            "risk_profile": "Balanced",
            "description": "",
            "created_at": datetime.utcnow().date().isoformat(),
        },
    )
    st.session_state.selected_portfolio = new_name.strip()
    st.rerun()

# delete portfolio (disallow deleting the default)
if sel != "My Portfolio":
    if st.sidebar.button("🗑 Delete", key="delete_portfolio_btn"):
        delete_portfolio(DB_PATH, sel)
        st.session_state.selected_portfolio = "My Portfolio"
        st.rerun()

# ---------------------------------------------------------------------
# Load portfolio data
# ---------------------------------------------------------------------
data = load_portfolio(DB_PATH, st.session_state.selected_portfolio)
if not data:
    data = {
        "allocations": {
            "Equities": {"SPY": 0.6},
            "Fixed Income": {"IEF": 0.3},
            "Alternatives": {"GLD": 0.1},
        },
        "aum": 0.0,
        "created_at": datetime.utcnow().date().isoformat(),
    }

# Ensure portfolio has a created_at timestamp
if "created_at" not in data or not data.get("created_at"):
    data["created_at"] = datetime.utcnow().date().isoformat()
    save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)

alloc = data.get("allocations", {})
aum = float(data.get("aum", 0.0))

# ---------------------------------------------------------------------
# Allocation sliders - REDESIGNED FOR CLARITY
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎯 Asset Allocation")
st.sidebar.caption("Adjust your portfolio mix (must total 100%)")

# Get current allocations
alt_default = sum(alloc.get("Alternatives", {}).values())
eq_default = sum(alloc.get("Equities", {}).values())
fi_default = sum(alloc.get("Fixed Income", {}).values())

# Create three independent sliders
eq = st.sidebar.slider(
    "📈 Equities",
    min_value=0,
    max_value=100,
    value=int(eq_default * 100),
    step=5,
    format="%d%%",
    key="eq_slider",
    help="Stocks and equity ETFs - Higher risk, higher potential return"
)

fi = st.sidebar.slider(
    "🏦 Fixed Income",
    min_value=0,
    max_value=100,
    value=int(fi_default * 100),
    step=5,
    format="%d%%",
    key="fi_slider",
    help="Bonds and bond ETFs - Lower risk, stable income"
)

alt = st.sidebar.slider(
    "🪙 Alternatives",
    min_value=0,
    max_value=100,
    value=int(alt_default * 100),
    step=5,
    format="%d%%",
    key="alt_slider",
    help="Gold, commodities, REITs - Portfolio diversification"
)

# Calculate total and show visual feedback
total_allocation = eq + fi + alt

# Visual indicator for allocation total
if total_allocation == 100:
    st.sidebar.success(f"✅ Total: {total_allocation}%")
elif total_allocation < 100:
    st.sidebar.warning(f"⚠️ Total: {total_allocation}% (Underallocated by {100-total_allocation}%)")
else:
    st.sidebar.error(f"❌ Total: {total_allocation}% (Overallocated by {total_allocation-100}%)")

# Normalize to ensure they sum to 100% when used
if total_allocation > 0:
    eq = (eq / total_allocation)
    fi = (fi / total_allocation)
    alt = (alt / total_allocation)
else:
    eq, fi, alt = 0.6, 0.3, 0.1

st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# helper to normalise sub-tickers
# ---------------------------------------------------------------------
def normalise(sub_tickers, weight):
    if not sub_tickers or weight <= 0:
        return {}
    equal = weight / len(sub_tickers)
    return {t: equal for t in sub_tickers}

# utility to convert search results into select options
def build_suggestion_options(results):
    """
    results: list of dicts from search_tickers (symbol, shortname)
    returns: list of (label, symbol) tuples. If no results, return empty list.
    """
    opts = []
    for r in results:
        sym = (r.get("symbol") or "").upper().strip()
        name = r.get("shortname") or ""
        label = f"{sym} — {name}" if name else sym
        if sym and sym not in [s for _, s in opts]:
            opts.append((label, sym))
    return opts

# ---------------------------------------------------------------------
# AUM input
# ---------------------------------------------------------------------
aum_input = st.sidebar.number_input(
    "💵 Total Investable Assets ($)",
    min_value=0.0,
    value=float(aum),
    step=10_000.0,
    key="aum_input",
)

# ---------------------------------------------------------------------
# Ticker management with required matched suggestion
# ---------------------------------------------------------------------
# equities
st.sidebar.subheader("Equity Holdings")
eq_tickers = list(alloc.get("Equities", {}).keys())

eq_options = ASSET_CHOICES["Equities"] + ["Custom"]
eq_choice = st.sidebar.selectbox("Choose Equity to add", eq_options, index=0, key="select_eq_choice")

eq_suggestion_selected = None
if eq_choice == "Custom":
    # free-text: any input triggers suggestions, but we require a selected suggestion to add.
    eq_custom_typed = st.sidebar.text_input("Search ticker or company name (e.g., K or Coca-Cola)", key="add_eq_query")
    if eq_custom_typed and eq_custom_typed.strip():
        try:
            results = search_tickers(eq_custom_typed.strip(), count=10)
        except Exception:
            results = []
        opts = build_suggestion_options(results)
        if opts:
            labels = [lbl for lbl, _ in opts]
            eq_sel_label = st.sidebar.selectbox("Suggestions (pick one to add)", labels, index=0, key="eq_sugg_sel")
            # map back to symbol
            eq_suggestion_selected = dict(opts)[eq_sel_label]
        else:
            st.sidebar.info("No matching tickers found. Refine your search.")
            eq_suggestion_selected = None
    else:
        st.sidebar.caption("Type a company name or ticker to see matches.")
        eq_suggestion_selected = None

if st.sidebar.button("Add Equity", key="add_eq_btn"):
    # Only add if a suggested symbol was selected from the search results or a pre-defined option chosen.
    if eq_choice == "Custom":
        ticker = (eq_suggestion_selected or "").upper().strip()
    else:
        ticker = eq_choice.upper().strip()
    if ticker and ticker != "CUSTOM" and ticker not in eq_tickers:
        if eq_choice == "Custom" and not eq_suggestion_selected:
            # Don't add raw typed input — user must pick a suggestion
            st.sidebar.error("Please select a matching ticker from Suggestions to add.")
        else:
            eq_tickers.append(ticker)
            new_alloc = {
                "Equities": normalise(eq_tickers, eq),
                "Fixed Income": normalise(list(alloc.get("Fixed Income", {}).keys()), fi),
                "Alternatives": normalise(list(alloc.get("Alternatives", {}).keys()), alt),
            }
            data["allocations"] = new_alloc
            data["aum"] = aum_input
            save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
            st.rerun()

for t in eq_tickers.copy():
    if st.sidebar.button(f"❌ Remove {t}", key=f"remove_eq_{t}"):
        eq_tickers.remove(t)
        new_alloc = {
            "Equities": normalise(eq_tickers, eq),
            "Fixed Income": normalise(list(alloc.get("Fixed Income", {}).keys()), fi),
            "Alternatives": normalise(list(alloc.get("Alternatives", {}).keys()), alt),
        }
        data["allocations"] = new_alloc
        data["aum"] = aum_input
        save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
        st.rerun()

# fixed income
st.sidebar.subheader("Fixed Income Holdings")
fi_tickers = list(alloc.get("Fixed Income", {}).keys())

fi_options = ASSET_CHOICES["Fixed Income"] + ["Custom"]
fi_choice = st.sidebar.selectbox("Choose Fixed Income to add", fi_options, index=0, key="select_fi_choice")

fi_suggestion_selected = None
if fi_choice == "Custom":
    fi_custom_typed = st.sidebar.text_input("Search ticker or fund name (e.g., IEF)", key="add_fi_query")
    if fi_custom_typed and fi_custom_typed.strip():
        try:
            results = search_tickers(fi_custom_typed.strip(), count=10)
        except Exception:
            results = []
        opts = build_suggestion_options(results)
        if opts:
            labels = [lbl for lbl, _ in opts]
            fi_sel_label = st.sidebar.selectbox("Suggestions (pick one to add)", labels, key="fi_sugg_sel")
            fi_suggestion_selected = dict(opts)[fi_sel_label]
        else:
            st.sidebar.info("No matching tickers found. Refine your search.")
            fi_suggestion_selected = None
    else:
        st.sidebar.caption("Type a fund name or ticker to see matches.")
        fi_suggestion_selected = None

if st.sidebar.button("Add Fixed Income", key="add_fi_btn"):
    if fi_choice == "Custom":
        ticker = (fi_suggestion_selected or "").upper().strip()
    else:
        ticker = fi_choice.upper().strip()
    if ticker and ticker != "CUSTOM" and ticker not in fi_tickers:
        if fi_choice == "Custom" and not fi_suggestion_selected:
            st.sidebar.error("Please select a matching ticker from Suggestions to add.")
        else:
            fi_tickers.append(ticker)
            new_alloc = {
                "Equities": normalise(list(alloc.get("Equities", {}).keys()), eq),
                "Fixed Income": normalise(fi_tickers, fi),
                "Alternatives": normalise(list(alloc.get("Alternatives", {}).keys()), alt),
            }
            data["allocations"] = new_alloc
            data["aum"] = aum_input
            save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
            st.rerun()

for t in fi_tickers.copy():
    if st.sidebar.button(f"❌ Remove {t}", key=f"remove_fi_{t}"):
        fi_tickers.remove(t)
        new_alloc = {
            "Equities": normalise(list(alloc.get("Equities", {}).keys()), eq),
            "Fixed Income": normalise(fi_tickers, fi),
            "Alternatives": normalise(list(alloc.get("Alternatives", {}).keys()), alt),
        }
        data["allocations"] = new_alloc
        data["aum"] = aum_input
        save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
        st.rerun()

# alternatives
st.sidebar.subheader("Alternative Holdings")
alt_tickers = list(alloc.get("Alternatives", {}).keys())

alt_options = ASSET_CHOICES["Alternatives"] + ["Custom"]
alt_choice = st.sidebar.selectbox("Choose Alternative to add", alt_options, index=0, key="select_alt_choice")

alt_suggestion_selected = None
if alt_choice == "Custom":
    alt_custom_typed = st.sidebar.text_input("Search ticker or asset name (e.g., GLD, Bitcoin)", key="add_alt_query")
    if alt_custom_typed and alt_custom_typed.strip():
        try:
            results = search_tickers(alt_custom_typed.strip(), count=10)
        except Exception:
            results = []
        opts = build_suggestion_options(results)
        if opts:
            labels = [lbl for lbl, _ in opts]
            alt_sel_label = st.sidebar.selectbox("Suggestions (pick one to add)", labels, key="alt_sugg_sel")
            alt_suggestion_selected = dict(opts)[alt_sel_label]
        else:
            st.sidebar.info("No matching tickers found. Refine your search.")
            alt_suggestion_selected = None
    else:
        st.sidebar.caption("Type an asset name or ticker to see matches.")
        alt_suggestion_selected = None

if st.sidebar.button("Add Alternative", key="add_alt_btn"):
    if alt_choice == "Custom":
        ticker = (alt_suggestion_selected or "").upper().strip()
    else:
        ticker = alt_choice.upper().strip()
    if ticker and ticker != "CUSTOM" and ticker not in alt_tickers:
        if alt_choice == "Custom" and not alt_suggestion_selected:
            st.sidebar.error("Please select a matching ticker from Suggestions to add.")
        else:
            alt_tickers.append(ticker)
            new_alloc = {
                "Equities": normalise(list(alloc.get("Equities", {}).keys()), eq),
                "Fixed Income": normalise(list(alloc.get("Fixed Income", {}).keys()), fi),
                "Alternatives": normalise(alt_tickers, alt),
            }
            data["allocations"] = new_alloc
            data["aum"] = aum_input
            save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
            st.rerun()

for t in alt_tickers.copy():
    if st.sidebar.button(f"❌ Remove {t}", key=f"remove_alt_{t}"):
        alt_tickers.remove(t)
        new_alloc = {
            "Equities": normalise(list(alloc.get("Equities", {}).keys()), eq),
            "Fixed Income": normalise(list(alloc.get("Fixed Income", {}).keys()), fi),
            "Alternatives": normalise(alt_tickers, alt),
        }
        data["allocations"] = new_alloc
        data["aum"] = aum_input
        save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
        st.rerun()

# rebuild allocation dictionary (final state for this run)
alloc = {
    "Equities": normalise(eq_tickers, eq),
    "Fixed Income": normalise(fi_tickers, fi),
    "Alternatives": normalise(alt_tickers, alt),
}

# persist changes
data["allocations"] = alloc
data["aum"] = aum_input
save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Portfolio", "🌎 Markets", "🧠 AI Advisor"])

# ---------------------------------------------------------------------
# Portfolio Overview tab
# ---------------------------------------------------------------------
with tab1:
    # Header with timeframe selector
    col_header, col_timeframe = st.columns([3, 1])
    with col_header:
        st.markdown(f"### 📊 {st.session_state.selected_portfolio}")
        st.caption(f"Created: {data.get('created_at', 'Unknown')}")
    with col_timeframe:
        timeframe_opts = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "1Y": 365,
            "ALL": None,
        }
        tf_display = st.selectbox("Period", list(timeframe_opts.keys()), index=4, key="timeframe_select", label_visibility="collapsed")
        # show last-updated box
        st.info(f"Data last updated: {get_data_last_updated()}")

    # flatten allocations
    flat_alloc = {ticker: weight for cat in alloc.values() for ticker, weight in cat.items()}

    # fetch price data
    if flat_alloc:
        df_prices = get_prices_smart(list(flat_alloc.keys()))
    else:
        df_prices = pd.DataFrame()

    # determine creation date and compute start date
    created_at_str = data.get("created_at")
    try:
        created_at_date = pd.to_datetime(created_at_str).date()
    except Exception:
        created_at_date = datetime.utcnow().date()

    now = datetime.utcnow().date()
    delta_days = timeframe_opts.get(tf_display)
    if tf_display == "ALL" or delta_days is None:
        start_date = pd.to_datetime(created_at_date)
    else:
        candidate = pd.to_datetime(now - timedelta(days=delta_days))
        start_date = max(pd.to_datetime(created_at_date), candidate)
        if tf_display == "1D" and created_at_date == now:
            start_date = pd.to_datetime(now - timedelta(days=1))

    # slice price series
    if not df_prices.empty:
        try:
            df_prices = df_prices.loc[start_date:]
        except Exception:
            df_prices = df_prices[df_prices.index >= pd.to_datetime(start_date)]

    # Calculate metrics with better error handling
    if not df_prices.empty and flat_alloc and len(df_prices) > 1:
        try:
            rets = df_prices.pct_change().dropna()
            tickers = [t for t in list(flat_alloc.keys()) if t in rets.columns]
            if tickers:
                weights = np.array([flat_alloc[t] for t in tickers])
                port_ret = rets[tickers].dot(weights)
                cumulative = (1 + port_ret).cumprod()

                if len(cumulative) > 0:
                    growth = (cumulative.iloc[-1] - 1.0) * 100.0
                    current_value = aum_input * (1.0 + growth / 100.0)
                    risk = np.std(port_ret) * np.sqrt(252)
                    sharpe = (np.mean(port_ret) * 252 - 0.04) / risk if risk > 0 else 0.0
                else:
                    cumulative = pd.Series(dtype=float)
                    growth = 0.0
                    current_value = aum_input
                    risk = 0.0
                    sharpe = 0.0
            else:
                cumulative = pd.Series(dtype=float)
                growth = 0.0
                current_value = aum_input
                risk = 0.0
                sharpe = 0.0
        except Exception:
            cumulative = pd.Series(dtype=float)
            growth = 0.0
            current_value = aum_input
            risk = 0.0
            sharpe = 0.0
    else:
        # Single day calculation or no price data (we still want to show a plot)
        if not df_prices.empty and flat_alloc:
            try:
                tickers = [t for t in list(flat_alloc.keys()) if t in df_prices.columns]
                if tickers and len(df_prices) >= 1:
                    start_prices = df_prices.iloc[0][tickers]
                    end_prices = df_prices.iloc[-1][tickers]
                    pct_change = (end_prices - start_prices) / start_prices.replace({0: np.nan})
                    weights = np.array([flat_alloc[t] for t in tickers])
                    port_change = np.nansum(pct_change.values * weights)
                    growth = float(port_change) * 100.0
                    current_value = aum_input * (1.0 + growth / 100.0)
                else:
                    growth = 0.0
                    current_value = aum_input
                risk = 0.0
                sharpe = 0.0
                cumulative = pd.Series(dtype=float)
            except Exception:
                cumulative = pd.Series(dtype=float)
                growth = 0.0
                current_value = aum_input
                risk = 0.0
                sharpe = 0.0
        else:
            # No price data available -> still fill defaults so UI shows values and chart shows a minimal line
            cumulative = pd.Series(dtype=float)
            growth = 0.0
            current_value = aum_input
            risk = 0.0
            sharpe = 0.0

    # Compute projected annual return (simple weighted assumption)
    try:
        projected = (
            eq * DEFAULT_PROJECTED_RETURNS["Equities"]
            + fi * DEFAULT_PROJECTED_RETURNS["Fixed Income"]
            + alt * DEFAULT_PROJECTED_RETURNS["Alternatives"]
        ) * 100.0
    except Exception:
        projected = 0.0

    # Metrics cards with better styling
    st.markdown("---")
    st.markdown("#### 📈 Performance Metrics")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.metric(
            "Portfolio Value", 
            f"${current_value:,.0f}",
            delta=f"${current_value - aum_input:,.0f}" if current_value != aum_input else None
        )
    with c2:
        st.metric(
            "Total Return", 
            f"{growth:.2f}%",
            delta=f"{growth:.2f}%" if growth != 0 else None,
            delta_color="normal"
        )
    with c3:
        st.metric(
            "Annualized Risk", 
            f"{risk:.2%}",
            help="Volatility (standard deviation) of returns"
        )
    with c4:
        st.metric(
            "Sharpe Ratio", 
            f"{sharpe:.2f}",
            help="Risk-adjusted return (>1 is good, >2 is very good)"
        )
    with c5:
        st.metric(
            "Projected Annual", 
            f"{projected:.2f}%",
            help="Expected return based on asset class assumptions"
        )
    
    st.markdown("---")
    
    # Charts in two columns
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown("##### 📈 Performance Over Time")
        # If we have a computed cumulative (returns based) series, plot it. Otherwise create a minimal time series
        if not cumulative.empty:
            try:
                fig = px.line(
                    cumulative, 
                    title="",
                    labels={"value": "Cumulative Return", "index": "Date"}, 
                    template="plotly_white"
                )
                fig.update_traces(line_color="#1f77b4", line_width=2.5)
                fig.update_layout(showlegend=False, hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("📊 Unable to render performance chart")
        elif not df_prices.empty and flat_alloc:
            try:
                tickers = [t for t in list(flat_alloc.keys()) if t in df_prices.columns]
                if tickers:
                    rets = df_prices[tickers].pct_change().fillna(0)
                    weights = np.array([flat_alloc[t] for t in tickers])
                    port_ret = rets[tickers].dot(weights)
                    growth_series = (1 + port_ret).cumprod() * aum_input
                    fig = px.line(
                        growth_series, 
                        title="",
                        labels={"value": "Portfolio Value ($)", "index": "Date"}, 
                        template="plotly_white"
                    )
                    fig.update_traces(line_color="#1f77b4", line_width=2.5)
                    fig.update_layout(showlegend=False, hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # No tickers available after filtering - fallback to minimal line
                    fallback_index = pd.to_datetime([start_date, pd.to_datetime(now)])
                    fallback_values = pd.Series([aum_input, current_value], index=fallback_index)
                    fig = px.line(fallback_values, labels={"value": "Portfolio Value ($)", "index": "Date"}, template="plotly_white")
                    fig.update_traces(line_color="#1f77b4", line_width=2.5)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("📊 Insufficient data for performance chart")
        else:
            # No price data at all — create a minimal 2-point time series (created_at -> now) so a line appears immediately.
            try:
                idx_start = pd.to_datetime(created_at_date)
                idx_now = pd.to_datetime(now)
                if idx_start >= idx_now:
                    # If created today, make a small backward step so line has two distinct points
                    idx_start = idx_now - pd.Timedelta(days=1)
                fallback_values = pd.Series([aum_input, current_value], index=[idx_start, idx_now])
                fig = px.line(fallback_values, labels={"value": "Portfolio Value ($)", "index": "Date"}, template="plotly_white")
                fig.update_traces(line_color="#1f77b4", line_width=2.5)
                fig.update_layout(showlegend=False, hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("📊 Add holdings to see performance chart")
    
    with chart_col2:
        st.markdown("##### 🥧 Asset Allocation")
        if sum([eq, fi, alt]) > 0:
            fig = px.pie(
                values=[eq, fi, alt],
                names=["Equities", "Fixed Income", "Alternatives"],
                title="",
                template="plotly_white",
                hole=0.4,
                color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🥧 Adjust allocations to see chart")
    
    st.markdown("---")
    
    # Holdings breakdown
    st.markdown("##### 📋 Current Holdings")
    holdings_data = []
    for asset_class, holdings in alloc.items():
        for ticker, weight in holdings.items():
            holdings_data.append({
                "Asset Class": asset_class,
                "Ticker": ticker,
                "Allocation": f"{weight*100:.1f}%",
                "Value": f"${aum_input * weight:,.2f}"
            })
    
    if holdings_data:
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No holdings yet. Add tickers in the sidebar.")
    
    # Action buttons
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            clear_cache()
            st.rerun()

# ---------------------------------------------------------------------
# Markets tab (unchanged)
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### 🌎 Global Markets Overview")
    
    today_label = datetime.utcnow().strftime("%B %d, %Y")
    st.caption(f"Market data as of {today_label}")
    # show last-updated box
    st.info(f"Data last updated: {get_data_last_updated()}")
    
    st.markdown("---")

    def fmt_market_data(df: pd.DataFrame) -> pd.DataFrame:
        """Format market data with colors and better presentation"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Format Change % with colors
        if "Change %" in df.columns:
            def format_change(val):
                try:
                    num = float(val)
                    color = "🟢" if num > 0 else "🔴" if num < 0 else "⚪"
                    return f"{color} {num:+.2f}%"
                except:
                    return val
            df["Change %"] = df["Change %"].apply(format_change)
        
        # Format Level
        if "Level" in df.columns:
            df["Level"] = pd.to_numeric(df["Level"], errors="coerce").apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )
        
        return df

    # Fetch market data
    us = get_us_indexes()
    tr = get_treasury_yields()
    gl = get_global_indexes()

    # Add descriptions
    if not us.empty:
        us.insert(1, "Name", ["Dow Jones Industrial", "S&P 500", "Nasdaq Composite"][: len(us)])
    if not tr.empty:
        tr.insert(1, "Name", ["3-Month Treasury", "10-Year Treasury", "30-Year Treasury"][: len(tr)])
    if not gl.empty:
        gl.insert(
            1,
            "Name",
            ["FTSE 100 (UK)", "Nikkei 225 (Japan)", "DAX (Germany)", "Hang Seng (HK)", "Shanghai Composite (China)"][: len(gl)],
        )

    # Display in cleaner cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 U.S. Equity Indexes")
        if not us.empty:
            # Remove index column and rename
            display_us = us.copy()
            display_us = display_us.rename(columns={"Index": "Symbol"})
            st.dataframe(
                fmt_market_data(display_us),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Market data temporarily unavailable")
        
        st.markdown("#### 💰 U.S. Treasury Yields")
        if not tr.empty:
            display_tr = tr.copy()
            display_tr = display_tr.rename(columns={"Index": "Symbol"})
            st.dataframe(
                fmt_market_data(display_tr),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Treasury data temporarily unavailable")
    
    with col2:
        st.markdown("#### 🌍 International Markets")
        if not gl.empty:
            display_gl = gl.copy()
            display_gl = display_gl.rename(columns={"Index": "Symbol"})
            st.dataframe(
                fmt_market_data(display_gl),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("International data temporarily unavailable")
        
        # Market summary box
        st.markdown("#### 📰 Market Summary")
        st.info("""
        **Key Points:**
        - Markets update daily after close
        - Data sourced from Yahoo Finance
        - Percentages show daily change
        """)
    
    st.markdown("---")
    st.caption("💡 Tip: Use this data to inform your portfolio allocation decisions")

# ---------------------------------------------------------------------
# AI Advisor tab (unchanged)
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### 🧠 AI Portfolio Advisor")
    st.caption("Get intelligent recommendations based on market data and your goals")
    # show last-updated box
    st.info(f"Data last updated: {get_data_last_updated()}")
    st.markdown("---")
    
    # Two-column layout for inputs
    input_col1, input_col2 = st.columns([2, 1])
    
    with input_col1:
        st.markdown("#### 💭 Ask Your Advisor")
        query = st.text_area(
            "What would you like help with?",
            value="How should I adjust my portfolio for better returns?",
            key="advisor_query",
            height=100,
            help="Examples: 'Add 3 defensive stocks', 'Increase tech exposure', 'Make portfolio more conservative'"
        )
    
    with input_col2:
        st.markdown("#### 📝 Portfolio Notes")
        desc = st.text_area(
            "Goals & Strategy",
            value=data.get("description", ""),
            key="portfolio_desc",
            height=100,
            placeholder="E.g., Long-term growth, Conservative income, etc."
        )
    
    # Action button with better styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        analyze_btn = st.button(
            "🤖 Get AI Advice", 
            key="submit_query",
            use_container_width=True
        )
    
    st.markdown("---")
    
    if analyze_btn:
        with st.spinner("🔍 Analyzing your portfolio and market conditions..."):
            advisor_info = {
                "risk_profile": data.get("risk_profile", "Balanced"),
                "metrics": {"growth": growth, "sharpe": sharpe},
            }
            response, suggested_alloc, suggested_tickers = get_ai_feedback(
                advisor_info, query, alloc
            )
            
            # Display response in a nice container
            st.markdown("#### 💡 AI Recommendations")
            st.markdown(response)
            
            st.session_state.ai_suggestions = suggested_alloc
            st.session_state.suggested_tickers = suggested_tickers
            
            data["last_advice"] = response
            data["description"] = desc
            save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)
    
    # Show apply/dismiss buttons if suggestions exist
    if st.session_state.ai_suggestions or st.session_state.suggested_tickers:
        st.markdown("---")
        st.markdown("#### ⚡ Quick Actions")
        
        col_apply, col_dismiss = st.columns(2)
        
        with col_apply:
            if st.button(
                "✅ Apply Suggestions", 
                key="apply_suggestions", 
                use_container_width=True,
                help="This will update your portfolio with the AI's recommendations"
            ):
                # Apply suggested asset-class allocations if provided
                if st.session_state.ai_suggestions:
                    new_eq = st.session_state.ai_suggestions.get("Equities", eq)
                    new_fi = st.session_state.ai_suggestions.get("Fixed Income", fi)
                    new_alt = st.session_state.ai_suggestions.get("Alternatives", alt)
                    
                    # update session sliders so UI reflects changes after rerun
                    st.session_state.eq_slider = int(new_eq * 100)
                    st.session_state.fi_slider = int(new_fi * 100)
                    st.session_state.alt_slider = int(new_alt * 100)
                else:
                    new_eq, new_fi, new_alt = eq, fi, alt

                # Add suggested tickers to equities (persist them)
                if st.session_state.suggested_tickers:
                    for ticker in st.session_state.suggested_tickers:
                        if ticker not in eq_tickers:
                            eq_tickers.append(ticker)

                # Rebuild allocations to persist the changes now
                new_alloc = {
                    "Equities": normalise(eq_tickers, new_eq),
                    "Fixed Income": normalise(fi_tickers, new_fi),
                    "Alternatives": normalise(alt_tickers, new_alt),
                }
                data["allocations"] = new_alloc
                data["aum"] = aum_input
                save_portfolio(DB_PATH, st.session_state.selected_portfolio, data)

                st.session_state.ai_suggestions = None
                st.session_state.suggested_tickers = []
                
                st.success("✅ Portfolio updated successfully!")
                st.balloons()
                st.rerun()
        
        with col_dismiss:
            if st.button(
                "❌ Dismiss", 
                key="cancel_suggestions",
                use_container_width=True
            ):
                st.session_state.ai_suggestions = None
                st.session_state.suggested_tickers = []
                st.info("Suggestions dismissed")
                st.rerun()
    
    # Show last advice if no new suggestions
    elif data.get("last_advice"):
        st.markdown("---")
        st.markdown("#### 📜 Previous Consultation")
        with st.expander("View Last Advice", expanded=False):
            st.markdown(data["last_advice"])
    
    # Help section
    st.markdown("---")
    with st.expander("💡 How to Use the AI Advisor"):
        st.markdown("""
        **Examples of what you can ask:**
        
        - *"Add 3 defensive stocks for stability"*
        - *"I want higher growth, shift to tech"*
        - *"Make my portfolio more conservative"*
        - *"Add stocks that performed well recently"*
        - *"Increase equity exposure to 70%"*
        
        **What the AI considers:**
        - Your current allocations and holdings
        - Market conditions and recent trends
        - Risk profile and investment goals
        - Portfolio metrics (growth, Sharpe ratio)
        
        The AI will suggest specific tickers and allocation changes you can apply with one click.
        """)
