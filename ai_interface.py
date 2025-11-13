import json
import yfinance as yf
import pandas as pd
from utils_marketdata import get_macro_headlines

async def get_ai_feedback_with_adjustments(info, user_query="", current_allocations=None):
    """
    Use Claude API to analyze user request and make intelligent portfolio adjustments.
    
    Args:
        info: Dictionary containing risk_profile and metrics
        user_query: The user's question/instruction
        current_allocations: Current portfolio allocation dict
    
    Returns:
        tuple: (response_text, suggested_allocations_dict or None, added_tickers list)
    """
    # Fetch market context
    headlines = get_macro_headlines(5)
    hl = '\n'.join([f"- {t}" for t, _, _ in headlines[:3]])
    
    rp = info.get('risk_profile', 'Balanced')
    gr = info.get('metrics', {}).get('growth', 0)
    shr = info.get('metrics', {}).get('sharpe', 0)
    
    # Get current portfolio summary
    current_summary = _summarize_portfolio(current_allocations)
    
    # Check if this is an adjustment request
    adjustment_keywords = ['add', 'increase', 'more', 'shift', 'change', 'reduce', 
                          'decrease', 'less', 'rebalance', 'adjust', 'move to',
                          'safer', 'aggressive', 'defensive', 'growth', 'high return']
    
    wants_adjustment = any(kw in user_query.lower() for kw in adjustment_keywords)
    
    if wants_adjustment:
        # Use Claude API to analyze request and suggest changes
        system_prompt = f"""You are a portfolio advisor AI. The user has a portfolio with these allocations:

{current_summary}

Current metrics:
- Risk Profile: {rp}
- Growth: {gr:.2f}%
- Sharpe Ratio: {shr:.2f}

Recent market headlines:
{hl}

The user wants to adjust their portfolio. You should:
1. Understand their intent (e.g., "add safe stocks", "increase equity exposure", "add high performers")
2. Suggest specific stock tickers based on real market data
3. Propose new allocation percentages
4. Explain your reasoning

Respond ONLY with valid JSON in this exact format:
{{
  "reasoning": "Brief explanation of your suggestions",
  "suggested_tickers": ["TICKER1", "TICKER2"],
  "allocation_changes": {{
    "Equities": 0.65,
    "Fixed Income": 0.25,
    "Alternatives": 0.10
  }},
  "specific_stocks": {{
    "TICKER1": "Brief reason (e.g., 'Defensive utility stock')",
    "TICKER2": "Brief reason"
  }}
}}

Guidelines for ticker selection:
- For "safe/defensive": Utilities (NEE, DUK), Consumer Staples (PG, KO, WMT), Healthcare (JNJ, UNH)
- For "growth/high return": Tech (NVDA, MSFT, GOOGL, AAPL, META, AMZN)
- For "recent performers": Check actual YTD performance data
- Only suggest well-known, liquid, large-cap stocks
- Limit to 2-5 ticker suggestions"""

        try:
            response = await fetch("https://api.anthropic.com/v1/messages", {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_query}]
                })
            })
            
            data = await response.json()
            ai_response = data.get("content", [{}])[0].get("text", "")
            
            # Parse AI response
            ai_json = json.loads(ai_response.strip())
            
            # Validate suggested tickers exist
            validated_tickers = []
            ticker_reasons = {}
            for ticker in ai_json.get("suggested_tickers", []):
                if _validate_ticker(ticker):
                    validated_tickers.append(ticker)
                    ticker_reasons[ticker] = ai_json.get("specific_stocks", {}).get(ticker, "Suggested stock")
            
            # Build response
            response_text = f"""**AI Portfolio Adjustment**

**Your Request:** {user_query}

**My Analysis:**
{ai_json.get('reasoning', 'Based on your request, here are my suggestions.')}

**Suggested Changes:**
"""
            
            if validated_tickers:
                response_text += "\n**Stocks to Add:**\n"
                for ticker in validated_tickers:
                    response_text += f"- **{ticker}**: {ticker_reasons[ticker]}\n"
            
            new_alloc = ai_json.get("allocation_changes", {})
            if new_alloc:
                response_text += f"\n**Proposed Allocation:**\n"
                response_text += f"- Equities: {new_alloc.get('Equities', 0)*100:.0f}%\n"
                response_text += f"- Fixed Income: {new_alloc.get('Fixed Income', 0)*100:.0f}%\n"
                response_text += f"- Alternatives: {new_alloc.get('Alternatives', 0)*100:.0f}%\n"
            
            response_text += "\n*Click 'Apply Suggestions' below to update your portfolio.*"
            
            return (response_text, new_alloc, validated_tickers)
            
        except Exception as e:
            return (_fallback_analysis(user_query, current_allocations, rp), None, [])
    
    else:
        # Regular advisory (no changes)
        txt = (
            f"**AI Portfolio Advisor**\n\n"
            f"**Your Question:** {user_query}\n\n"
            f"**Portfolio Metrics:**\n"
            f"- Risk Profile: **{rp}**\n"
            f"- Growth: **{gr:.2f}%**\n"
            f"- Sharpe Ratio: **{shr:.2f}**\n\n"
            f"**Recent Market Headlines:**\n{hl}\n\n"
            f"**Advisory Insight:**\n"
            f"Your portfolio maintains a {rp.lower()} approach with current growth of {gr:.2f}%. "
            f"Monitor sector rotation and consider rebalancing if equity allocation drifts significantly. "
            f"Maintain diversification across asset classes."
        )
        return (txt, None, [])


def _summarize_portfolio(allocations):
    """Create readable summary of current portfolio."""
    if not allocations:
        return "Empty portfolio"
    
    summary = []
    for asset_class, holdings in allocations.items():
        if holdings:
            total = sum(holdings.values()) * 100
            tickers = list(holdings.keys())
            summary.append(f"- {asset_class} ({total:.0f}%): {', '.join(tickers)}")
    return '\n'.join(summary)


def _validate_ticker(ticker):
    """Quick validation that a ticker exists and has data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return not hist.empty
    except:
        return False


def _fallback_analysis(query, allocations, risk_profile):
    """Fallback analysis if API fails."""
    query_lower = query.lower()
    
    response = f"**Portfolio Advisory**\n\n**Your Request:** {query}\n\n"
    
    # Simple keyword-based suggestions
    if any(word in query_lower for word in ['safe', 'defensive', 'conservative', 'protect']):
        response += """**Suggestion:** Consider defensive stocks:
- **JNJ** (Johnson & Johnson): Healthcare, stable dividends
- **PG** (Procter & Gamble): Consumer staples, recession-resistant
- **KO** (Coca-Cola): Established brand, consistent returns

These are traditionally lower-volatility stocks suitable for conservative portfolios.
"""
    
    elif any(word in query_lower for word in ['growth', 'aggressive', 'high return', 'tech']):
        response += """**Suggestion:** Consider growth stocks:
- **NVDA** (NVIDIA): AI/chip leader, high growth potential
- **MSFT** (Microsoft): Diversified tech giant, cloud leader
- **GOOGL** (Alphabet): Dominant search and cloud platform

These stocks offer higher growth potential but with increased volatility.
"""
    
    elif 'recent' in query_lower or 'perform' in query_lower:
        response += """**Suggestion:** Recent strong performers (check current data):
- **NVDA**, **MSFT**, **META**: Tech sector leaders
- **AAPL**: Consistent performance, strong brand
- **AMZN**: E-commerce and cloud dominance

Review current market conditions before adding.
"""
    
    else:
        response += f"""Based on your {risk_profile} risk profile, I recommend:
- Maintaining diversification across asset classes
- Regular rebalancing (quarterly or semi-annually)
- Monitoring individual position sizes (no single stock >10% ideally)

For specific stock additions, please specify your preferences (e.g., "add safe stocks" or "add growth stocks").
"""
    
    return response


# Synchronous wrapper for Streamlit
def get_ai_feedback(info, user_query="", current_allocations=None):
    """
    Synchronous version for Streamlit compatibility.
    Note: This is a simplified version. Full AI integration requires async support.
    """
    return _fallback_analysis(user_query, current_allocations, info.get('risk_profile', 'Balanced')), None, []
