"""Market-data lookup shared by the dashboard and validation checks."""

import math
import re

import yfinance as yf
from yfinance import Search


def load_market_data(query: str, period: str) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("Enter a company or ticker.")
    if period not in {"1mo", "3mo", "6mo", "1y", "2y", "5y"}:
        raise ValueError("Unsupported historical range.")
    if re.fullmatch(r"[A-Z^][A-Z0-9.^=-]{0,19}", query):
        symbol = query
    else:
        quotes = Search(query).quotes
        if not quotes:
            raise ValueError(f"No ticker found for {query!r}.")
        symbol = quotes[0]["symbol"]
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period)
    if history.empty or "Close" not in history:
        raise ValueError(f"No price history returned for {symbol}.")
    history = history.dropna(subset=["Close"]).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    if history.empty:
        raise ValueError(f"No valid closing prices returned for {symbol}.")
    latest = ticker.fast_info.get("lastPrice")
    if latest is None or not math.isfinite(float(latest)):
        latest = float(history["Close"].iloc[-1])
    return {"symbol": symbol, "history": history, "latest_price": float(latest)}
