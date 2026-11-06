from unittest.mock import Mock

import pandas as pd
import pytest

import market_data


def test_price_history_removes_missing_and_duplicate_closes(monkeypatch):
    frame = pd.DataFrame(
        {"Close": [101.0, None, 100.0, 102.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]),
    )
    ticker = Mock()
    ticker.history.return_value = frame
    ticker.fast_info = {"lastPrice": float("nan")}
    monkeypatch.setattr(market_data.yf, "Ticker", Mock(return_value=ticker))
    result = market_data.load_market_data("NVDA", "1y")
    assert result["history"]["Close"].tolist() == [100.0, 102.0]
    assert result["latest_price"] == 102.0


def test_unknown_company_does_not_silently_use_unrelated_symbol(monkeypatch):
    monkeypatch.setattr(market_data, "Search", Mock(return_value=Mock(quotes=[])))
    with pytest.raises(ValueError, match="No ticker"):
        market_data.load_market_data("Unknown company", "1mo")
