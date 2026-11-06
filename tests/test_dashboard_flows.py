"""Exercise dashboard interactions without changing real credentials or history."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Code")]


@pytest.fixture
def isolated_dashboard(monkeypatch, tmp_path):
    import credential_store
    import history_store

    monkeypatch.setenv("MARKET_ANALYST_HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setenv("APP_AUTH_PASSWORD", "")
    monkeypatch.setenv("APP_AUTH_REQUIRED", "false")
    monkeypatch.setattr(credential_store, "load_credentials", lambda _: {})
    monkeypatch.setattr(credential_store, "save_credentials", Mock())
    monkeypatch.setattr(credential_store, "clear_credentials", Mock())
    monkeypatch.setattr(history_store, "load_initial_history", lambda **_: [])
    return tmp_path


def button(app, label):
    return next(item for item in app.button if item.label == label)


def field(app, label):
    return next(item for item in app.text_input if item.label == label)


def test_overview_validation_and_completed_report_controls(isolated_dashboard):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    button(app, "Run analysis").click().run()
    assert any("enter a company" in item.value.lower() for item in app.warning)
    assert [item.label for item in app.toggle] == ["Deep Dive"]
    app.toggle[0].set_value(True).run()
    assert app.toggle[0].value
    app.session_state.report_markdown = (
        "# Test report\n\n## SWOT Analysis\nSupported report content."
    )
    app.session_state.topic_analyzed = "Test company"
    app.session_state.agent_steps = [{"agent": "Researcher", "time": "12:00", "msg": "Completed"}]
    app.run()
    assert not app.get("tab")
    logs = [item for item in app.get("expander") if item.label == "Agent activity log"]
    assert len(logs) == 1 and not logs[0].proto.expanded
    assert len(app.get("download_button")) == 2
    button(app, "New report").click().run()
    assert not app.session_state.report_markdown
    assert not app.exception


def test_financial_market_news_results_survive_reruns(isolated_dashboard, monkeypatch):
    import news_twitter_api
    import rag_agent

    import market_data

    history = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.date_range("2026-01-01", periods=2))
    lookup = Mock(return_value={"symbol": "NVDA", "history": history, "latest_price": 101.0})
    monkeypatch.setattr(market_data, "load_market_data", lookup)
    news = Mock(
        return_value=[{"title": "Source article", "url": "https://example.org/news", "score": 0.9}]
    )
    monkeypatch.setattr(news_twitter_api, "get_news", Mock(invoke=news))
    collect = Mock(return_value={"24-25": {"balance_sheet": {}}})
    monkeypatch.setattr(rag_agent, "collect_company_financials", collect)
    monkeypatch.setattr(rag_agent, "save_benchmark", lambda results, path: path)
    app = AppTest.from_file(str(ROOT / "pages/2_Financial_Intelligence.py"), default_timeout=15)
    app.session_state.session_tavily_api_key = "test-session-tavily"
    app.session_state.session_pinecone_api_key = "test-session-pinecone"
    app.run()
    assert not app.exception
    field(app, "Company or ticker").set_value("NVDA")
    button(app, "Load market data").click().run()
    assert app.session_state.market_result["symbol"] == "NVDA"
    field(app, "Company name").set_value("NVIDIA")
    button(app, "Search news").click().run()
    app.run()
    assert lookup.call_count == 1 and news.call_count == 1
    assert len(app.dataframe) == 2
    field(app, "Company names (comma separated)").set_value("Swiggy")
    field(app, "Financial years (YY-YY, comma separated)").set_value("25-25")
    button(app, "Run filing analysis").click().run()
    assert any("consecutive" in item.value for item in app.error)
    collect.assert_not_called()
    field(app, "Financial years (YY-YY, comma separated)").set_value("24-25")
    button(app, "Run filing analysis").click().run()
    app.run()
    collect.assert_called_once_with("Swiggy", ["24-25"], pinecone_api_key="test-session-pinecone")
    assert app.session_state.financial_result["results"]["companies"]
    assert not app.exception


def test_hosted_authentication_and_logout(monkeypatch, isolated_dashboard):
    monkeypatch.setenv("APP_AUTH_REQUIRED", "true")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert any("locked" in item.value for item in app.error)
    monkeypatch.setenv("APP_AUTH_PASSWORD", "test-only-access-password")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    field(app, "Access password").set_value("wrong")
    button(app, "Sign in").click().run()
    assert any("incorrect" in item.value for item in app.error)
    field(app, "Access password").set_value("test-only-access-password")
    button(app, "Sign in").click().run()
    app.run()
    assert app.session_state.app_authenticated
    button(app, "Log out").click().run()
    assert not app.session_state.app_authenticated


def test_load_and_clear_history_and_report(isolated_dashboard, monkeypatch):
    import history_store

    records = [
        {
            "topic": name,
            "mode": "Custom LangChain Orchestrator",
            "timestamp": "06 Nov",
            "markdown": f"# {name}\n\n## SWOT Analysis\nArchived content.",
            "scores": {},
            "archived_example": True,
        }
        for name in ["Reliance Industries", "Broadcom", "NVIDIA"]
    ]
    monkeypatch.setattr(history_store, "load_initial_history", lambda **_: records)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    button(app, "Load report").click().run()
    assert app.session_state.topic_analyzed == "Reliance Industries"
    assert any("Archived AI-generated" in item.value for item in app.info)
    button(app, "Clear report").click().run()
    assert not app.session_state.report_markdown
    button(app, "Clear history").click().run()
    assert not app.session_state.history
    assert history_store.load_history() == []
