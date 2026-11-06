"""Streamlit page for filing-based financial analysis and live market context."""

import json
import logging
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(CODE_DIR / ".env", override=False)

from news_twitter_api import get_news  # noqa: E402

from auth import require_auth  # noqa: E402
from credential_store import clear_credentials, load_credentials, save_credentials  # noqa: E402
from market_data import load_market_data  # noqa: E402
from navigation import render_workspace_toggle  # noqa: E402

logger = logging.getLogger("financial_intelligence")
LOCAL_CREDENTIAL_PATH = PROJECT_ROOT / ".streamlit" / "api_credentials.dpapi"


def _local_credential_storage_available() -> bool:
    if get_script_run_ctx(suppress_warning=True) is None:
        return False
    address = str(st.get_option("server.address") or "").strip().lower()
    hosted = bool(
        os.environ.get("RENDER")
        or os.environ.get("APP_AUTH_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}
        or os.environ.get("APP_ENV", "").strip().lower() in {"prod", "production"}
    )
    return os.name == "nt" and address in {"127.0.0.1", "localhost", "::1"} and not hosted


def _forget_pinecone_if_disabled():
    if LOCAL_CREDENTIAL_STORAGE and not st.session_state.get("remember_pinecone_api_key", True):
        clear_credentials(LOCAL_CREDENTIAL_PATH, keys=("pinecone_api_key",))


st.set_page_config(page_title="Financial Intelligence", layout="wide")
theme_path = PROJECT_ROOT / "styles.css"
if theme_path.is_file():
    st.markdown(f"<style>{theme_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
require_auth()
LOCAL_CREDENTIAL_STORAGE = _local_credential_storage_available()
if "session_pinecone_api_key" not in st.session_state:
    saved_credentials = {}
    if LOCAL_CREDENTIAL_STORAGE:
        try:
            saved_credentials = load_credentials(LOCAL_CREDENTIAL_PATH)
        except Exception as error:
            logger.warning("Could not load encrypted financial API key: %s", error)
    st.session_state.session_pinecone_api_key = saved_credentials.get(
        "pinecone_api_key"
    ) or os.environ.get("PINECONE_API_KEY", "")
st.session_state.setdefault("remember_pinecone_api_key", True)

if "session_tavily_api_key" not in st.session_state:
    research_credentials = (
        load_credentials(LOCAL_CREDENTIAL_PATH) if LOCAL_CREDENTIAL_STORAGE else {}
    )
    for state_key, stored_key, env_key in (
        ("session_tavily_api_key", "tavily_api_key", "TAVILY_API_KEY"),
        ("session_gemini_api_key", "gemini_api_key", "GEMINI_API_KEY"),
    ):
        st.session_state[state_key] = research_credentials.get(stored_key) or os.environ.get(
            env_key, ""
        )

# Keep Overview's widget-backed keys available to the News tab across reruns.
for credential_key in ("session_gemini_api_key", "session_tavily_api_key", "remember_api_keys"):
    if credential_key in st.session_state:
        st.session_state[credential_key] = st.session_state[credential_key]
st.title("Financial & Market Intelligence")
st.caption(
    "Extend competitor research with filing-grounded statements, cross-company benchmarks, "
    "market prices, history, and recent news."
)
render_workspace_toggle("Financial intelligence")

with st.expander("API keys and model settings"):
    if LOCAL_CREDENTIAL_STORAGE:
        remember_pinecone = st.checkbox(
            "Remember Pinecone API key on this device",
            key="remember_pinecone_api_key",
            on_change=_forget_pinecone_if_disabled,
            help="Encrypts the key for your Windows account and saves it only for this local app.",
        )
    else:
        remember_pinecone = False
        st.caption("The key stays in this browser session. Use deployment secrets for hosted keys.")
    pinecone_key = st.text_input(
        "Pinecone API Key",
        type="password",
        key="session_pinecone_api_key",
        help="Enter the API key from your Pinecone project to index and query financial filings.",
    ).strip()
    st.caption("Filing analysis uses Ollama for local chat and embeddings.")
    if LOCAL_CREDENTIAL_STORAGE:
        try:
            if remember_pinecone and pinecone_key:
                save_credentials(LOCAL_CREDENTIAL_PATH, {"pinecone_api_key": pinecone_key})
                st.caption("Pinecone API key is saved encrypted for this Windows account.")
            elif not remember_pinecone:
                st.caption("Pinecone API key will only be kept for this browser session.")
            else:
                st.caption("The key will be remembered after you enter it.")
        except Exception as error:
            logger.warning("Could not save encrypted financial API key: %s", error)
            st.warning("Could not remember the key securely. It will remain in this session only.")

effective_pinecone_key = pinecone_key or os.environ.get("PINECONE_API_KEY", "").strip()
if effective_pinecone_key:
    st.caption("Pinecone key available")
else:
    st.caption(
        "Missing: Pinecone. Add a key in API keys and model settings to run filing analysis."
    )

financial_tab, market_tab, news_tab = st.tabs(["Filings", "Market", "News"])

with financial_tab:
    st.subheader("Extract and compare financial statements")
    st.write(
        "The filing pipeline selects reports from `Code/report_data.json`, downloads and indexes "
        "them when needed, then extracts profit and loss, cash flow, and balance sheet figures "
        "using Ollama and Pinecone."
    )
    with st.form("financial_benchmark_form"):
        companies_text = st.text_input(
            "Company names (comma separated)",
            value="Reliance Industries, Swiggy",
            help="Enter one company for statement extraction or two or more for a benchmark.",
        )
        years_text = st.text_input("Financial years (YY-YY, comma separated)", value="25-26, 24-25")
        run_financials = st.form_submit_button("Run filing analysis", type="primary")

    if run_financials:
        companies = list(
            dict.fromkeys(name.strip() for name in companies_text.split(",") if name.strip())
        )
        years = list(dict.fromkeys(year.strip() for year in years_text.split(",") if year.strip()))
        invalid_years = [
            year
            for year in years
            if not re.fullmatch(r"\d{2}-\d{2}", year) or (int(year[3:]) - int(year[:2])) % 100 != 1
        ]
        if not companies:
            st.error("Enter at least one company.")
        elif not years or invalid_years:
            st.error("Use consecutive financial years in YY-YY format, such as 25-26.")
        elif not effective_pinecone_key:
            st.error(
                "Add a Pinecone API key in API keys and model settings before running filing analysis."
            )
        else:
            try:
                from rag_agent import (  # noqa: PLC0415
                    collect_company_financials,
                    financial_benchmark,
                    save_benchmark,
                )

                with st.spinner(
                    "Collecting filings and extracting statements. This may download PDFs "
                    "and call Ollama and Pinecone."
                ):
                    if len(companies) > 1:
                        results = financial_benchmark(
                            companies, years, pinecone_api_key=effective_pinecone_key
                        ).run()
                    else:
                        company_results = collect_company_financials(
                            companies[0], years, pinecone_api_key=effective_pinecone_key
                        )
                        results = {
                            "financial_years": years,
                            "companies": {companies[0]: company_results},
                            "errors": {},
                        }
                    output = save_benchmark(
                        results, PROJECT_ROOT / "Reports" / "financial_benchmark.json"
                    )
                st.session_state.financial_result = {"results": results, "output": str(output)}
            except Exception as error:
                st.error(f"Financial analysis failed: {type(error).__name__}: {error}")

    if st.session_state.get("financial_result"):
        results = st.session_state.financial_result["results"]
        output = Path(st.session_state.financial_result["output"])
        if results.get("companies"):
            st.success(f"Analysis saved to {output.relative_to(PROJECT_ROOT)}")
        else:
            st.error("No company could be analyzed. Review the errors below.")
        for company, error in results.get("errors", {}).items():
            st.warning(f"{company}: {error}")
        comparison = results.get("comparison")
        if comparison:
            st.subheader("Cross-company comparison")
            for year, statements in comparison.items():
                st.markdown(f"#### FY {year}")
                for statement, fields in statements.items():
                    st.markdown(f"**{statement}**")
                    for metric, field_data in fields.items():
                        if isinstance(field_data, dict) and "values" in field_data:
                            entries = field_data["values"]
                            comparable = field_data.get("comparable", False)
                            note = field_data.get("comparison_note", "Values are not comparable.")
                        else:
                            entries = field_data
                            comparable = False
                            note = "Legacy extraction has no normalized comparison metadata."
                        if comparable:
                            st.caption(field_data.get("comparison_note", note))
                        else:
                            st.warning(f"{metric}: {note}")

                        rows = []
                        for company, entry in entries.items():
                            if isinstance(entry, dict) and "reported_value" in entry:
                                reported = entry.get("reported_value", "Not Found")
                                amount = entry.get("normalized_value")
                                currency = entry.get("normalized_currency", "unknown")
                                if amount is not None:
                                    try:
                                        shown_amount = f"{Decimal(str(amount)):,.2f} {currency}"
                                    except (InvalidOperation, ValueError):
                                        shown_amount = f"{amount} {currency}"
                                else:
                                    shown_amount = "Not normalized"
                                source_refs = entry.get("sources", [])
                                source_label = (
                                    "; ".join(
                                        f"{source.get('report', 'report')}"
                                        f" p.{source.get('page', '?')}"
                                        for source in source_refs
                                        if isinstance(source, dict)
                                    )
                                    or "No page evidence"
                                )
                                period_label = entry.get("period", year)
                            else:
                                reported = entry
                                shown_amount = "Not normalized"
                                currency = "unknown"
                                source_refs = []
                                source_label = "No page evidence"
                                period_label = year
                            rows.append(
                                {
                                    "Company": company,
                                    "Reported value": reported,
                                    "Normalized base amount": shown_amount,
                                    "Currency / scale": (
                                        f"{entry.get('currency', currency)} / {entry.get('scale', 'unknown')}"
                                        if isinstance(entry, dict)
                                        else currency
                                    ),
                                    "Period": period_label,
                                    "Filing pages": source_label,
                                    "Evidence": (
                                        "Matched page"
                                        if isinstance(entry, dict)
                                        and entry.get("evidence_quality") == "matched"
                                        else "Retrieved context"
                                    ),
                                }
                            )
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows), hide_index=True, use_container_width=True
                            )
                        evidence_sources = [
                            (company, source)
                            for company, entry in entries.items()
                            if isinstance(entry, dict)
                            for source in entry.get("sources", [])
                            if isinstance(source, dict)
                        ]
                        if evidence_sources:
                            with st.expander(f"Filing evidence — {metric}"):
                                for company, source in evidence_sources:
                                    st.markdown(
                                        f"**{company} · {source.get('report', 'report')} · "
                                        f"page {source.get('page', 'unknown')}**"
                                    )
                                    st.caption(source.get("source", ""))
                                    st.text(source.get("text", ""))
        with st.expander("Full extraction JSON"):
            st.json(results)
        st.download_button(
            "Download benchmark JSON",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name="financial_benchmark.json",
            mime="application/json",
        )

with market_tab:
    st.subheader("Price and historical performance")
    with st.form("market_data_form"):
        company_query = st.text_input(
            "Company or ticker", placeholder="e.g. Reliance Industries or RELIANCE.NS"
        )
        period = st.selectbox("Historical range", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        run_market = st.form_submit_button("Load market data")
    if run_market:
        try:
            st.session_state.market_result = load_market_data(company_query, period)
        except Exception as error:
            st.error(f"Market data lookup failed: {type(error).__name__}: {error}")
    if st.session_state.get("market_result"):
        market = st.session_state.market_result
        st.caption(f"Ticker: {market['symbol']}")
        st.caption("Closing price history")
        st.line_chart(market["history"]["Close"])
        st.dataframe(market["history"].tail(10), width="stretch")
        st.metric("Latest available price", f"{market['latest_price']:,.2f}")

with news_tab:
    st.subheader("Recent company news")
    with st.form("company_news_form"):
        news_company = st.text_input("Company name", placeholder="e.g. Reliance Industries")
        run_news = st.form_submit_button("Search news")
    if run_news:
        tavily_api_key = (
            st.session_state.get("session_tavily_api_key", "").strip()
            or os.environ.get("TAVILY_API_KEY", "").strip()
        )
        if not news_company.strip():
            st.warning("Enter a company name.")
        elif not tavily_api_key:
            st.error("Add a Tavily API key in the Overview page's API keys and model settings.")
        else:
            try:
                with st.spinner("Searching recent news with Tavily…"):
                    items = get_news.invoke(
                        {"company": news_company.strip(), "api_key": tavily_api_key}
                    )
                st.session_state.news_result = {"company": news_company.strip(), "items": items}
            except Exception as error:
                st.error(f"News search failed: {type(error).__name__}: {error}")

    if st.session_state.get("news_result"):
        news = st.session_state.news_result
        st.caption(f"News for {news['company']}")
        if news["items"]:
            st.dataframe(
                pd.DataFrame(news["items"]),
                column_config={"url": st.column_config.LinkColumn("Source")},
                width="stretch",
            )
        else:
            st.info("No news results were returned.")
