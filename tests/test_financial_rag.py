"""Regression coverage for annual-period filtering and filing evidence normalization."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

rag_agent = importlib.import_module("rag_agent")


def test_session_key_reaches_indexing_and_retrieval_without_changing_environment(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "deployment-key")
    selection = Mock()
    selection.report_selection_function.return_value = {"A-25-26": "report.pdf"}
    monkeypatch.setattr(rag_agent, "report_selection_logic", lambda: selection)
    indexer = Mock()
    indexing_factory = Mock(return_value=indexer)
    tools_factory = Mock(return_value=[])
    monkeypatch.setattr(rag_agent, "indexing_reports", indexing_factory)
    monkeypatch.setattr(rag_agent, "get_rag_tools", tools_factory)

    assert rag_agent.collect_company_financials(
        "Company A", ["25-26"], pinecone_api_key="session-key"
    ) == {"25-26": {}}
    indexing_factory.assert_called_once_with("Company A", pinecone_api_key="session-key")
    tools_factory.assert_called_once_with("Company A", pinecone_api_key="session-key")
    assert rag_agent.os.environ["PINECONE_API_KEY"] == "deployment-key"


def test_retrieval_uses_session_key_instead_of_deployment_key(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "deployment-key")
    monkeypatch.setattr(rag_agent, "OllamaEmbeddings", Mock())
    monkeypatch.setattr(rag_agent, "ChatOllama", Mock())
    pinecone = Mock()
    monkeypatch.setattr(rag_agent, "Pinecone", pinecone)
    tools = rag_agent.get_rag_tools("Company A", pinecone_api_key="session-key")
    pinecone.assert_called_once_with(api_key="session-key")
    assert len(tools) == 3


def test_benchmark_passes_session_key_to_each_company(monkeypatch):
    collect = Mock(return_value={})
    monkeypatch.setattr(rag_agent, "collect_company_financials", collect)
    result = rag_agent.financial_benchmark(
        ["Company A", "Company B"], ["25-26"], pinecone_api_key="session-key"
    ).run()
    assert not result["errors"]
    assert collect.call_count == 2
    assert all(
        call.kwargs == {"pinecone_api_key": "session-key"} for call in collect.call_args_list
    )


def test_truncated_extraction_is_rejected_instead_of_becoming_financial_data(monkeypatch):
    monkeypatch.setattr(rag_agent, "OllamaEmbeddings", Mock())
    chat = Mock()
    chat.invoke.return_value = SimpleNamespace(
        content='{"values":{"Revenue":', response_metadata={"done_reason": "length"}
    )
    monkeypatch.setattr(rag_agent, "ChatOllama", Mock(return_value=chat))
    index = Mock()
    index.query.return_value.matches = [
        SimpleNamespace(metadata={"report_name": "A-24-25", "page": 4, "text": "Revenue 100"})
    ]
    client = Mock()
    client.Index.return_value = index
    monkeypatch.setattr(rag_agent, "Pinecone", Mock(return_value=client))
    tool = rag_agent.get_rag_tools("Company A", pinecone_api_key="session-key")[0]
    with pytest.raises(RuntimeError, match="incomplete JSON was not accepted"):
        tool.invoke({"year": "24-25"})


def test_report_filter_keeps_annual_period_only():
    assert rag_agent._report_names("25-26") == ["A-25-26"]


def test_valid_citation_does_not_verify_an_invented_amount():
    value = rag_agent.parse_extracted_values(
        {
            "period": "FY2024-2025",
            "values": {
                "Revenue": {
                    "value": "999",
                    "currency": "INR",
                    "scale": "million",
                    "evidence_ids": ["E1"],
                }
            },
            "sources": [{"id": "E1", "text": "Revenue 100 INR million", "page": 10}],
        },
        "24-25",
    )["Revenue"]
    assert value["evidence_quality"] == "unverified"
    assert value["normalized_value"] is None


def test_partial_benchmark_is_not_claimed_comparable(monkeypatch):
    def collect(company, years, **kwargs):
        if company == "Missing company":
            raise ValueError("No filing")
        return {
            "24-25": {
                "balance_sheet": {
                    "Assets": {
                        "reported_value": "100",
                        "normalized_value": "100",
                        "normalized_currency": "INR",
                        "period": "FY2024-2025",
                    }
                }
            }
        }

    monkeypatch.setattr(rag_agent, "collect_company_financials", collect)
    result = rag_agent.financial_benchmark(["Swiggy", "Missing company"], ["24-25"]).run()
    assert "Missing company" in result["errors"]
    assert not result["comparison"]["24-25"]["balance_sheet"]["Assets"]["comparable"]


def test_extraction_normalizes_amount_and_retains_source_page():
    payload = {
        "period": "FY2025-2026",
        "values": {
            "Revenue": {
                "value": "₹ 1.25",
                "currency": "INR",
                "scale": "crore",
                "evidence_ids": ["E1"],
            }
        },
        "sources": [
            {
                "id": "E1",
                "report": "A-25-26",
                "page": 41,
                "source": "swiggy_annual.pdf",
                "text": "Revenue from operations 1.25; ₹ in crore.",
            }
        ],
    }

    parsed = rag_agent.parse_extracted_values(json.dumps(payload), expected_year="25-26")
    revenue = parsed["Revenue"]
    assert revenue["reported_value"] == "₹ 1.25"
    assert revenue["normalized_value"] == "12500000.00"
    assert revenue["normalized_currency"] == "INR"
    assert revenue["period"] == "FY2025-2026"
    assert revenue["evidence_quality"] == "matched"
    assert revenue["sources"][0]["page"] == 41


def test_benchmark_marks_mixed_currencies_not_comparable(monkeypatch):
    def company_data(company, _years, *, pinecone_api_key=None):
        currency, amount = ("INR", "100000") if company == "Company A" else ("USD", "1200")
        return {
            "25-26": {
                "statement": {
                    "_period": "FY2025-2026",
                    "Revenue": {
                        "reported_value": amount,
                        "normalized_value": amount,
                        "normalized_currency": currency,
                        "currency": currency,
                        "scale": "unit",
                        "period": "FY2025-2026",
                        "sources": [],
                    },
                }
            }
        }

    monkeypatch.setattr(rag_agent, "collect_company_financials", company_data)
    results = rag_agent.financial_benchmark(["Company A", "Company B"], ["25-26"]).run()
    revenue = results["comparison"]["25-26"]["statement"]["Revenue"]
    assert revenue["comparable"] is False
    assert revenue["comparison_currency"] is None
    assert "currencies differ" in revenue["comparison_note"]
