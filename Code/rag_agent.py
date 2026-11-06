"""Local RAG over company reports using Ollama embeddings and Pinecone."""

import argparse
import ast
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from document_indexing import indexing_reports
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone
from selection import report_selection_logic

load_dotenv(Path(__file__).with_name(".env"), override=True)


def pinecone_index_name(company_name: str) -> str:
    """Name of the Pinecone index containing local nomic embeddings."""
    return company_name.lower().replace(" ", "-") + "-local-v2"


def _report_names(year: str) -> list[str]:
    try:
        start, end = (int(part) for part in year.split("-", maxsplit=1))
    except ValueError as error:
        raise ValueError("year must have the form YY-YY, for example 25-26") from error
    # Benchmark rows represent full financial years. Quarterly filings are
    # intentionally excluded so a top-k query cannot mix periods.
    return [f"A-{start}-{end}"]


def _period_label(year: str) -> str:
    start, end = (int(part) for part in year.split("-", maxsplit=1))
    return f"FY20{start:02d}-20{end:02d}"


_SCALE_FACTORS = {
    "unit": Decimal("1"),
    "thousand": Decimal("1000"),
    "lakh": Decimal("100000"),
    "million": Decimal("1000000"),
    "crore": Decimal("10000000"),
    "billion": Decimal("1000000000"),
}


def _decode_dict(candidate: str) -> dict | None:
    candidate = candidate.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:].strip()
        if candidate.startswith("python") or candidate.startswith("json"):
            candidate = candidate.split("\n", maxsplit=1)[-1].strip()
        candidate = candidate.removesuffix("```").strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(candidate)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _currency_code(*values: object) -> str:
    explicit_codes = ("INR", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "SGD")
    for value in values:
        text = str(value or "")
        for code in explicit_codes:
            if re.search(rf"\b{code}\b", text, flags=re.IGNORECASE):
                return code
    text = " ".join(str(value or "") for value in values).lower()
    symbols = (
        ("INR", r"₹|\brupees?\b|\brs\.?\b"),
        ("USD", r"us\$|\$"),
        ("EUR", r"€|\beuros?\b"),
        ("GBP", r"£|\bpounds?\b"),
        ("JPY", r"¥|\byen\b"),
        ("CAD", r"c\$"),
        ("AUD", r"a\$"),
        ("SGD", r"s\$"),
    )
    for code, pattern in symbols:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return code
    return "unknown"


def _scale_name(*values: object) -> str:
    patterns = (
        ("crore", r"\bcrores?\b|\bcrs?\.?\b"),
        ("lakh", r"\blakhs?\b|\blacs?\b"),
        ("billion", r"\bbillions?\b|\bbn\b"),
        ("million", r"\bmillions?\b|\bmn\b"),
        ("thousand", r"\bthousands?\b|\b000s\b"),
    )
    for value in values:
        text = str(value or "").lower()
        if text.strip() in {"unit", "units", "actual", "ones", "rupees", "dollars"}:
            return "unit"
        matches = {
            name for name, pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)
        }
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            return "unknown"
    return "unknown"


def _decimal_amount(value: str) -> Decimal | None:
    if not value or value.strip().lower() in {"not found", "n/a", "na", "none", "null", "-"}:
        return None
    negative = bool(re.search(r"\(\s*[^)]*\d[^)]*\)", value))
    match = re.search(r"[-+]?\s*\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    try:
        amount = Decimal(match.group(0).replace(",", "").replace(" ", ""))
    except InvalidOperation:
        return None
    return -abs(amount) if negative else amount


def get_rag_tools(company_name: str, *, pinecone_api_key: str | None = None):
    """Build the three statement-retrieval tools for an already indexed company."""
    embedding_model = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), keep_alive=0
    )
    llm = ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        temperature=0,
        keep_alive=0,
        reasoning=False,
        # Seven-field statement JSON exceeds 256 tokens even without reasoning.
        num_predict=1024,
        format="json",
        # qwen3:8b can exceed the available GPU memory alongside the desktop.
        # CPU inference is slower but makes this local pipeline dependable.
        num_gpu=int(os.getenv("OLLAMA_NUM_GPU", "0")),
    )
    api_key = pinecone_api_key if pinecone_api_key is not None else os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Add a Pinecone API key in Financial Intelligence settings or configure PINECONE_API_KEY."
        )
    index = Pinecone(api_key=api_key).Index(pinecone_index_name(company_name))

    def retrieve_and_extract(year: str, section: str, query: str, fields: str) -> str:
        chunks = index.query(
            vector=embedding_model.embed_query(query),
            filter={
                "$and": [
                    {"report_name": {"$in": _report_names(year)}},
                    {"section": {"$eq": section}},
                ]
            },
            top_k=5,
            include_metadata=True,
        )
        source_evidence = []
        for position, match in enumerate(chunks.matches, start=1):
            metadata = match.metadata or {}
            page = metadata.get("page")
            try:
                page = int(page) + 1
            except (TypeError, ValueError):
                page = None
            source_evidence.append(
                {
                    "id": f"E{position}",
                    "report": metadata.get("report_name", "unknown report"),
                    "page": page,
                    "source": Path(str(metadata.get("source", ""))).name or "unknown file",
                    "text": str(metadata.get("text", ""))[:1600],
                }
            )
        source = "\n\n".join(
            f"[{item['id']}] Report: {item['report']}; page: {item['page'] or 'unknown'}\n"
            f"{item['text']}"
            for item in source_evidence
        )
        print(f"{section}: retrieved {len(chunks.matches)} matching chunks for {year}.")
        if not source.strip():
            return json.dumps(
                {
                    "period": _period_label(year),
                    "values": {},
                    "sources": [],
                    "error": "No matching annual financial-statement text was found.",
                }
            )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You extract financial figures for comparisons. Use only the supplied source text. "
                        'Return JSON with this shape: {"period":"requested period","values": '
                        '{"field name":{"value":"exact reported value","currency":"ISO 4217 '
                        'code or unknown","scale":"unit|thousand|lakh|crore|million|billion|unknown",'
                        '"evidence_ids":["E1"]}}}. Include every requested field; use value '
                        '"Not Found" when absent. Use evidence_ids from the supplied excerpts. Do not '
                        "calculate or guess a currency, scale, period, or value. If a table heading states "
                        "a scale, report that scale for its figures. The excerpts have already been "
                        "filtered to the requested annual report. A missing year heading in a chunk "
                        "does not make a visible figure unavailable. Use the requested-year column "
                        "(normally the first numeric column), not the prior-year comparative. Match "
                        "equivalent labels such as 'net cash generated from / (used in) financing "
                        "activities' to the requested financing cash-flow field. Return a visible "
                        "figure with its evidence ID; use Not Found only when the metric is absent."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Requested annual financial year: {_period_label(year)}\n"
                        f"Fields: {fields}\n\nSource excerpts:\n{source}"
                    )
                ),
            ]
        )
        metadata = getattr(response, "response_metadata", {}) or {}
        if metadata.get("done_reason") == "length":
            raise RuntimeError(
                f"{section} extraction reached the model response limit. "
                "The incomplete JSON was not accepted as financial figures."
            )
        model_result = response.content
        if not isinstance(model_result, str):
            model_result = "".join(getattr(part, "text", str(part)) for part in model_result)
        extracted = _decode_dict(model_result) or {"raw_extraction": model_result.strip()}
        values = extracted.get("values", extracted)
        if not isinstance(values, dict):
            values = {"raw_extraction": model_result.strip()}
        return json.dumps(
            {
                "period": extracted.get("period", _period_label(year)),
                "values": values,
                "sources": source_evidence,
            },
            ensure_ascii=False,
        )

    @tool
    def statement_of_profit_and_loss(year: str) -> str:
        """Retrieve profit-and-loss figures for a YY-YY financial year."""
        return retrieve_and_extract(
            year,
            "Consolidated Statement of Profit and Loss",
            f"consolidated statement of profit and loss financial year 20{year}",
            "Revenue; Total Income; Total Expenses; Loss Before Tax; Profit Before Tax; "
            "Other Comprehensive Income; Other Comprehensive Loss (net of tax)",
        )

    @tool
    def statement_of_cash_flow(year: str) -> str:
        """Retrieve operating, financing and investing cash flow plus cash balances for a YY-YY financial year."""
        return retrieve_and_extract(
            year,
            "Consolidated Statement of Cash Flow",
            f"consolidated statement of cash flow financial year 20{year}",
            "Net cash used in operating activities; Net cash used in financing activities; "
            "Net cash used in investing activities; Opening balance of cash and cash equivalents; "
            "Cash and cash equivalents at the beginning of the period; Closing balance of cash "
            "and cash equivalents; Cash and cash equivalents at the end of the year",
        )

    @tool
    def balance_sheet(year: str) -> str:
        """Retrieve Total Assets, Total Equity, Total Liabilities, and Total Equity and Liabilities for a YY-YY financial year."""
        return retrieve_and_extract(
            year,
            "Consolidated Balance Sheet",
            f"consolidated balance sheet as at 31 March 20{year.split('-')[1]}",
            "Total Assets; Total Equity; Total Liabilities; Total Equity and Liabilities",
        )

    return [statement_of_profit_and_loss, statement_of_cash_flow, balance_sheet]


def test_tools(company_name: str, years: list[str]) -> None:
    """Invoke every retrieval tool directly as an end-to-end health check."""
    for year in years:
        print(f"\n=== Direct RAG tool test: {company_name}, FY {year} ===")
        for rag_tool in get_rag_tools(company_name):
            print(f"\n[{rag_tool.name}]")
            print(rag_tool.invoke({"year": year}))


def _canonical_period(value: object, fallback: str) -> str:
    if not value:
        return fallback
    years = re.findall(r"(?<!\d)(?:20)?\d{2}(?!\d)", str(value))
    if len(years) >= 2:
        first, second = years[:2]
        first = int(first) if len(first) == 4 else 2000 + int(first)
        second = int(second) if len(second) == 4 else 2000 + int(second)
        return f"FY{first}-{second}"
    return str(value).strip()


def parse_extracted_values(result: str | dict, expected_year: str | None = None) -> dict:
    """Parse statement values and attach normalized amounts plus page evidence."""
    candidate = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    payload = _decode_dict(candidate)
    if payload is None:
        return {"_raw_extraction": candidate.strip(), "_period": "unknown", "_sources": []}

    nested = payload.get("extraction")
    if isinstance(nested, str):
        payload = _decode_dict(nested) or {"raw_extraction": nested}
    raw_values = payload.get("values")
    if not isinstance(raw_values, dict):
        raw_values = {
            key: value
            for key, value in payload.items()
            if key not in {"period", "sources", "currency", "scale", "error"}
        }
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    expected_period = _period_label(expected_year) if expected_year else "unknown"
    period = _canonical_period(payload.get("period"), expected_period)
    source_text = "\n".join(str(item.get("text", "")) for item in sources if isinstance(item, dict))
    if not raw_values:
        result_values = {"_period": period, "_sources": sources}
        if payload.get("error"):
            result_values["_error"] = str(payload["error"])
        if payload.get("raw_extraction"):
            result_values["_raw_extraction"] = str(payload["raw_extraction"])
        return result_values

    normalized = {"_period": period, "_sources": sources}
    for field_name, field in raw_values.items():
        if str(field_name).startswith("_"):
            continue
        details = field if isinstance(field, dict) else {"value": field}
        reported = details.get("value", details.get("reported_value", details.get("amount")))
        reported = "Not Found" if reported is None else str(reported).strip()
        # Currency and scale must be explicit in the filing, not inferred by the LLM.
        currency = _currency_code(reported, source_text)
        scale = _scale_name(source_text)
        amount = _decimal_amount(reported)
        normalized_amount = None
        if amount is not None and currency != "unknown" and scale != "unknown":
            normalized_amount = format(amount * _SCALE_FACTORS[scale], "f")

        ids = details.get("evidence_ids", details.get("source_ids", []))
        if isinstance(ids, str):
            ids = [ids]
        matched_sources = [
            item for item in sources if isinstance(item, dict) and item.get("id") in ids
        ]
        evidence_amounts = {
            _decimal_amount(match.group())
            for source in matched_sources
            for match in re.finditer(
                r"\(?[-+]?\s*\d[\d,]*(?:\.\d+)?\)?", str(source.get("text", ""))
            )
        }
        verified = amount is not None and amount in evidence_amounts
        evidence_quality = "matched" if verified else "unverified"
        if not verified:
            normalized_amount = None
        if not matched_sources:
            matched_sources = [item for item in sources if isinstance(item, dict)]
        item_period = _canonical_period(details.get("period") or period, period)
        normalized[str(field_name)] = {
            "reported_value": reported,
            "normalized_value": normalized_amount,
            "normalized_currency": currency if normalized_amount is not None else "unknown",
            "currency": currency,
            "scale": scale,
            "period": item_period,
            "sources": matched_sources,
            "evidence_quality": evidence_quality,
        }
    return normalized


def collect_company_financials(
    company_name: str, years: list[str], *, pinecone_api_key: str | None = None
) -> dict:
    """Index one company if needed and collect all statement extracts by year."""
    selected_reports = report_selection_logic().report_selection_function(company_name)
    requested_reports = {name for year in years for name in _report_names(year)}
    selected_reports = {
        name: url for name, url in selected_reports.items() if name in requested_reports
    }
    missing_reports = requested_reports - selected_reports.keys()
    if missing_reports:
        raise ValueError(
            f"Annual reports unavailable for {company_name}: {', '.join(sorted(missing_reports))}"
        )
    if not selected_reports:
        raise ValueError(f"No reports are available for {company_name!r}")

    indexing_reports(company_name, pinecone_api_key=pinecone_api_key).load_reports(selected_reports)
    tools = get_rag_tools(company_name, pinecone_api_key=pinecone_api_key)
    company_results = {}
    for year in years:
        company_results[year] = {}
        for rag_tool in tools:
            result = rag_tool.invoke({"year": year})
            company_results[year][rag_tool.name] = parse_extracted_values(
                result, expected_year=year
            )
    return company_results


def save_benchmark(results: dict, output_path: str | Path) -> Path:
    """Write benchmark results as UTF-8 JSON, creating the output directory."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class rag_agent:  # noqa: N801 - retained as a public CLI class name
    def __init__(self, company_name: str, financial_years: list[str]):
        self.company_name = company_name
        self.financial_years = financial_years

    def main(self) -> dict:
        results = collect_company_financials(self.company_name, self.financial_years)
        for year, statements in results.items():
            print(f"\n=== Agent result: {self.company_name}, FY {year} ===")
            print(json.dumps(statements, indent=2, ensure_ascii=False))
        return results


class financial_benchmark:  # noqa: N801 - retained as a public API class name
    """Collect comparable statement extracts for multiple companies and years."""

    def __init__(
        self,
        company_names: list[str],
        financial_years: list[str],
        *,
        pinecone_api_key: str | None = None,
    ):
        self.company_names = list(
            dict.fromkeys(name.strip() for name in company_names if name.strip())
        )
        self.financial_years = financial_years
        self.pinecone_api_key = pinecone_api_key
        if len(self.company_names) < 2:
            raise ValueError("Benchmark mode requires at least two company names")

    def run(self) -> dict:
        results = {
            "financial_years": self.financial_years,
            "companies": {},
            "errors": {},
        }
        for company_name in self.company_names:
            print(f"\n=== Benchmarking {company_name} ===")
            try:
                results["companies"][company_name] = collect_company_financials(
                    company_name, self.financial_years, pinecone_api_key=self.pinecone_api_key
                )
            except Exception as error:
                # Preserve partial benchmark results when one company's report
                # catalog or vector index is unavailable.
                results["errors"][company_name] = f"{type(error).__name__}: {error}"
                print(f"Unable to benchmark {company_name}: {error}")
        comparison = {}
        for year in self.financial_years:
            comparison[year] = {}
            for company_name, company_data in results["companies"].items():
                for statement_name, fields in company_data.get(year, {}).items():
                    statement_view = comparison[year].setdefault(statement_name, {})
                    for field_name, value in fields.items():
                        if field_name.startswith("_"):
                            continue
                        statement_view.setdefault(field_name, {"values": {}})["values"][
                            company_name
                        ] = value

            for statement_view in comparison[year].values():
                for _field_name, field_data in statement_view.items():
                    values = field_data["values"]
                    entries = [
                        value
                        for value in values.values()
                        if isinstance(value, dict) and value.get("normalized_value") is not None
                    ]
                    currencies = {value.get("normalized_currency") for value in entries}
                    periods = {
                        value.get("period") for value in values.values() if isinstance(value, dict)
                    }
                    period_matches = periods == {_period_label(year)}
                    is_comparable = (
                        len(values) >= 2
                        and len(values) == len(self.company_names)
                        and len(entries) == len(values)
                        and len(currencies) == 1
                        and "unknown" not in currencies
                        and period_matches
                    )
                    if is_comparable:
                        field_data["comparable"] = True
                        field_data["comparison_currency"] = next(iter(currencies))
                        field_data["comparison_note"] = (
                            f"All values normalized to {field_data['comparison_currency']} base units "
                            f"for {_period_label(year)}."
                        )
                    else:
                        reasons = []
                        if len(values) < len(self.company_names) or len(values) < 2:
                            reasons.append("fewer than two companies or missing company data")
                        if len(currencies) > 1:
                            reasons.append("reporting currencies differ")
                        elif not currencies or "unknown" in currencies:
                            reasons.append("currency or reporting scale is unknown")
                        if not period_matches:
                            reasons.append("reporting periods differ")
                        field_data["comparable"] = False
                        field_data["comparison_currency"] = None
                        field_data["comparison_note"] = "; ".join(reasons) or (
                            "one or more values could not be normalized"
                        )
        results["comparison"] = comparison
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the local Ollama + Pinecone financial RAG agent."
    )
    parser.add_argument("--company", default="Swiggy")
    parser.add_argument("--years", nargs="+", default=["25-26", "24-25"])
    parser.add_argument(
        "--compare-companies",
        nargs="+",
        metavar="COMPANY",
        help="Benchmark two or more companies using the same statement retrieval workflow.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Defaults to Reports/financial_benchmark.json in benchmark mode.",
    )
    parser.add_argument(
        "--test-tools", action="store_true", help="Call the three tools without indexing."
    )
    args = parser.parse_args()
    if args.test_tools:
        test_tools(args.company, args.years)
    elif args.compare_companies:
        benchmark = financial_benchmark(args.compare_companies, args.years)
        results = benchmark.run()
        output_path = save_benchmark(
            results,
            args.output
            or Path(__file__).resolve().parent.parent / "Reports" / "financial_benchmark.json",
        )
        print(f"\nBenchmark saved to {output_path}")
    else:
        agent = rag_agent(args.company, args.years)
        results = agent.main()
        if args.output:
            output_path = save_benchmark({args.company: results}, args.output)
            print(f"\nFinancial extracts saved to {output_path}")
