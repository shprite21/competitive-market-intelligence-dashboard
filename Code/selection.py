import json
import re
import warnings
from datetime import date
from pathlib import Path


class report_selection_logic:
    """
    Selects the relevant filing documents (annual report + trailing quarters)
    for a given company, based on today's date and report_data.json.
    """

    JSON_PATH = Path(__file__).with_name("report_data.json")
    MAX_LOOKBACK_YEARS = (
        3  # how many years back to search if current/prior year annual report is missing
    )

    def report_selection_function(self, company_name: str) -> dict:
        today = date.today()
        data = self._load_data()

        return_report_data = {}
        found = False
        cur_year = today.year
        iterations = 0

        # --- Step 1: find the current-year and/or prior-year annual report,
        #             falling back further back in time if neither exists ---
        while not found and iterations < self.MAX_LOOKBACK_YEARS:
            for report in data:
                if report["company_name"] != company_name:
                    continue

                if report["url"] is None:
                    if today > date.fromisoformat(report["published"]):
                        warnings.warn(
                            f"The json file is outdated -> {report['report_name']} missing from json file",
                            UserWarning,
                            stacklevel=2,
                        )
                    # else: not published yet, nothing to warn about, just skip
                    continue

                if report["report_name"].split("-")[0] != "A":
                    continue  # not an annual report, skip for this step

                report_year = int(report["report_name"].split("-")[2])

                if report_year == cur_year % 100 or report_year == (cur_year % 100) - 1:
                    return_report_data[report["report_name"]] = report["url"]
                    found = True
                    print(f"annual report found {report['report_name']} on iteration {iterations}")

            cur_year -= 1
            iterations += 1

        if not found:
            warnings.warn("Required annual report missing -> json file is corrupted", stacklevel=2)

        # --- Step 2: find all quarterly reports published after the most
        #             recent annual report's period-end, up to today ---
        end_period = date(date.today().year, 3, 31)
        print(f"end period of report: {end_period}")

        for report in data:
            if (
                report["company_name"] == company_name
                and report["url"] is not None
                and report["report_name"].split("-")[0] != "A"
                and date.fromisoformat(report["published"]) > end_period
                and date.fromisoformat(report["published"]) < today
            ):
                return_report_data[report["report_name"]] = report["url"]

        return return_report_data

    def _load_data(self) -> list:
        with open(self.JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return validate_report_catalog(data)


def validate_report_catalog(data: object) -> list[dict]:
    """Validate filing metadata once and report malformed records precisely."""
    if not isinstance(data, list):
        raise ValueError("report_data.json must contain a JSON array of filing records")

    seen_reports = set()
    for record_number, report in enumerate(data, start=1):
        label = f"report_data.json record {record_number}"
        if not isinstance(report, dict):
            raise ValueError(f"{label} must be a JSON object")

        for field in ("company_name", "report_name", "published"):
            if not isinstance(report.get(field), str) or not report[field].strip():
                raise ValueError(f"{label} requires a non-empty {field!r} string")
        for field in ("url", "cached", "file_location"):
            if field not in report:
                raise ValueError(f"{label} is missing required field {field!r}")

        filing_match = re.fullmatch(r"(?:A|Q[1-4])-(\d{2})-(\d{2})", report["report_name"])
        if not filing_match:
            raise ValueError(
                f"{label} has invalid report_name {report['report_name']!r}; "
                "expected A-YY-YY or Q1-YY-YY through Q4-YY-YY"
            )
        start_year, end_year = (int(part) for part in filing_match.groups())
        if end_year != (start_year + 1) % 100:
            raise ValueError(f"{label} report_name must span consecutive fiscal years")

        try:
            date.fromisoformat(report["published"])
        except ValueError as error:
            raise ValueError(
                f"{label} has invalid published date {report['published']!r}; use YYYY-MM-DD"
            ) from error
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["published"]):
            raise ValueError(f"{label} published date must use YYYY-MM-DD format")

        url = report.get("url")
        if url is not None and (not isinstance(url, str) or not url.strip()):
            raise ValueError(f"{label} url must be a non-empty string or null")
        file_location = report.get("file_location")
        if file_location is not None and not isinstance(file_location, str):
            raise ValueError(f"{label} file_location must be a string or null")
        if report["cached"] is not None and not isinstance(report["cached"], bool):
            raise ValueError(f"{label} cached must be true, false, or null")

        key = (report["company_name"], report["report_name"])
        if key in seen_reports:
            raise ValueError(f"{label} duplicates filing {key[1]!r} for {key[0]!r}")
        seen_reports.add(key)

    return data
