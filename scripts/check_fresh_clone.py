"""Open the dashboard and export bundled reports without keys or local caches."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import history_store  # noqa: E402
from pdf_generator import generate_pdf_from_markdown  # noqa: E402


def main():
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(history_store, "_DEFAULT_PATH", Path(directory) / "history.json"),
        patch.dict(
            os.environ,
            {
                "MARKET_ANALYST_HISTORY_PATH": "",
                "APP_AUTH_REQUIRED": "false",
                "APP_AUTH_PASSWORD": "",
                "APP_ENV": "",
                "RENDER": "",
                "GEMINI_API_KEY": "",
                "TAVILY_API_KEY": "",
                "PINECONE_API_KEY": "",
            },
        ),
        patch("credential_store.load_credentials", return_value={}),
        patch("credential_store.save_credentials"),
        patch("credential_store.clear_credentials"),
    ):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception, [error.message for error in app.exception]
        assert {row["topic"] for row in app.session_state.history} == {
            "Broadcom",
            "NVIDIA",
            "Reliance Industries",
        }
        next(button for button in app.button if button.label == "Load report").click().run()
        assert not app.exception, [error.message for error in app.exception]
        assert app.session_state.topic_analyzed == "Reliance Industries"
        assert len(app.get("download_button")) == 2
        samples = json.loads(
            (ROOT / "examples" / "recent_analyses.json").read_text(encoding="utf-8")
        )["items"]
        for i, sample in enumerate(samples):
            output = Path(directory) / f"report-{i}.pdf"
            generate_pdf_from_markdown(sample["markdown"], sample["topic"], str(output))
            document = PdfReader(output)
            assert (
                document.pages
                and sum(len(page.extract_text() or "") for page in document.pages) > 1000
            )
        print(
            "Fresh-clone smoke check passed: dashboard opens, three analyses load, PDF and Markdown exports are available. No API keys or cached filings used."
        )


if __name__ == "__main__":
    main()
