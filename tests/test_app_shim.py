"""Tests for the app.py compatibility shim — `_call_get_market_scores`.

Streamlit's hot-reload can leave a stale `agents.get_market_scores` in
memory. The shim inspects the signature at call time so the app works
with both the 2-arg old form and the 3-arg form with `prefer_llm`.
"""

import sys
from unittest.mock import patch


def _import_app():
    """Import app.py fresh, with streamlit warnings suppressed."""
    import warnings

    warnings.filterwarnings("ignore")
    if "app" in sys.modules:
        del sys.modules["app"]
    return __import__("app")


class TestCallGetMarketScores:
    def test_bare_import_does_not_enable_credential_storage(self):
        with (
            patch("credential_store.clear_credentials") as forget,
            patch("credential_store.save_credentials") as save,
        ):
            app = _import_app()
        forget.assert_not_called()
        save.assert_not_called()
        with patch.object(app, "get_script_run_ctx", return_value=None):
            assert app._local_credential_storage_available() is False

    def test_forget_credentials_requires_explicit_disabled_checkbox(self):
        app = _import_app()
        with (
            patch.object(app, "LOCAL_CREDENTIAL_STORAGE", True),
            patch.object(app.st, "session_state", {}),
            patch.object(app, "clear_credentials") as forget,
        ):
            app._forget_credentials_if_disabled()
            forget.assert_not_called()
            app.st.session_state["remember_api_keys"] = False
            app._forget_credentials_if_disabled()
            forget.assert_called_once_with(
                app.LOCAL_CREDENTIAL_PATH, keys=("gemini_api_key", "tavily_api_key")
            )

    def test_current_3arg_signature_with_demo_mode(self):
        app = _import_app()
        scores = app._call_get_market_scores(
            "# Report about Figma with growth and opportunity", "Figma", demo_mode=True
        )
        # All 6 keys present
        for k in (
            "market_opportunity",
            "competitive_pressure",
            "growth_trajectory",
            "innovation_score",
            "risk_level",
            "market_maturity",
        ):
            assert k in scores, f"missing key: {k}"
        # All values in range
        for v in scores.values():
            assert isinstance(v, int)
            assert 0 <= v <= 100

    def test_2arg_fallback_for_stale_module(self):
        """Simulate an old version of get_market_scores that only takes 2 args."""
        app = _import_app()

        # Patch the imported get_market_scores to a 2-arg function
        old_sig_calls = []

        def old_get_market_scores(report, topic):
            old_sig_calls.append((report, topic))
            return dict.fromkeys(
                (
                    "market_opportunity",
                    "competitive_pressure",
                    "growth_trajectory",
                    "innovation_score",
                    "risk_level",
                    "market_maturity",
                ),
                42,
            )

        with patch.object(app, "get_market_scores", old_get_market_scores):
            scores = app._call_get_market_scores("Some report", "Figma", demo_mode=False)

        assert old_sig_calls == [("Some report", "Figma")]
        assert all(v == 42 for v in scores.values())

    def test_handles_signature_inspection_failure(self):
        """If signature inspection itself fails, fall back to a 2-arg call."""
        app = _import_app()

        # Make inspect.signature raise so the shim takes the except branch
        two_arg_calls = []

        def old_get_market_scores(report, topic):
            two_arg_calls.append((report, topic))
            return dict.fromkeys(
                (
                    "market_opportunity",
                    "competitive_pressure",
                    "growth_trajectory",
                    "innovation_score",
                    "risk_level",
                    "market_maturity",
                ),
                50,
            )

        with (
            patch.object(app, "get_market_scores", old_get_market_scores),
            patch("app.inspect.signature", side_effect=ValueError("nope")),
        ):
            scores = app._call_get_market_scores("Some report", "Figma", demo_mode=True)

        assert two_arg_calls == [("Some report", "Figma")]
        assert all(v == 50 for v in scores.values())

    def test_prefer_llm_passes_through_to_dispatcher(self):
        """When the 3-arg form is in use, `demo_mode=True` must disable LLM scoring."""
        app = _import_app()

        # Track what `prefer_llm` value is being passed
        seen = {}

        def fake_get_market_scores(report, topic, prefer_llm=True):
            seen["prefer_llm"] = prefer_llm
            return dict.fromkeys(
                (
                    "market_opportunity",
                    "competitive_pressure",
                    "growth_trajectory",
                    "innovation_score",
                    "risk_level",
                    "market_maturity",
                ),
                60,
            )

        with patch.object(app, "get_market_scores", fake_get_market_scores):
            app._call_get_market_scores("Some report", "Figma", demo_mode=True)
            assert seen["prefer_llm"] is False, "Demo mode should disable LLM scoring"

            app._call_get_market_scores("Some report", "Figma", demo_mode=False)
            assert seen["prefer_llm"] is True, "Real mode should enable LLM scoring"
