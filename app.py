import contextlib
import importlib
import inspect
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

import agents as _agents
from auth import require_auth
from credential_store import clear_credentials, load_credentials, save_credentials
from history_store import append_history, clear_history, load_initial_history, save_history
from navigation import render_workspace_toggle
from pdf_generator import generate_pdf_from_markdown

_agents_source_mtime = Path(_agents.__file__).stat().st_mtime_ns
if getattr(_agents, "_app_source_mtime_ns", None) != _agents_source_mtime:
    _agents = importlib.reload(_agents)
    _agents._app_source_mtime_ns = Path(_agents.__file__).stat().st_mtime_ns

BudgetExceededError = _agents.BudgetExceededError
CostGuard = _agents.CostGuard
get_market_scores = _agents.get_market_scores
normalize_report_markdown = _agents.normalize_report_markdown
run_analysis = _agents.run_analysis

load_dotenv()
logger = logging.getLogger("streamlit_app")


def _call_get_market_scores(
    report: str,
    topic: str,
    demo_mode: bool,
    *,
    cost_guard: CostGuard | None = None,
    gemini_api_key: str | None = None,
    model_override: str | None = None,
) -> dict:
    """
    Invoke get_market_scores in a way that's compatible with both the
    current 3-arg signature (with `prefer_llm`) and the older 2-arg
    signature, so a stale module cache after a hot-reload doesn't break
    Demo Mode. Falls back to the heuristic on any signature mismatch.
    """
    try:
        sig = inspect.signature(get_market_scores)
    except (TypeError, ValueError) as e:
        logger.warning("Could not inspect get_market_scores signature: %s", e)
        return get_market_scores(report, topic)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    optional = {
        "prefer_llm": not demo_mode,
        "cost_guard": cost_guard,
        "gemini_api_key": gemini_api_key,
        "model_override": model_override,
    }
    kwargs = {name: value for name, value in optional.items() if name in params or accepts_kwargs}
    return get_market_scores(report, topic, **kwargs)


st.set_page_config(
    page_title="Competitive Market Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════
MAX_TOPIC_LENGTH = 100
MAX_HISTORY_ITEMS = 20
RESET_KEYS = (
    "report_markdown",
    "orchestrator_mode",
    "topic_analyzed",
    "market_scores",
    "agent_steps",
    "loaded_from_history",
    "demo_mode_active",
    "archived_example",
)
RESET_DEFAULTS = {
    "agent_steps": [],
    "market_scores": {},
    "demo_mode_active": False,
    "archived_example": False,
}


def _reset_session_state(keys=RESET_KEYS, defaults=RESET_DEFAULTS):
    """Reset a set of session-state keys to their default values."""
    for k in keys:
        st.session_state[k] = defaults.get(k, "")


# ═══════════════════════════════════════════════
#  LOAD EXTERNAL CSS
# ═══════════════════════════════════════════════
def _load_css() -> str:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
require_auth()

LOCAL_CREDENTIAL_PATH = Path(__file__).parent / ".streamlit" / "api_credentials.dpapi"


def _local_credential_storage_available() -> bool:
    """Only persist entered keys for a local Windows-only app instance."""
    # Bare imports (including compatibility tests) have no widget state.
    # Their checkbox defaults must never write or delete real credentials.
    if get_script_run_ctx(suppress_warning=True) is None:
        return False
    address = str(st.get_option("server.address") or "").strip().lower()
    hosted = bool(
        os.environ.get("RENDER")
        or os.environ.get("APP_AUTH_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}
        or os.environ.get("APP_ENV", "").strip().lower() in {"prod", "production"}
    )
    return os.name == "nt" and address in {"127.0.0.1", "localhost", "::1"} and not hosted


LOCAL_CREDENTIAL_STORAGE = _local_credential_storage_available()

for financial_key in ("session_pinecone_api_key", "remember_pinecone_api_key"):
    if financial_key in st.session_state:
        st.session_state[financial_key] = st.session_state[financial_key]


def _forget_credentials_if_disabled():
    """Remove remembered keys only after the user changes the checkbox."""
    if LOCAL_CREDENTIAL_STORAGE and not st.session_state.get("remember_api_keys", True):
        clear_credentials(LOCAL_CREDENTIAL_PATH, keys=("gemini_api_key", "tavily_api_key"))


# ═══════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════
defaults = {
    "report_markdown": "",
    "orchestrator_mode": "",
    "topic_analyzed": "",
    "market_scores": {},
    "agent_steps": [],
    "history": [],
    "loaded_from_history": False,
    "running": False,  # concurrent-run guard
    "demo_mode_active": False,  # tracks whether the current report came from demo mode
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "api_keys_initialized" not in st.session_state or any(
    key not in st.session_state for key in ("session_gemini_api_key", "session_tavily_api_key")
):
    saved_credentials = {}
    if LOCAL_CREDENTIAL_STORAGE:
        try:
            saved_credentials = load_credentials(LOCAL_CREDENTIAL_PATH)
        except Exception as credential_error:
            logger.warning("Could not load encrypted API keys: %s", credential_error)
    for state_key, credential_key, environment_key in (
        ("session_gemini_api_key", "gemini_api_key", "GEMINI_API_KEY"),
        ("session_tavily_api_key", "tavily_api_key", "TAVILY_API_KEY"),
    ):
        if state_key not in st.session_state:
            st.session_state[state_key] = ""
        if not str(st.session_state[state_key]).strip():
            st.session_state[state_key] = saved_credentials.get(credential_key) or os.environ.get(
                environment_key, ""
            )
    st.session_state.setdefault("remember_api_keys", True)
    st.session_state.api_keys_initialized = True

# Hydrate history from disk on first load so a refresh doesn't lose it.
if "history_hydrated" not in st.session_state:
    try:
        st.session_state.history = list(reversed(load_initial_history(max_items=MAX_HISTORY_ITEMS)))
    except Exception:
        st.session_state.history = []
    st.session_state.history_hydrated = True

# MAIN DASHBOARD
st.title("Competitor intelligence")
st.caption("Research a company or market, compare competitors, and generate a structured report.")
render_workspace_toggle("Overview")

with st.container(border=True):
    st.subheader("Analysis workspace")
    col_inp, col_btn = st.columns([5, 1.5], vertical_alignment="bottom")
    with col_inp:
        topic = st.text_input(
            "Company or market",
            placeholder="Enter a company or product market",
            max_chars=MAX_TOPIC_LENGTH,
        )
    with col_btn:
        run_btn = st.button(
            "Run analysis",
            use_container_width=True,
            disabled=st.session_state.running,
            type="primary",
        )

    setting_cols = st.columns([1, 2])
    with setting_cols[0]:
        deep_dive = st.toggle(
            "Deep Dive",
            value=False,
            help="Off runs Quick research. On uses a more capable model and deeper research.",
            key="deep_dive_enabled",
        )
        research_depth = "Deep Dive" if deep_dive else "Quick"
    with setting_cols[1]:
        gemini_present = bool(
            st.session_state.get("session_gemini_api_key", "").strip()
            or os.environ.get("GEMINI_API_KEY")
        )
        tavily_present = bool(
            st.session_state.get("session_tavily_api_key", "").strip()
            or os.environ.get("TAVILY_API_KEY")
        )
        if gemini_present and tavily_present:
            st.caption("Gemini and Tavily keys available")
        else:
            missing = [
                name
                for name, present in (("Gemini", gemini_present), ("Tavily", tavily_present))
                if not present
            ]
            st.caption(f"Missing: {', '.join(missing)}. Add keys below to run analysis.")

    with st.expander("API keys and model settings"):
        if LOCAL_CREDENTIAL_STORAGE:
            remember_keys = st.checkbox(
                "Remember API keys on this device",
                key="remember_api_keys",
                on_change=_forget_credentials_if_disabled,
                help="Keys are encrypted for your Windows account and saved only for this local app.",
            )
            st.caption(
                "Keys are encrypted for this Windows account when remembering is enabled and keys are entered."
            )
        else:
            remember_keys = False
            st.caption(
                "Keys stay in this browser session. For persistent hosted keys, configure deployment secrets."
            )
        key_cols = st.columns(2)
        with key_cols[0]:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                key="session_gemini_api_key",
                help="Google AI Studio — free tier available",
            )
        with key_cols[1]:
            tavily_key = st.text_input(
                "Tavily API Key",
                type="password",
                key="session_tavily_api_key",
                help="Tavily dashboard — free tier available",
            )
        st.caption(
            "Leave model names blank to use the auto-detected models. Unavailable overrides fall back automatically."
        )
        model_cols = st.columns(2)
        with model_cols[0]:
            fast_override = st.text_input(
                "Quick-tier model",
                value="",
                placeholder="e.g. gemini-3.5-flash-lite",
                key="session_gemini_model_fast",
            )
        with model_cols[1]:
            pro_override = st.text_input(
                "Deep Dive model",
                value="",
                placeholder="e.g. gemini-3.1-pro-preview",
                key="session_gemini_model_pro",
            )
        if LOCAL_CREDENTIAL_STORAGE:
            try:
                if remember_keys and (gemini_key.strip() or tavily_key.strip()):
                    save_credentials(
                        LOCAL_CREDENTIAL_PATH,
                        {
                            "gemini_api_key": gemini_key,
                            "tavily_api_key": tavily_key,
                        },
                    )
                    st.caption("API keys are saved encrypted for this Windows account.")
                elif not remember_keys:
                    st.caption("API keys will only be kept for this browser session.")
                else:
                    st.caption("Keys will be remembered after you enter them.")
            except Exception as credential_error:
                logger.warning("Could not save encrypted API keys: %s", credential_error)
                st.warning(
                    "Could not remember keys securely. They will remain in this session only."
                )

with st.expander(f"Recent analyses ({len(st.session_state.history)})"):
    if not st.session_state.history:
        st.caption("No analyses yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-6:])):
            history_cols = st.columns([4, 2, 1.2], vertical_alignment="center")
            with history_cols[0]:
                st.markdown(f"**{item['topic']}**")
            with history_cols[1]:
                st.caption(f"{item['timestamp']} · {item['mode']}")
            with history_cols[2]:
                if st.button("Load report", key=f"hist_{i}", use_container_width=True):
                    st.session_state.report_markdown = normalize_report_markdown(
                        item.get("markdown", "")
                    )
                    st.session_state.topic_analyzed = item["topic"]
                    st.session_state.orchestrator_mode = item["mode"]
                    st.session_state.market_scores = item["scores"]
                    st.session_state.agent_steps = []
                    st.session_state.loaded_from_history = True
                    st.session_state.demo_mode_active = item.get("mode") == "Simulation Mode"
                    st.session_state.archived_example = item.get("archived_example", False)
                    st.rerun()
        if st.button("Clear history", key="clear_history"):
            try:
                clear_history()
                save_history([])
            finally:
                st.session_state.history = []
                st.rerun()

if st.session_state.report_markdown and st.button("Clear report"):
    _reset_session_state()
    st.rerun()

# ═══════════════════════════════════════════════
#  EXECUTION
# ═══════════════════════════════════════════════
if run_btn:
    topic_clean = topic.strip() if topic else ""
    if not topic_clean:
        st.warning("Please enter a company name or product niche.")
    elif len(topic_clean) > MAX_TOPIC_LENGTH:
        st.warning(f"Topic must be {MAX_TOPIC_LENGTH} characters or less.")
    elif not (gemini_key.strip() or os.environ.get("GEMINI_API_KEY")):
        st.error("Gemini API key is missing. Add it in API keys and model settings.")
    elif not (tavily_key.strip() or os.environ.get("TAVILY_API_KEY")):
        st.error("Tavily API key is missing. Add it in API keys and model settings.")
    else:
        # Concurrent-run guard: flip the flag, release on every exit path.
        st.session_state.running = True
        st.session_state.agent_steps = []
        steps_placeholder = st.empty()

        def render_steps(steps):
            with steps_placeholder.container(), st.expander("Agent activity", expanded=True):
                for step in steps:
                    st.markdown(f"**{step['agent']} · {step['time']}**")
                    st.text(step["msg"])
                    st.divider()

        def status_cb(agent, msg):
            st.session_state.agent_steps.append(
                {"agent": agent, "msg": msg, "time": datetime.now().strftime("%H:%M:%S")}
            )
            render_steps(st.session_state.agent_steps)

        try:
            run_cost_guard = CostGuard()
            scoring_budget_exceeded = False
            effective_gemini_key = gemini_key.strip() or os.environ.get("GEMINI_API_KEY", "")
            effective_tavily_key = tavily_key.strip() or os.environ.get("TAVILY_API_KEY", "")
            tier_override = (
                pro_override if research_depth == "Deep Dive" else fast_override
            ).strip()
            report, mode = run_analysis(
                topic=topic_clean,
                depth=research_depth,
                status_callback=status_cb,
                cost_guard=run_cost_guard,
                gemini_api_key=effective_gemini_key,
                tavily_api_key=effective_tavily_key,
                model_override=tier_override or None,
            )

            try:
                scores = _call_get_market_scores(
                    report,
                    topic_clean,
                    demo_mode=False,
                    cost_guard=run_cost_guard,
                    gemini_api_key=effective_gemini_key,
                    model_override=fast_override.strip() or None,
                )
            except BudgetExceededError:
                # Keep the completed report available while making the budget
                # stop visible and using the non-billable heuristic scores.
                scores = _call_get_market_scores(report, topic_clean, demo_mode=True)
                scoring_budget_exceeded = True

            st.session_state.report_markdown = report
            st.session_state.orchestrator_mode = mode
            st.session_state.topic_analyzed = topic_clean
            st.session_state.market_scores = scores
            st.session_state.demo_mode_active = False
            st.session_state.archived_example = False

            new_entry = {
                "topic": topic_clean,
                "mode": mode,
                "markdown": report,
                "scores": scores,
                "timestamp": datetime.now().strftime("%d %b %H:%M"),
            }
            st.session_state.history.append(new_entry)
            if len(st.session_state.history) > MAX_HISTORY_ITEMS:
                st.session_state.history = st.session_state.history[-MAX_HISTORY_ITEMS:]
            # Persist to disk so a browser refresh doesn't lose the entry.
            try:
                append_history(new_entry, max_items=MAX_HISTORY_ITEMS)
            except Exception as persist_err:
                logger.warning("Could not persist history: %s", persist_err)

            steps_placeholder.empty()
            st.success(f"Report compiled via **{mode}** — topic: **{topic_clean}**")
            st.caption(f"Estimated Gemini spend this run: ${run_cost_guard.spent:.4f}")
            if scoring_budget_exceeded:
                st.warning(
                    "The shared run budget was reached during LLM scoring. "
                    "The report is saved with heuristic scores; no more model calls were made."
                )

        except Exception as e:
            st.error(f"Orchestration error: {str(e)}")
        finally:
            st.session_state.running = False

# ═══════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════
if st.session_state.report_markdown:
    # Also repair reports already in session/history that were stored as a
    # Python repr of Gemini's text content blocks.
    st.session_state.report_markdown = normalize_report_markdown(st.session_state.report_markdown)
    st.divider()

    st.subheader(st.session_state.topic_analyzed)
    word_count = len(st.session_state.report_markdown.split())
    st.caption(f"{word_count:,} words")
    if st.session_state.get("archived_example", False):
        st.info(
            "Archived AI-generated example. Its dates and financial claims have not been independently verified; run a new analysis for current research."
        )
    if st.session_state.demo_mode_active:
        st.warning("Demo Mode: this report contains illustrative data.")

    action_cols = st.columns(3)
    with action_cols[0]:
        pdf_fn = f"{st.session_state.topic_analyzed.lower().replace(' ', '_')}_report.pdf"
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                pdf_path = tmp.name
            generate_pdf_from_markdown(
                st.session_state.report_markdown, st.session_state.topic_analyzed, pdf_path
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "Download PDF", data=pdf_bytes, file_name=pdf_fn, mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF error: {e}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                with contextlib.suppress(OSError):
                    os.remove(pdf_path)

    with action_cols[1]:
        md_fn = f"{st.session_state.topic_analyzed.lower().replace(' ', '_')}_report.md"
        st.download_button(
            "Download Markdown",
            data=st.session_state.report_markdown.encode("utf-8"),
            file_name=md_fn,
            mime="text/markdown",
        )

    with action_cols[2]:
        if st.button("New report"):
            _reset_session_state()
            st.rerun()

    scores = st.session_state.market_scores
    if scores:
        st.subheader("Market scores")
        st.caption(
            "Scores are illustrative — derived from the report content. "
            "Use them as a quick signal, not ground truth."
        )

        score_items = [
            (
                "Market Opportunity",
                scores["market_opportunity"],
                "How strong is the market growth potential?",
            ),
            (
                "Competitive Pressure",
                scores["competitive_pressure"],
                "How intense is competition in this space?",
            ),
            (
                "Growth Trajectory",
                scores["growth_trajectory"],
                "Momentum and expansion rate of the company",
            ),
            (
                "Innovation Score",
                scores["innovation_score"],
                "Differentiation through product innovation",
            ),
            (
                "Risk Level",
                scores["risk_level"],
                "Overall strategic and market risk exposure",
            ),
            (
                "Market Maturity",
                scores["market_maturity"],
                "How mature/saturated is the target market?",
            ),
        ]

        sc_cols = st.columns(2)
        for i, (label, val, tip) in enumerate(score_items):
            with sc_cols[i % 2]:
                st.metric(label, val, help=tip)
                st.progress(val / 100)

    st.divider()

    if st.session_state.agent_steps and not st.session_state.loaded_from_history:
        with st.expander("Agent activity log", expanded=False):
            for step in st.session_state.agent_steps:
                st.markdown(f"**{step['agent']} · {step['time']}**")
                st.text(step["msg"])
                st.divider()

    st.markdown(st.session_state.report_markdown)

    st.session_state.loaded_from_history = False
