# Competitive Market Intelligence System

An integrated market research and financial intelligence dashboard. The Streamlit application combines multi-agent competitor research with this project's filing-based financial RAG, stock data, and company news.

The competitor research workflow is based on [AI-Competitor-Intelligence-Market-Analyst](https://github.com/pinaki-bit/AI-Competitor-Intelligence-Market-Analyst) (MIT licensed; see [LICENSE](LICENSE)). It includes CrewAI orchestration with a LangChain fallback, Gemini model selection and fallback, Tavily web research, SWOT and competitor analysis, market scoring, PDF/Markdown exports, and persistent analysis history.

## Run the dashboard

**For teammates cloning on another laptop: follow [Team setup](docs/TEAM_SETUP.md).** Install Python 3.12 and Git, clone this repository, then double-click `start.bat` (Windows) or run `bash start.sh` (macOS/Linux). The launcher creates the environment, installs dependencies and starts the app. The original developer's laptop is not required.

```sh
git clone https://github.com/shprite21/competitive-market-intelligence-dashboard.git
cd competitive-market-intelligence-dashboard
python run.py
```

Use `py -3.12 run.py` on Windows or `python3.12 run.py` on macOS/Linux if `python` is not available. API keys must be entered privately on the new laptop. Financial analysis also requires Ollama and downloaded models; the guide includes the complete checklist.

For a manual setup instead:

Install dependencies and configure keys:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` and `TAVILY_API_KEY` in `.env` (or enter them in **Overview → API keys and model settings**), then launch:

```powershell
streamlit run app.py
```

Open the **Financial Intelligence** page in Streamlit's page navigation to access filing analysis, statement comparisons, market prices, historical charts, and company news.

The two-position workspace toggle switches pages. **Deep Dive** off uses Quick research; on requests the Pro tier, with explicit fallback when a model is unavailable or its quota is exhausted. The agent activity log is collapsible. SWOT is part of the report. Financial, market and news results persist across reruns within the browser session.

## Included recent analyses

A fresh install includes the requested **Custom LangChain Orchestrator** analyses for [Broadcom](examples/analyses/broadcom.md), [NVIDIA](examples/analyses/nvidia.md), and [Reliance Industries](examples/analyses/reliance.md) in Recent analyses. These are archived AI-generated examples, clearly labelled in the dashboard. Their financial claims are not independently verified. Clearing history persists the empty history; the examples do not reappear on refresh.

## Verification and publication safety

Run `pytest tests/ -q`, `ruff check .`, and `ruff format --check .`. See [the validation record](docs/VALIDATION.md) for live checks and known limits. After staging a release, run `python scripts/check_secrets.py --staged` before committing. CI also scans tracked files.

API keys, encrypted credential files, `.env` files, local runtime files, notebook outputs and cached/generated reports are excluded from Git. Docker builds have separate secret exclusions in `.dockerignore`. Example environment files contain placeholders only. Local Windows keys use account-scoped DPAPI encryption; hosted keys remain in the session or deployment secrets.

## Financial report analysis

The filing workflow downloads and indexes configured company reports, then extracts profit and loss, cash flow, and balance sheet fields for a financial year. Enter a Pinecone key under **Financial Intelligence → API keys and model settings**, or configure `PINECONE_API_KEY` in the project `.env` or `Code/.env`. The local Windows dashboard can remember this key encrypted for your Windows account; hosted keys remain session-scoped. Ollama must be running with the configured chat model (`qwen3:8b` by default) and embedding model (`nomic-embed-text` by default). `Code/report_data.json` is the filing catalog; cached PDF locations resolve relative to the project.

```powershell
python Code/rag_agent.py --company Swiggy --years 25-26 24-25
```

Compare companies and years:

```powershell
python Code/rag_agent.py --compare-companies Swiggy "Reliance Industries" --years 25-26 24-25
python Code/rag_agent.py --compare-companies Swiggy "Reliance Industries" --years 25-26 --output Reports/benchmark_fy25-26.json
```

By default, benchmark results are written to `Reports/financial_benchmark.json`. Output preserves source extracts and page-level filing evidence, and provides a comparison grouped by annual reporting year, statement, and metric. It records per-company errors so a missing report for one company does not discard successful results for others. Amounts are normalized to base currency units when currency and scale are explicit; cross-currency comparisons are marked non-comparable unless the currencies match.

Only requested annual reports are indexed. A missing year is reported as an error. A number receives matched evidence status only if it occurs in its cited filing excerpts. Missing or unverified values are not normalized. Comparisons require at least two successful companies and matching currency, scale and annual period. Cash-flow labels can differ between companies; always inspect the page evidence before drawing financial conclusions.

Run the RAG tools on an existing index without re-indexing:

```powershell
python Code/rag_agent.py --company Swiggy --years 25-26 --test-tools
```

The existing stock and news command-line tools are in `Code/web_agent.py` and `Code/news_twitter_api.py`. Configure `TAVILY_API_KEY` in `Code/.env` for the CLI workflows.

## Agent dashboard features

- Researcher, competitor analyst, and report writer agents
- CrewAI workflow with a custom LangChain fallback
- Gemini model fallback and Quick / Deep Dive research settings
- Tavily multi-angle company and product research
- SWOT, competitor benchmarking, and six market score dimensions
- PDF and Markdown report export
- Local analysis history
- Per-run cost guard, API rate limiter, and retry handling
- Session-scoped API key and model override controls
- Optional hosted shared-password login (`APP_AUTH_PASSWORD`); Render and Docker deployments require it
- Deep Dive on/off toggle and a two-position workspace switch

## Project structure

- `app.py`, `agents.py`, `history_store.py`, `pdf_generator.py`, `styles.css`: competitor research dashboard and services
- `pages/2_Financial_Intelligence.py`: UI for the existing financial and market components
- `Code/`: filing selection, PDF indexing, Pinecone RAG, company news, and stock agent
- `Reports/`: cached filings and generated financial benchmarks
- `tests/`: upstream unit suite for agent helpers, PDF generation, history persistence, and app compatibility

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` and `TAVILY_API_KEY`. Set a long unique `APP_AUTH_PASSWORD` for hosted deployments; local development does not require it. Render and the Docker image fail closed if the password is missing. The same file includes optional `PINECONE_API_KEY`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_NUM_GPU` settings for financial RAG; `Code/.env` can hold the financial-specific settings instead.

## License and attribution

The competitor analyst dashboard files were copied from [pinaki-bit/AI-Competitor-Intelligence-Market-Analyst](https://github.com/pinaki-bit/AI-Competitor-Intelligence-Market-Analyst) under its MIT license, retained in `LICENSE`. The original project's notices apply to those files.
