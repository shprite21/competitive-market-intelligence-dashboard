# Dashboard validation

Validation date: 6 November 2026 (Asia/Calcutta).

The [machine-readable live record](validation-results.json) records the three research runs, all six market ranges, news count and per-statement evidence counts. Local automated verification passed 148 tests. Lint and formatting checks passed.

## Automated interaction coverage

The suite combines service tests with Streamlit AppTest interactions. Tests isolate real credentials and history and do not call external research services.

| Feature | Verification |
| --- | --- |
| Overview and Financial Intelligence navigation | Shared required single-selection workspace switch; browser verification and page rendering |
| Quick / Deep Dive | Toggle mapping, model selection, retired-model aliases, live research runs |
| Missing keys / invalid topics | Explicit messages; no billable request on invalid input |
| Research, competitor benchmarking, SWOT | Live named-company research, structured report content, source links |
| Agent activity | Expanded during progress, collapsible after completion; no redundant SWOT tab |
| Six market scores | Integer range 0–100, structured parsing, deterministic heuristic fallback |
| Model fallback / cost guard | Retired models, overload, quota, shared budgets, no duplicate external work after kickoff |
| PDF and Markdown | Export controls rendered, real reports exported, PDFs parsed, unit coverage of text/tables |
| Recent analyses | Broadcom, NVIDIA and Reliance bundled; newest first, loading, reset, persistence and clear-history behavior |
| API key settings | Password inputs, separate session state, Windows encryption round-trip, selective forgetting, preservation between pages |
| Financial filing extraction | Live Pinecone retrieval + local Ollama; complete JSON, page evidence, requested year filtering |
| Company/year comparison | Cross-company live run; mixed currency, missing companies and unknown scale fail comparability |
| Market lookup | Company name and ticker, all six historical ranges; missing and duplicate closing prices removed |
| News | Live Tavily articles with titles and source URLs; results survive unrelated reruns |
| Hosted access | Fail closed when required password absent; incorrect password, sign-in and logout interactions |
| Publication | Exact staged-content secret scan, known local credential-value scan, ignore rules and Docker exclusions |

## Reproduction

```sh
pip install -r requirements.txt
pytest tests/ -q
ruff check .
ruff format --check .
python scripts/check_secrets.py
```

Live verification requires Gemini and Tavily credentials, Pinecone credentials, and Ollama with `qwen3:8b` and `nomic-embed-text`. Live results depend on provider availability. No credentials are included in this repository.

## Limits

- AI market reports are research drafts, not verified financial statements. Archived reports retain the original AI output and are labelled accordingly.
- Model availability and quotas vary by key. Deep Dive can use a fallback model, and the activity log names the model used. The current Pro default and conservative cost estimates were checked against [Google's model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) and [pricing](https://ai.google.dev/gemini-api/docs/pricing).
- The installed optional CrewAI version rejects LangChain LLM objects during setup. The tested automatic fallback runs the Custom LangChain Orchestrator before any CrewAI external work begins. CrewAI runtime success is not claimed.
- Provider billing can differ from the dashboard's estimate; the guard is an estimate, not a billing limit at the provider.
- Financial comparisons do not convert currencies or silently fill missing values. Unknown currency/scale, missing values and absent companies are marked non-comparable.
- PDF and notebook caches are local and are not published. Cached catalog paths are checked before use and missing PDFs are downloaded from the catalog URL.
- Local verification uses Windows/Python 3.12. Linux/Python 3.11 and 3.12 are covered by repository CI. Dependency deprecation warnings do not indicate failed assertions.
