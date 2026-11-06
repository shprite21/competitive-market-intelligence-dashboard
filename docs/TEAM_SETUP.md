# Run this project on a teammate's laptop

Your teammate's laptop runs the dashboard and local models. The original developer's laptop can stay switched off. Internet is required for live research, stock data, news, Pinecone and initial downloads.

## Before coming to college

1. Install [Git](https://git-scm.com/downloads) and [Python 3.12](https://www.python.org/downloads/) (Python 3.11 also works). On Windows, enable the Python launcher and “Add Python to PATH”.
2. Clone the public repository:

   ```sh
   git clone https://github.com/shprite21/competitive-market-intelligence-dashboard.git
   cd competitive-market-intelligence-dashboard
   ```

3. Set up and launch:

   **Windows:** double-click `start.bat`, or run this in PowerShell:

   ```powershell
   py -3.12 run.py
   ```

   **macOS / Linux:**

   ```sh
   bash start.sh
   ```

   The launcher creates `.venv`, installs the project dependencies, and starts Streamlit. Open **http://127.0.0.1:8501**. First setup needs internet and takes longer; subsequent starts reuse the environment. On Linux, if Python reports that `ensurepip` is unavailable, install your distribution's Python venv package first.

4. Confirm that **Recent analyses** includes NVIDIA, Broadcom and Reliance Industries. These archived reports and PDF/Markdown exports work before adding keys.

## Enable the live features

| Feature | What the teammate needs |
| --- | --- |
| Read the three archived analyses and export them | No API key or local model |
| Live company/competitor research and Deep Dive | Gemini + Tavily keys |
| Market prices and all historical ranges | Internet; no API key |
| Company news | Tavily key |
| Financial filing extraction and comparison | Pinecone key + local Ollama + both models below |

Enter **Gemini** and **Tavily** in **Overview → API keys and model settings**. Enter **Pinecone** in **Financial intelligence → API keys and model settings**. Keys must come from your own accounts or be shared privately by the project owner. The repository deliberately contains no working keys. A clone does not copy another Windows account's encrypted credentials.

For persistent local settings on any OS, optionally copy `.env.example` to `.env` and replace the placeholders locally. `.env` is ignored by Git. On local Windows, the dashboard also offers encrypted “Remember on this device” storage. On macOS/Linux, entered keys last for the browser session unless `.env` is configured.

## Set up financial analysis

Install [Ollama](https://ollama.com/download) and open it. If the service is not running, run `ollama serve` in another terminal. Then download the models:

```sh
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Alternatively, after installing and opening Ollama:

```sh
python run.py --setup-only --pull-models
```

On Windows without a `python` command, use `py -3.12` in place of `python`. On macOS/Linux use `python3.12` or `python3.11`.

Download these **before college**. [Qwen3 8B alone is about 5.2 GB](https://ollama.com/library/qwen3:8b), and the embedding model and Python environment need additional disk space. Model inference needs available memory; the default uses CPU to avoid GPU-memory problems and can take several minutes per statement. Close other heavy applications if memory is limited.

The app uses the Pinecone project belonging to the supplied key. With the same privately shared project key it can use existing indexes without the original laptop. A different Pinecone project will need to create and populate its own indexes, which takes longer and is subject to that project's quota. Supported filings are listed in `Code/report_data.json`; PDFs missing from the laptop are downloaded automatically. Start with **Swiggy, FY 24-25**, then compare **Swiggy, Reliance Industries** for **24-25**.

## Check readiness before leaving

```sh
python run.py --check
python run.py --check --financial
python run.py --setup-only --smoke-test
```

The first command checks Python dependencies and reports key/model readiness without showing secret values. A missing key does not prevent opening archived reports. The second returns a failure status if Pinecone or either local financial model is missing. These commands can read `.env`, environment variables and Windows device storage; they cannot see keys entered only in a browser session. Presence checks do not validate provider quotas or billing; run one live analysis and one filing extraction on that laptop before leaving.

The third command opens the app in Streamlit's test runtime, loads a bundled analysis and checks PDF exports for all three examples. It uses temporary history and never reads, changes or transmits your real keys.

## Tomorrow: start again

Run `start.bat` on Windows or `bash start.sh` on macOS/Linux. Keep the terminal open while presenting. Stop the dashboard with **Ctrl+C**. The URL is local to the teammate's laptop. A public GitHub repository is source hosting; it is not a hosted running dashboard.

If port 8501 is occupied, use `python run.py --port 8502` and open http://127.0.0.1:8502. To update the source later, stop the dashboard, run `git pull`, then start it again; changed dependency requirements trigger installation automatically.

The optional experimental CrewAI integration is not needed for the tested Custom LangChain workflow. Install it only if desired with `.venv`'s Python and `pip install -r requirements-crewai.txt`.
