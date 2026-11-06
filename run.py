"""First-run setup and launcher for a fresh Windows, macOS or Linux clone."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
VENV = ROOT / ".venv"
MODULES = (
    "streamlit",
    "langchain_google_genai",
    "tavily",
    "fpdf",
    "langchain",
    "langchain_community",
    "langchain_text_splitters",
    "langgraph",
    "langchain_ollama",
    "pinecone",
    "langchain_pinecone",
    "pypdf",
    "yfinance",
    "pandas",
    "requests",
    "dotenv",
    "matplotlib",
)


def environment_python(folder=None):
    folder = VENV if folder is None else folder
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def requirements_digest():
    return hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest()


def install_environment():
    python = environment_python()
    if not python.is_file():
        import venv

        print("Creating the project's private Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    marker = VENV / ".requirements.sha256"
    digest = requirements_digest()
    if not marker.is_file() or marker.read_text().strip() != digest:
        print("Installing project dependencies. First setup needs internet access...", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            check=True,
            cwd=ROOT,
        )
        marker.write_text(digest, encoding="utf-8")
    return python


def configured_keys():
    values = {}
    if importlib.util.find_spec("dotenv"):
        from dotenv import dotenv_values

        for path in (ROOT / ".env", ROOT / "Code" / ".env"):
            values.update({key: value for key, value in dotenv_values(path).items() if value})
    values.update(os.environ)
    if os.name == "nt":
        from credential_store import load_credentials

        try:
            saved = load_credentials(ROOT / ".streamlit" / "api_credentials.dpapi")
        except (OSError, ValueError):
            saved = {}
        for key in ("gemini", "tavily", "pinecone"):
            if saved.get(f"{key}_api_key"):
                values[f"{key.upper()}_API_KEY"] = saved[f"{key}_api_key"]
    return values


def usable_key(value):
    value = str(value or "").strip()
    return bool(value and not value.lower().startswith(("your_", "replace_", "<")))


def ollama_status():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            return {
                model["name"].removesuffix(":latest")
                for model in json.load(response).get("models", [])
            }
    except (OSError, ValueError, urllib.error.URLError):
        return None


def check_setup(financial=False):
    print(
        "Checking local configuration; keys entered only in a browser session are not visible here."
    )
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    print("Python dependencies: " + ("ready" if not missing else "missing " + ", ".join(missing)))
    keys = configured_keys()
    for name in ("GEMINI_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"):
        print(
            f"{name}: {'configured (not remotely validated)' if usable_key(keys.get(name)) else 'not configured; enter it in the dashboard'}"
        )
    models = ollama_status()
    needed = {
        keys.get("OLLAMA_CHAT_MODEL", "qwen3:8b").removesuffix(":latest"),
        keys.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").removesuffix(":latest"),
    }
    print(
        "Ollama: "
        + ("not running; open Ollama or run ollama serve" if models is None else "running")
    )
    absent = needed - (models or set())
    if absent:
        print("Financial models needed: " + ", ".join(sorted(absent)))
    if financial:
        return 1 if missing or absent or not usable_key(keys.get("PINECONE_API_KEY")) else 0
    return 1 if missing else 0


def pull_models():
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError(
            "Install Ollama from https://ollama.com/download, open it, then rerun with --pull-models."
        )
    if ollama_status() is None:
        raise RuntimeError("Open Ollama or run 'ollama serve' in another terminal, then try again.")
    values = configured_keys()
    for model in (
        values.get("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        values.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    ):
        subprocess.run([executable, "pull", model], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Install dependencies without starting the dashboard",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Show readiness without installing or starting anything",
    )
    parser.add_argument(
        "--financial",
        action="store_true",
        help="With --check, require Pinecone and both local models",
    )
    parser.add_argument(
        "--pull-models",
        action="store_true",
        help="Download financial models using installed Ollama (several GB)",
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Check dashboard startup and bundled exports without API keys",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use Python 3.12 (recommended) or Python 3.11 for this project.")
    if args.check:
        existing = environment_python()
        if existing.is_file() and Path(sys.prefix).resolve() != VENV.resolve():
            return subprocess.call([str(existing), str(ROOT / "run.py"), *arguments], cwd=ROOT)
        return check_setup(args.financial)
    python = install_environment()
    if Path(sys.prefix).resolve() != VENV.resolve():
        return subprocess.call([str(python), str(ROOT / "run.py"), *arguments], cwd=ROOT)
    if args.pull_models:
        pull_models()
    if args.smoke_test:
        subprocess.run(
            [str(python), str(ROOT / "scripts" / "check_fresh_clone.py")], check=True, cwd=ROOT
        )
    if args.setup_only:
        print("Setup complete. Start with: python run.py")
        return 0
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print("Enter API keys in the dashboard. Saved example reports work without keys.", flush=True)
    return subprocess.call(
        [
            str(python),
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.address=127.0.0.1",
            f"--server.port={args.port}",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
