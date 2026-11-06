from unittest.mock import Mock

import run


def test_install_is_repeatable_and_reacts_to_dependency_changes(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "requirements.txt").write_text("streamlit==1.66.1\n")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "VENV", root / ".venv")
    python = run.environment_python()
    python.parent.mkdir(parents=True)
    python.write_text("test interpreter placeholder")
    install = Mock()
    monkeypatch.setattr(run.subprocess, "run", install)
    assert run.install_environment() == python
    run.install_environment()
    assert install.call_count == 1
    (root / "requirements.txt").write_text("streamlit==1.66.1\npypdf\n")
    run.install_environment()
    assert install.call_count == 2


def test_check_uses_project_environment_instead_of_system_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "VENV", tmp_path / "venv")
    python = run.environment_python()
    python.parent.mkdir(parents=True)
    python.write_text("test interpreter placeholder")
    delegated = Mock(return_value=0)
    monkeypatch.setattr(run.subprocess, "call", delegated)
    assert run.main(["--check", "--financial"]) == 0
    assert delegated.call_args.args[0] == [
        str(python),
        str(run.ROOT / "run.py"),
        "--check",
        "--financial",
    ]


def test_financial_readiness_does_not_print_key_values(monkeypatch, capsys):
    keys = dict.fromkeys(
        ["GEMINI_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"], "test-private-value"
    )
    monkeypatch.setattr(run, "configured_keys", lambda: keys)
    monkeypatch.setattr(run, "ollama_status", lambda: {"qwen3:8b", "nomic-embed-text"})
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda _: object())
    assert run.check_setup(financial=True) == 0
    assert "test-private-value" not in capsys.readouterr().out
    keys.pop("PINECONE_API_KEY")
    assert run.check_setup(financial=True) == 1


def test_placeholder_keys_are_not_counted_as_configured():
    assert not run.usable_key("your_gemini_api_key_here")
    assert not run.usable_key("")
    assert run.usable_key("test-key")
