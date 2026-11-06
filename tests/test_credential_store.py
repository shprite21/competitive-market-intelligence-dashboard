"""Credential updates from either dashboard page must preserve the other keys."""

import os

import pytest

from credential_store import clear_credentials, load_credentials, save_credentials

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI storage")


def test_independent_page_saves_preserve_other_encrypted_keys(tmp_path):
    path = tmp_path / "credentials.dpapi"
    save_credentials(path, {"gemini_api_key": "test-gemini", "tavily_api_key": "test-tavily"})
    save_credentials(path, {"pinecone_api_key": "test-pinecone"})
    save_credentials(path, {"gemini_api_key": "updated-gemini", "tavily_api_key": "test-tavily"})
    assert load_credentials(path) == {
        "gemini_api_key": "updated-gemini",
        "tavily_api_key": "test-tavily",
        "pinecone_api_key": "test-pinecone",
    }
    assert b"test-pinecone" not in path.read_bytes()
    assert b"updated-gemini" not in path.read_bytes()


def test_forget_pinecone_preserves_overview_keys(tmp_path):
    path = tmp_path / "credentials.dpapi"
    save_credentials(path, {"gemini_api_key": "test-gemini", "pinecone_api_key": "test-pinecone"})
    clear_credentials(path, keys=("pinecone_api_key",))
    assert load_credentials(path) == {"gemini_api_key": "test-gemini"}


def test_forget_overview_preserves_pinecone(tmp_path):
    path = tmp_path / "credentials.dpapi"
    save_credentials(path, {"tavily_api_key": "test-tavily", "pinecone_api_key": "test-pinecone"})
    clear_credentials(path, keys=("gemini_api_key", "tavily_api_key"))
    assert load_credentials(path) == {"pinecone_api_key": "test-pinecone"}
    clear_credentials(path, keys=("pinecone_api_key",))
    assert not path.exists()
