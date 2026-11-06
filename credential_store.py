"""Encrypted per-Windows-user storage for local API credentials."""

from __future__ import annotations

import ctypes
import json
import os
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any

_CREDENTIAL_KEYS = ("gemini_api_key", "tavily_api_key", "pinecone_api_key")
_storage_lock = threading.RLock()


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _dpapi(data: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Encrypted local credential storage is only available on Windows.")

    source_buffer = ctypes.create_string_buffer(data)
    source_blob = _DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    if protect:
        function = crypt32.CryptProtectData
        function.restype = wintypes.BOOL
        function.argtypes = [
            ctypes.POINTER(_DataBlob),
            wintypes.LPCWSTR,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        succeeded = function(
            ctypes.byref(source_blob),
            "Competitive Market Intelligence API keys",
            None,
            None,
            None,
            0x01,  # CRYPTPROTECT_UI_FORBIDDEN
            ctypes.byref(result_blob),
        )
    else:
        function = crypt32.CryptUnprotectData
        function.restype = wintypes.BOOL
        function.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        succeeded = function(
            ctypes.byref(source_blob),
            None,
            None,
            None,
            None,
            0x01,  # CRYPTPROTECT_UI_FORBIDDEN
            ctypes.byref(result_blob),
        )

    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())

    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(result_blob.pbData, result_blob.cbData)
    finally:
        kernel32.LocalFree(result_blob.pbData)


def load_credentials(path: Path) -> dict[str, str]:
    """Decrypt previously saved keys for the current Windows user."""
    if not path.is_file():
        return {}
    payload = _dpapi(path.read_bytes(), protect=False)
    values: Any = json.loads(payload.decode("utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Saved credential data has an invalid format.")
    return {
        key: value
        for key, value in values.items()
        if key in _CREDENTIAL_KEYS and isinstance(value, str)
    }


def save_credentials(path: Path, values: dict[str, str]) -> None:
    """Update supplied keys without removing credentials belonging to another page."""
    with _storage_lock:
        merged = load_credentials(path)
        merged.update({key: values[key].strip() for key in _CREDENTIAL_KEYS if key in values})
        _write_credentials(path, merged)


def _write_credentials(path: Path, values: dict[str, str]) -> None:
    """Encrypt and atomically replace the complete credential payload."""
    payload = json.dumps(
        values,
        separators=(",", ":"),
    ).encode("utf-8")
    encrypted = _dpapi(payload, protect=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(encrypted)
    os.replace(temporary_path, path)


def clear_credentials(path: Path, *, keys: tuple[str, ...] | None = None) -> None:
    """Forget selected keys, preserving the other page's credentials."""
    with _storage_lock:
        if keys is None:
            path.unlink(missing_ok=True)
            return
        remaining = load_credentials(path)
        for key in keys:
            remaining.pop(key, None)
        if remaining:
            _write_credentials(path, remaining)
        else:
            path.unlink(missing_ok=True)
