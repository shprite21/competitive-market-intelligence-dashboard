"""Fail closed on credential files and provider tokens before publication.

Run after git add with: python scripts/check_secrets.py --staged
Only file names and rule names are printed; matched values are never logged.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "Google API credential": r"(?:AIza[\w-]{30,}|AQ\.[\w-]{30,})",
    "Tavily credential": r"tvly-(?:dev-)?[\w-]{25,}",
    "Pinecone credential": r"pcsk_[\w-]{25,}",
    "OpenAI credential": r"sk-(?:proj-|svcacct-)?[\w-]{30,}",
    "GitHub credential": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "Private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "AWS access identifier": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
}


def scan(files: dict[str, bytes], known_secrets=()) -> list[tuple[str, str]]:
    findings = []
    for name, content in files.items():
        path = Path(name)
        if (
            path.name == ".env"
            or (path.name.startswith(".env.") and path.name != ".env.example")
            or path.suffix in {".dpapi", ".pem", ".key"}
            or name.endswith("secrets.toml")
        ):
            findings.append((name, "credential file"))
        text = content.decode("utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                findings.append((name, label))
        for secret in known_secrets:
            if secret and len(secret) > 12 and secret.encode() in content:
                findings.append((name, "configured credential value"))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if args.staged
        else ["git", "ls-files", "-z"]
    )
    names = subprocess.check_output(command).decode().split("\0")
    files = {}
    for name in filter(None, names):
        files[name] = (
            subprocess.check_output(["git", "show", f":{name}"])
            if args.staged
            else Path(name).read_bytes()
        )
    known = [
        value for key, value in os.environ.items() if key.endswith(("API_KEY", "TOKEN", "PASSWORD"))
    ]
    findings = scan(files, known)
    for name, label in findings:
        print(f"BLOCKED: {name}: {label}")
    if findings:
        raise SystemExit(1)
    print(f"Secret scan passed: {len(files)} files; no credential files or matching tokens.")


if __name__ == "__main__":
    main()
