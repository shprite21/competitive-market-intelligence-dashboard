#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if command -v python3.12 >/dev/null 2>&1; then
  exec python3.12 run.py "$@"
elif command -v python3.11 >/dev/null 2>&1; then
  exec python3.11 run.py "$@"
else
  echo "Install Python 3.12 or 3.11 from https://www.python.org/downloads/ first."
  exit 1
fi
