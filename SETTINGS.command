#!/bin/bash
# Hankscribe 2.0 — open the Settings window (edit config.json before launch).
cd "$(dirname "$0")"

PYTHON=""
for candidate in \
    /opt/homebrew/bin/python3.13 \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

exec "$PYTHON" settings.py
