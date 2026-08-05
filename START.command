#!/bin/bash
cd "$(dirname "$0")"

echo "🎙️  Hankscribe 2.0 — Real-time Transcription + Claude Q&A"
echo "========================================================"
echo ""

# Pick the interpreter: prefer Homebrew Python 3.13 (has audio deps),
# fall back to any Homebrew python3, then system python3.
PYTHON=""
for candidate in \
    /opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
    /opt/homebrew/bin/python3.13 \
    /opt/homebrew/bin/python3 \
    python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
echo "Python: $PYTHON"

# First run: create personal config from the templates if missing
if [ ! -f "config.json" ] && [ -f "config.example.json" ]; then
    echo "⚠️  No config.json found — creating one from config.example.json."
    echo "    Edit config.json (project_dir, whisper.model, user/project) then re-run."
    cp config.example.json config.json
fi
if [ ! -f "transcription-corrections.json" ] && [ -f "transcription-corrections.example.json" ]; then
    cp transcription-corrections.example.json transcription-corrections.json
fi

# Install any missing dependencies into THAT interpreter.
# (import name -> pip package where they differ)
MISSING=""
check() {  # $1 = import name, $2 = pip package
    "$PYTHON" -c "import $1" 2>/dev/null || MISSING="$MISSING $2"
}
check sounddevice sounddevice
check numpy numpy
check boto3 boto3
check pynput pynput
check objc pyobjc-core
check AppKit pyobjc-framework-Cocoa
check Quartz pyobjc-framework-Quartz
# Vision.framework is loaded at runtime via objc.loadBundle (no top-level import).
# Detect the pyobjc Vision metadata package by its module name.
"$PYTHON" -c "import Vision" 2>/dev/null || MISSING="$MISSING pyobjc-framework-Vision"

if [ -n "$MISSING" ]; then
    echo "⚠️  Installing missing packages:$MISSING"
    "$PYTHON" -m pip install --break-system-packages $MISSING
    echo ""
fi

# Build context index on first run. Read the index filename from config.json
# (falls back to context-index.json) so this matches whatever the app expects.
INDEX_NAME=$("$PYTHON" -c "import json;print(json.load(open('config.json')).get('paths',{}).get('context_index','context-index.json'))" 2>/dev/null || echo context-index.json)
if [ ! -f "$INDEX_NAME" ]; then
    echo "⚠️  Building project context index for fast Q&A ($INDEX_NAME)..."
    "$PYTHON" build-context.py
    echo ""
fi

# Launch with the SAME interpreter the deps were installed into
exec "$PYTHON" hankscribe2.py
