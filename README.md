# Hankscribe 2.0

Real-time meeting transcription + AI Q&A co-pilot for macOS. It listens to your
meeting audio, transcribes it live with Whisper, and lets you ask Claude
(Opus 5, via Amazon Bedrock) questions answered from **your own project
documents** — plus optional screen recording with on-device OCR so the AI knows
what was on screen.

Everything project-specific lives in `config.json`, so anyone can clone this and
point it at their own project folder.

> **Platform:** macOS only. It relies on macOS-specific pieces — the Vision
> framework for on-device OCR, `screencapture`, and CoreAudio Aggregate Devices.

---

## What it does

- **Live transcription** with speaker attribution (`[You]` = your mic,
  `[Them]` = meeting audio), using a locally-run Whisper server.
- **Answer questions** asked in the meeting, or your own typed questions, from a
  searchable index of your project docs (semantic retrieval via local
  embeddings, keyword fallback).
- **Screen recording** (press `r`): polls the screen, OCRs it on-device, and
  keeps a frame whenever the on-screen **text** changes — so typed content is
  captured, not just big visual changes. Diagrams get an AI visual description.
- **Post-meeting digest**: on quit, an exhaustive summary (decisions, action
  items, open questions) cross-referencing transcript + screen content, appended
  to the saved transcript.
- **Two AI modes** (toggle with `m`): FAST (Opus 5, medium effort) and DEEP
  (Opus 5, xhigh effort, most thorough).

---

## Prerequisites

You need these installed **before** first run:

1. **Homebrew Python 3.11+** — `brew install python@3.13`
2. **BlackHole 2ch** (virtual audio device) — `brew install blackhole-2ch`
3. **whisper.cpp** (provides `whisper-server` / `whisper-cli`) —
   `brew install whisper-cpp`
4. **A Whisper model file**, e.g.:
   ```bash
   mkdir -p ~/whisper-models
   curl -L -o ~/whisper-models/ggml-large-v3-turbo.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
   ```
   (Point `whisper.model` in your config at whatever file you download.)
5. **Ollama** (for semantic retrieval embeddings) — `brew install ollama`, then:
   ```bash
   ollama serve &            # if not already running
   ollama pull nomic-embed-text
   ```
   Semantic search is optional — without Ollama it falls back to keyword search.
6. **AWS credentials with Amazon Bedrock access** to Claude Opus 5 in your
   region (default `us-east-1`). The app uses the standard AWS credential chain
   (`~/.aws`, env vars, SSO, etc.). Verify with:
   ```bash
   aws bedrock list-foundation-models --region us-east-1 >/dev/null && echo OK
   ```

### Audio setup (one time)

Open **Audio MIDI Setup** → create an **Aggregate Device** that combines
**BlackHole 2ch** + your microphone. Route your meeting app's output to
BlackHole (or a Multi-Output Device) so the app hears both sides. The mic is
expected on channel index `mic_channel` (default `2`) — adjust in config if your
aggregate device orders channels differently.

---

## Setup

```bash
git clone <your-repo-url>
cd "Hankscribe 2.0"

# 1. Create your personal config from the templates
cp config.example.json config.json
cp transcription-corrections.example.json transcription-corrections.json

# 2. Edit config.json — at minimum set:
#    paths.project_dir        -> the folder of docs to answer questions from
#    paths.master_context     -> a key doc always in context (optional)
#    paths.transcript_dir     -> where meeting transcripts are saved
#    user.name / user.role    -> you
#    project.name / .description -> your project
#    whisper.model            -> path to your downloaded model
#    ai.bedrock_region        -> your Bedrock region

# 3. Install Python deps (or let START.command do it)
pip install -r requirements.txt

# 4. Build the searchable index from your project folder
python3 build-context.py
```

Put the docs you want the AI to draw on in `paths.project_dir`. Supported file
types: `.md .txt .vtt .docx .pdf .pptx .olm`. Re-run `build-context.py` when they
change (the app also auto-rebuilds when it detects changes on startup).

---

## Running

```bash
./START.command        # double-clickable; auto-installs core deps, builds index
# or
python3 hankscribe2.py
```

### Controls (terminal focused)

| Key | Action |
|-----|--------|
| `1` | Answer questions detected in the transcript |
| `2` | Ask your own question (type it in) |
| `3` | Auto-summary of the meeting so far |
| `4` | Screenshot current screen for context |
| `r` | Start/stop screen recording (prompts for which app to lock to) |
| `m` | Toggle AI mode: FAST (Opus 5 medium) ↔ DEEP (Opus 5 xhigh) |
| `s` | Show session stats (calls, cost, storage) |
| `q` | Quit — saves transcript + generates the post-meeting digest |

### Global hotkeys (work from Zoom/Teams/anywhere)

`⌃⌥1` answer · `⌃⌥3` summary · `⌃⌥4` screenshot · `⌃⌥R` record · `⌃⌥M` mode

Global hotkeys need **Accessibility** permission: System Settings → Privacy &
Security → Accessibility → add your terminal app, then restart Hankscribe.
Screen recording/screenshots need **Screen Recording** permission the first time.

---

## Privacy — what is and isn't in this repo

This repository is intentionally shipped **without** any personal or project
data. `.gitignore` excludes: your `config.json`, your corrections file, the
built context index and embeddings (which contain your documents' text), all
screenshots/recordings, and saved transcripts. Only code and `.example`
templates are tracked. **Do not force-add any of the gitignored files** — that
would publish your (or a client's) data.

---

## Configuration reference

All behavior is in `config.json` (see `config.example.json` for the full
template with comments). Highlights:

- **`ai.*_effort`** — `low` / `medium` / `high` / `xhigh` / `max`. Controls how
  hard Opus 5 thinks. FAST uses `medium`, DEEP uses `xhigh`, background/utility
  calls use `low`.
- **`ai.master_context_chars`**, **`ai.retrieval_*`** — how much project context
  is fed to the model and how wide semantic retrieval casts.
- **`recording.*`** — the content-driven recorder's cadence and thresholds. The
  keep/discard decision is made on OCR text change, not pixels, so typing is
  captured; `burst_ocr_dup_similarity` controls how aggressively near-identical
  frames are dropped.
- **`audio.sample_rate`** — a hint only; the app auto-detects your device's real
  rate at startup and resamples correctly regardless (so swapping headphones
  won't garble transcription).

---

## Troubleshooting

- **"No suitable audio input device found"** — create the Aggregate Device (see
  Audio setup) or set `audio.preferred_devices` to your device's name.
- **Transcription is garbled / sped up** — was fixed by runtime rate detection;
  if it recurs, check the startup line shows the right `Hz` for your device.
- **"Bedrock unavailable, using Ollama fallback"** — your AWS creds can't reach
  Claude Opus 5. Check region, model access, and (for Amazon internal SSO)
  refresh your credentials.
- **Global hotkeys do nothing** — grant Accessibility permission and restart.
- **Semantic index never builds** — start Ollama and `ollama pull
  nomic-embed-text`; keyword search still works without it.
