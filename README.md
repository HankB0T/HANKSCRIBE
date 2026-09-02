# Hankscribe 2.0

Real-time meeting transcription + AI Q&A co-pilot for macOS. It listens to your
meeting audio, transcribes it live with Whisper, and lets you ask an AI
questions answered from **your own project documents** — plus optional screen
recording with on-device OCR so the AI knows what was on screen. Works with
whatever AI access you have: Claude (AWS Bedrock or Anthropic API), OpenAI,
Google Gemini (**free tier**), or a fully local Ollama model (free, offline).

Everything project-specific lives in `config.json`, so anyone can clone this and
point it at their own project folder.

> **Platform:** macOS only. It relies on macOS-specific pieces — the Vision
> framework for on-device OCR, `screencapture`, and CoreAudio Aggregate Devices.

---

## What it does

- **Live transcription** with speaker attribution (`[You]` = your mic,
  `[Them]` = meeting audio), using a locally-run Whisper server.
- **Real speaker names** (any meeting app — Zoom, Teams, Meet, Chime, Webex,
  Slack): when someone talks, the app reads the name the meeting app shows on
  screen and labels the line `[Alex]` instead of `[Them]`. Free (on-device
  text reading), zero storage (the screenshot is deleted immediately), and no
  guessing — ambiguous frames stay `[Them]`. Add colleagues to
  `speakers.roster` in config for best results.
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
6. **AI access — use whatever you have.** With `ai.provider: "auto"` the app
   picks the first configured option, in this order:

   | You have | Setup | Cost |
   |---|---|---|
   | An Anthropic API key | `pip install anthropic`, key from console.anthropic.com | paid per use |
   | An OpenAI API key | key from platform.openai.com | paid per use |
   | **Nothing — want free** | a free key from https://aistudio.google.com (Gemini free tier, rate-limited; a paid tier exists if you outgrow it) | **free** |
   | Any other AI service (Groq, Mistral, DeepSeek, OpenRouter, LM Studio, ...) | set `ai.custom_base_url`, `ai.custom_model`, and your key — works with anything that speaks the standard OpenAI-compatible API | varies (OpenRouter also lists free models) |
   | An AWS account | Bedrock access to Claude Opus 5 (default region `us-east-1`), standard AWS credential chain | paid per use |
   | No internet AI at all | just run Ollama (step 5) — the app falls back to it automatically | free, local |

   **Where the key goes** — either place works:
   - your shell: `export ANTHROPIC_API_KEY=...` (or `OPENAI_API_KEY`,
     `GEMINI_API_KEY`, `CUSTOM_API_KEY`) in `~/.zshrc`, or
   - `config.json` (it's gitignored, so keys stay private):
     `ai.anthropic_api_key` / `ai.openai_api_key` / `ai.gemini_api_key` /
     `ai.custom_api_key`.

   ⚠️ A ChatGPT / Claude / Gemini **chat subscription is not an API key** —
   API keys come from each provider's developer console (Gemini's is free).

   The same `config.json` works everywhere. To pin a provider, set
   `ai.provider` to `"bedrock"`, `"anthropic"`, `"openai"`, `"gemini"`, or
   `"custom"`. Claude providers get the full FAST/DEEP effort modes; on other
   providers the `m` toggle only changes answer length.

### Audio setup (one time)

Open **Audio MIDI Setup** → create an **Aggregate Device** that combines
**BlackHole 2ch** + your microphone. Route your meeting app's output to
BlackHole (or a Multi-Output Device) so the app hears both sides. The mic is
expected on channel index `mic_channel` (default `2`) — adjust in config if your
aggregate device orders channels differently.

#### Headphones — wired, Bluetooth, and swapping pairs

- **Hearing the meeting** through any headphones (wired, AirPods, other
  Bluetooth) always works — the meeting app's *output* goes to a Multi-Output
  Device (BlackHole + your headphones); the app never captures from your
  headphones' output side.
- **Sample rates are handled automatically.** The app reads the Aggregate
  Device's real rate at startup and resamples whatever it gets (44.1kHz,
  48kHz, 16kHz...), so swapping headphones won't garble transcription.
- **Which microphone matters.** If the mic in your Aggregate Device is your
  built-in Mac mic or a wired headset mic, any headphones work with zero
  changes. If you want to use a **Bluetooth headset's mic**, add that mic to
  the Aggregate Device and note two things: (1) in Audio MIDI Setup, enable
  **Drift Correction** on every sub-device except the first — Bluetooth mics
  drift against BlackHole's clock; (2) Bluetooth mics force the headset into
  low-quality HFP mode, which noticeably degrades what the *meeting* hears
  and can reduce transcription accuracy for your side. The built-in Mac mic
  usually transcribes better.
- **Per-pair Aggregate Devices.** An Aggregate Device pins specific
  sub-devices; a different headset = a different mic device. Either build one
  Aggregate per setup and list them all in `audio.preferred_devices` (first
  match wins), or keep the built-in mic in your one Aggregate so it works
  regardless of headphones.
- **Check `mic_channel` after changing the Aggregate.** BlackHole 2ch occupies
  channels 0-1, so a mic added after it sits on channel 2 (the default). If
  you reorder sub-devices, update `audio.mic_channel` — a startup warning
  appears if the device has fewer channels than expected.

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

Put the docs you want the AI to draw on in your project folder(s). Supported
file types: `.md .txt .vtt .docx .pdf .pptx .olm`. Re-run `build-context.py`
when they change (the app also auto-rebuilds when it detects changes on
startup).

**Multiple folders & multiple projects.** A project can span several folders
(`project_dirs` is a list), and `config.json` can define several named
projects under `"projects"` (see the template). Press **`c`** in the app to
switch projects, create one, edit folders, or change the AI provider — no
restart, no manual JSON editing. Each project keeps its own search index and
embeddings, so switching back is instant.

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
| `r` | Start/stop screen recording — app picker lists every running app; type to search, 1-9 to pick, Enter for first match, 0 for all screens, Esc to cancel |
| `m` | Choose the AI model/mode from a menu: Opus 5 deep / fast / quick, Sonnet 4.5, Haiku 4.5 — with speed and cost shown per option (`⌃⌥M` cycles them) |
| `c` | Settings menu: switch between projects, create a project, edit its folders, change AI provider — all saved to config.json, no restart needed |
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
- **"AWS Bedrock unavailable, using Ollama fallback"** — your AWS creds can't
  reach Claude Opus 5. Check region, model access, and (for Amazon internal
  SSO) refresh your credentials — or set `ANTHROPIC_API_KEY` to use the
  Anthropic API instead.
- **"Anthropic API unavailable, using Ollama fallback"** — check that
  `ANTHROPIC_API_KEY` is exported in the shell that launches the app and that
  `pip install anthropic` has run for the same interpreter.
- **Transcript only shows [You], never [Them] (common with headsets)** — the
  meeting app is sending the other side's audio straight to your headset,
  bypassing BlackHole. The app warns about this ~75s in. Fix: meeting app →
  Settings → Audio → Speaker = "Multi-Output Device" (or "Same as System"
  with System Sound Output = Multi-Output), and make sure BlackHole 2ch is
  ticked inside the Multi-Output Device in Audio MIDI Setup.
- **No cloud account at all?** Set `ai.provider: "ollama"` for fully local,
  free Q&A (`brew install ollama`, `ollama serve`, `ollama pull llama3.2:3b`).
  Screenshot descriptions are unavailable in this mode (no vision model);
  transcription is unaffected.
- **Volume keys don't work during meetings** — that's macOS, not Hankscribe:
  the volume keys are disabled whenever sound goes through a Multi-Output
  Device (the BlackHole + headphones combo). Workarounds: change volume
  inside the meeting app (Zoom/Teams have their own output slider); or open
  Audio MIDI Setup → select the Multi-Output Device → adjust your headphones'
  sub-device volume slider; or use a free menu-bar app like SoundSource /
  Background Music for full volume-key control.
- **Global hotkeys do nothing** — grant Accessibility permission and restart.
- **Semantic index never builds** — start Ollama and `ollama pull
  nomic-embed-text`; keyword search still works without it.
