#!/usr/bin/env python3
"""
Hankscribe 2.0 — Settings window.

A standalone, zero-dependency GUI for config.json that runs BEFORE Hankscribe:
it serves one local web page (stdlib http.server) and opens it in your browser.
Nothing leaves your machine — the server binds to 127.0.0.1 only and shuts
down when you close it or launch Hankscribe.

Run it:
    - double-click SETTINGS.command, or
    - python3 settings.py

Buttons on the page:
    Save                 — write config.json, keep editing
    Save & Launch        — write config.json, quit this window, start Hankscribe
    Close                — quit without launching (unsaved edits are lost)
"""

import json
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
EXAMPLE_PATH = os.path.join(BASE_DIR, "config.example.json")

# What happens after the server stops: "launch" starts Hankscribe.
_exit_action = {"action": "close"}
_shutdown = threading.Event()


def load_config():
    for path in (CONFIG_PATH, EXAMPLE_PATH):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)} unreadable ({e})")
    return {}


def save_config(new_values):
    """Merge the page's values into the existing config.json, preserving any
    keys the form doesn't manage."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    def deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v

    deep_merge(raw, new_values)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    return raw


# ── Page ───────────────────────────────────────────────────────────────────────
# One self-contained page. Sections mirror config.json; the JS collects inputs
# into the same nested structure and POSTs it back as JSON.

PAGE_CSS = """
:root {
  --bg: #101418; --panel: #171d24; --panel2: #1d242d; --border: #2a333f;
  --text: #d7dee8; --dim: #8a97a6; --cyan: #4fd2e0; --green: #58d68d;
  --yellow: #f5c76a; --red: #ef6a6a; --magenta: #c88ae0; --blue: #6aa6ef;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); margin: 0;
  font: 14px/1.5 -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif; }
.wrap { max-width: 880px; margin: 0 auto; padding: 28px 20px 120px; }
h1 { font-size: 21px; letter-spacing: .04em; color: var(--cyan); margin: 0; }
h1 .sub { color: var(--dim); font-size: 13px; font-weight: 400; margin-left: 10px; }
.section { background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; margin-top: 18px; overflow: hidden; }
.section > summary { cursor: pointer; padding: 13px 18px; font-weight: 600;
  font-size: 15px; color: var(--cyan); list-style: none; display: flex;
  align-items: center; gap: 10px; user-select: none; }
.section > summary::-webkit-details-marker { display: none; }
.section > summary .chev { transition: transform .15s; color: var(--dim); font-size: 11px; }
.section[open] > summary .chev { transform: rotate(90deg); }
.section > summary .hint { color: var(--dim); font-weight: 400; font-size: 12px; }
.body { padding: 4px 18px 16px; border-top: 1px solid var(--border); }
.row { display: grid; grid-template-columns: 230px 1fr; gap: 14px;
  align-items: start; padding: 9px 0; border-bottom: 1px solid #1c232c; }
.row:last-child { border-bottom: none; }
.row label { padding-top: 6px; }
.row label .k { display: block; font-weight: 550; }
.row label .d { display: block; color: var(--dim); font-size: 12px; margin-top: 1px; }
input[type=text], input[type=number], input[type=password], select, textarea {
  width: 100%; background: var(--panel2); color: var(--text);
  border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
  font: 13px/1.45 "SF Mono", ui-monospace, Menlo, monospace; }
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--cyan); }
textarea { resize: vertical; min-height: 74px; }
.check { display: flex; align-items: center; gap: 10px; padding: 8px 0;
  border-bottom: 1px solid #1c232c; }
.check:last-child { border-bottom: none; }
.check input { width: 16px; height: 16px; accent-color: var(--green); }
.check .d { color: var(--dim); font-size: 12px; }
.badge { font-size: 11px; padding: 1px 8px; border-radius: 99px;
  border: 1px solid var(--magenta); color: var(--magenta); }
.bar { position: fixed; left: 0; right: 0; bottom: 0; background: #141a21ee;
  border-top: 1px solid var(--border); backdrop-filter: blur(6px); }
.bar .in { max-width: 880px; margin: 0 auto; padding: 13px 20px;
  display: flex; gap: 10px; align-items: center; }
button { border: 0; border-radius: 8px; padding: 9px 18px; font-weight: 600;
  font-size: 14px; cursor: pointer; }
#save { background: var(--panel2); color: var(--text); border: 1px solid var(--border); }
#save:hover { border-color: var(--green); color: var(--green); }
#launch { background: var(--green); color: #0c1116; }
#launch:hover { filter: brightness(1.1); }
#close { background: none; color: var(--dim); }
#close:hover { color: var(--red); }
#status { margin-left: auto; font-size: 13px; color: var(--dim); }
#status.ok { color: var(--green); } #status.err { color: var(--red); }
.note { color: var(--dim); font-size: 12px; margin: 14px 2px 0; }
"""

PAGE_JS = """
function collect() {
  const cfg = {};
  const set = (path, value) => {
    const keys = path.split('.');
    let o = cfg;
    for (let i = 0; i < keys.length - 1; i++) o = (o[keys[i]] = o[keys[i]] || {});
    o[keys[keys.length - 1]] = value;
  };
  document.querySelectorAll('[data-path]').forEach(el => {
    const t = el.dataset.type;
    let v;
    if (t === 'bool') v = el.checked;
    else if (t === 'number') { v = parseFloat(el.value); if (isNaN(v)) return; }
    else if (t === 'int') { v = parseInt(el.value, 10); if (isNaN(v)) return; }
    else if (t === 'list') v = el.value.split(/\\n|,/).map(s => s.trim()).filter(Boolean);
    else v = el.value.trim();
    if (t === 'secret' && v === '') return;   // never blank out a stored key
    set(el.dataset.path, v);
  });
  return cfg;
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'},
                               body: JSON.stringify(body || {}) });
  return r.ok;
}
function flash(msg, cls) {
  const s = document.getElementById('status');
  s.textContent = msg; s.className = cls || '';
}
document.getElementById('save').onclick = async () => {
  flash('Saving…');
  const ok = await post('/save', collect());
  flash(ok ? '✓ Saved to config.json' : '✗ Save failed', ok ? 'ok' : 'err');
  // Re-render so project fields track a newly selected active project
  if (ok) setTimeout(() => location.reload(), 600);
};
document.getElementById('launch').onclick = async () => {
  flash('Saving…');
  if (!await post('/save', collect())) { flash('✗ Save failed', 'err'); return; }
  flash('✓ Saved — launching Hankscribe…', 'ok');
  await post('/launch');
  document.body.innerHTML = '<div class="wrap"><h1>✓ Hankscribe is starting in Terminal.</h1>' +
    '<p class="note">You can close this tab.</p></div>';
};
document.getElementById('close').onclick = async () => {
  await post('/quit');
  document.body.innerHTML = '<div class="wrap"><h1>Settings closed.</h1>' +
    '<p class="note">You can close this tab.</p></div>';
};
"""


def esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ── Audio setup detection ──────────────────────────────────────────────────────
# The one part of Hankscribe that a fresh Mac always gets wrong. The page
# scans the ACTUAL audio devices and shows exactly which setup step is missing.

def _scan_audio_devices():
    """[{name, in, out, rate}] for every audio device. Uses sounddevice when
    installed; falls back to system_profiler (always present on macOS) so the
    check works even before START.command has installed dependencies."""
    try:
        import sounddevice as sd
        return [{"name": d["name"], "in": d["max_input_channels"],
                 "out": d["max_output_channels"],
                 "rate": int(d.get("default_samplerate") or 0)}
                for d in sd.query_devices()]
    except Exception:
        pass
    try:
        r = subprocess.run(["system_profiler", "-json", "SPAudioDataType"],
                           capture_output=True, timeout=15)
        items = (json.loads(r.stdout).get("SPAudioDataType") or [{}])[0].get("_items", [])
        return [{"name": it.get("_name", ""),
                 "in": int(it.get("coreaudio_device_input") or 0),
                 "out": int(it.get("coreaudio_device_output") or 0),
                 "rate": int(it.get("coreaudio_device_srate") or 0)}
                for it in items]
    except Exception:
        return None


def _audio_checks(cfg, devices):
    """[(ok, what, how-to-fix)] — every automated check we can run without
    touching the microphone."""
    preferred = _get(cfg, "audio.preferred_devices", []) or []
    mic_ch = _get(cfg, "audio.mic_channel", 2)
    names = [(d["name"] or "").lower() for d in devices]

    def has(sub):
        return any(sub in n for n in names)

    checks = [(has("blackhole"),
               "BlackHole 2ch installed",
               "Run: brew install blackhole-2ch  (then log out/in or reboot)")]

    # Same selection order as the app's find_audio_device: preference wins
    agg = next((d for p in preferred for d in devices
                if p.lower() in (d["name"] or "").lower() and d["in"] > 0), None)
    checks.append((agg is not None,
                   f"Aggregate input device found (looked for: {', '.join(preferred)})",
                   "Audio MIDI Setup → + → Create Aggregate Device → tick "
                   "BlackHole 2ch FIRST, then your microphone"))
    if agg:
        checks.append((agg["in"] > max(2, int(mic_ch) if isinstance(mic_ch, (int, float)) else 2) - 1
                       and agg["in"] > 2,
                       f"Aggregate has a mic channel ({agg['name']}: {agg['in']} input channels, "
                       f"mic expected on channel {mic_ch})",
                       "The Aggregate must contain BOTH BlackHole (channels 0-1) AND a "
                       "microphone (channel 2). Add your mic in Audio MIDI Setup, or fix "
                       "'Mic channel' below to match the channel order."))
    checks.append((has("multi-output"),
                   "Multi-Output Device exists (so you hear the meeting AND BlackHole gets a copy)",
                   "Audio MIDI Setup → + → Create Multi-Output Device → tick "
                   "your headphones/speakers AND BlackHole 2ch"))
    return checks


MANUAL_AUDIO_STEPS = [
    ("System sound output", "System Settings → Sound → Output → Multi-Output Device"),
    ("Meeting app speaker — THE #1 TRAP", "Teams/Zoom → Settings → Audio/Devices → "
     "Speaker = 'Multi-Output Device' (or 'Same as System'). If the speaker is set to "
     "your headset directly, the meeting bypasses BlackHole and Hankscribe hears "
     "nothing from the other side — only [You] lines appear."),
    ("Meeting app microphone", "Keep it on your real mic (headset or built-in) — "
     "NEVER BlackHole."),
    ("Drift correction (Aggregate Device)", "Audio MIDI Setup → your Aggregate Device → "
     "tick 'Drift Correction' on the microphone row."),
    ("Verify live", "Start Hankscribe, press n during a meeting — it measures each "
     "audio channel and tells you if routing or the mic channel is wrong."),
]


def render_audio_setup(cfg):
    devices = _scan_audio_devices()
    rows = ""
    if devices is None:
        rows = ('<p class="d" style="color:var(--yellow)">⚠ Could not scan audio devices. '
                'Run START.command once (installs dependencies), then hit Re-scan.</p>')
        checks_html = ""
    else:
        checks_html = ""
        for ok, what, fix in _audio_checks(cfg, devices):
            icon = '<span style="color:var(--green)">✓</span>' if ok else \
                   '<span style="color:var(--red)">✗</span>'
            fix_html = "" if ok else f'<div class="d" style="color:var(--yellow)">→ {esc(fix)}</div>'
            checks_html += f'<div class="check" style="display:block"><div>{icon} {esc(what)}</div>{fix_html}</div>'
        rows = "".join(
            f"<tr><td>{esc(d['name'])}</td><td>{d['in']}</td><td>{d['out']}</td>"
            f"<td>{d['rate'] or ''}</td></tr>" for d in devices)
        rows = ('<table style="width:100%;border-collapse:collapse;font:12px ui-monospace,Menlo,monospace">'
                '<tr style="color:var(--dim);text-align:left"><th>Device</th><th>In ch</th>'
                '<th>Out ch</th><th>Rate</th></tr>' + rows + "</table>")

    steps_html = "".join(
        f'<div class="check" style="display:block"><div><b>{i}. {esc(t)}</b></div>'
        f'<div class="d">{esc(s)}</div></div>'
        for i, (t, s) in enumerate(MANUAL_AUDIO_STEPS, 1))

    body = (checks_html
            + '<div style="margin:10px 0 4px"><button type="button" onclick="location.reload()" '
              'style="background:var(--panel2);color:var(--text);border:1px solid var(--border)">'
              'Re-scan devices</button></div>'
            + rows
            + '<h3 style="color:var(--cyan);font-size:13px;margin:16px 0 4px">'
              'One-time manual steps (macOS/meeting-app settings — no app can do these for you; '
              'full guide: AUDIO-SETUP.md)</h3>'
            + steps_html)
    n_bad = 0 if devices is None else sum(1 for ok, *_ in _audio_checks(cfg, devices) if not ok)
    hint = "all automated checks pass ✓" if devices is not None and n_bad == 0 else \
           (f"{n_bad} check(s) failing" if devices is not None else "scan unavailable")
    return section("Audio setup — new-Mac checklist", hint, body, open_=(n_bad > 0 or devices is None))


def _get(cfg, path, default=""):
    o = cfg
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return default
        o = o[k]
    return o


def field(cfg, path, label, desc, ftype="text", options=None, placeholder=""):
    v = _get(cfg, path, "")
    if ftype == "bool":
        return (f'<div class="check"><input type="checkbox" data-path="{path}" '
                f'data-type="bool" {"checked" if v else ""} id="f_{path}">'
                f'<label for="f_{path}"><span class="k">{esc(label)}</span> '
                f'<span class="d">{esc(desc)}</span></label></div>')
    if ftype == "select":
        opts = "".join(
            f'<option value="{esc(o)}" {"selected" if o == v else ""}>{esc(o)}'
            f'{" — current" if o == v else ""}</option>' for o in options)
        control = f'<select data-path="{path}" data-type="text">{opts}</select>'
    elif ftype == "list":
        val = "\n".join(v) if isinstance(v, list) else esc(v)
        control = (f'<textarea data-path="{path}" data-type="list" '
                   f'placeholder="{esc(placeholder)}">{esc(val)}</textarea>')
    elif ftype in ("number", "int"):
        control = (f'<input type="number" step="any" data-path="{path}" '
                   f'data-type="{ftype}" value="{esc(v)}">')
    elif ftype == "secret":
        control = (f'<input type="password" data-path="{path}" data-type="secret" '
                   f'value="" placeholder="{"(saved — type to replace)" if v else "(not set)"}">')
    else:
        control = (f'<input type="text" data-path="{path}" data-type="text" '
                   f'value="{esc(v)}" placeholder="{esc(placeholder)}">')
    return (f'<div class="row"><label><span class="k">{esc(label)}</span>'
            f'<span class="d">{esc(desc)}</span></label>{control}</div>')


def section(title, hint, body, open_=True):
    return (f'<details class="section" {"open" if open_ else ""}>'
            f'<summary><span class="chev">▶</span>{esc(title)}'
            f'<span class="hint">{esc(hint)}</span></summary>'
            f'<div class="body">{body}</div></details>')


def render_page():
    cfg = load_config()
    active = _get(cfg, "projects.active", "")
    projects = cfg.get("projects", {}) or {}
    project_names = [k for k in projects if k != "active" and isinstance(projects[k], dict)]
    ap = f"projects.{active}" if active in project_names else "paths"

    s_project = section(
        "Project", f"active: {active or '(legacy paths)'}",
        field(cfg, "projects.active", "Active project", "Which project profile Hankscribe uses",
              "select", options=project_names or [active or ""])
        + field(cfg, f"{ap}.project_dirs", "Project folder(s)",
                "One path per line — all are indexed for Q&A", "list", placeholder="~/Desktop/MyProject")
        + field(cfg, f"{ap}.master_context", "Master context file",
                "Markdown file always included in the AI's context")
        + field(cfg, f"{ap}.transcript_dir", "Transcript folder", "Where meeting transcripts are saved")
        + field(cfg, f"{ap}.description", "Description", "Used in AI prompts: 'the … engagement'"))

    s_speakers = section(
        "Speakers — who's in your meetings", "powers the [Them] → real-name feature",
        field(cfg, "speakers.enabled", "Name speakers", "Resolve [Them] to real names from on-screen text", "bool")
        + field(cfg, "speakers.roster", "Roster",
                "One name per line. ONLY these names can ever appear as speaker "
                "labels — keep it complete and current for your meetings.", "list",
                placeholder="Alice\nBob Smith\n…")
        + field(cfg, "user.name", "Your name", "Lines from your mic are labeled [You]")
        + field(cfg, "user.role", "Your role", "Used in AI prompts when drafting your replies")
        + field(cfg, "user.name_variants", "Your name as Whisper mishears it",
                "One per line — prevents you being named as a [Them] speaker", "list"))

    s_ai = section(
        "AI provider", _get(cfg, "ai.provider", "auto"),
        field(cfg, "ai.provider", "Provider",
              "auto picks the first configured: anthropic → openai → gemini → custom → bedrock",
              "select", options=["auto", "bedrock", "anthropic", "openai", "gemini", "custom", "ollama"])
        + field(cfg, "ai.bedrock_region", "Bedrock region", "For provider=bedrock (AWS credentials)")
        + field(cfg, "ai.anthropic_api_key", "Anthropic API key", "Optional — or export ANTHROPIC_API_KEY", "secret")
        + field(cfg, "ai.openai_api_key", "OpenAI API key", "Optional — or export OPENAI_API_KEY", "secret")
        + field(cfg, "ai.gemini_api_key", "Gemini API key", "Free tier: aistudio.google.com", "secret")
        + field(cfg, "ai.custom_base_url", "Custom endpoint URL", "Any OpenAI-compatible service (Groq, LM Studio, …)")
        + field(cfg, "ai.custom_model", "Custom model", "Model name at the custom endpoint")
        + field(cfg, "ai.ollama_model", "Ollama model", "Local fallback / provider=ollama")
        + field(cfg, "ai.embed_model", "Embedding model", "Ollama model for semantic document search"))

    s_audio_setup = render_audio_setup(cfg)

    s_audio = section(
        "Audio", "BlackHole + Aggregate Device",
        field(cfg, "audio.preferred_devices", "Preferred input devices",
              "Tried in order; first match wins. Must name your Aggregate "
              "Device (see the checklist above).", "list")
        + field(cfg, "audio.mic_channel", "Mic channel",
                "Which channel of the Aggregate Device is YOUR mic (BlackHole is "
                "usually 0-1, mic 2). Wrong value swaps [You]/[Them] — verify "
                "with the n key in Hankscribe.", "int")
        + field(cfg, "audio.sample_rate", "Expected sample rate (Hz)",
                "Auto-adapted at startup to the device's real rate — this is "
                "just the initial guess.", "int")
        + field(cfg, "audio.silence_threshold", "Silence threshold",
                "Lower = more sensitive speech detection", "number")
        + field(cfg, "audio.vad_pause_sec", "Pause before flush (s)",
                "Silence gap that ends a speech segment", "number")
        + field(cfg, "audio.vad_max_segment_sec", "Max segment length (s)",
                "Longest single transcription chunk", "number"),
        open_=False)

    s_whisper = section(
        "Whisper (transcription)", os.path.basename(_get(cfg, "whisper.model", "")),
        field(cfg, "whisper.model", "Model file", "Path to a whisper.cpp .bin model")
        + field(cfg, "whisper.port", "Server port", "", "int")
        + field(cfg, "whisper.threads", "Threads", "", "int"),
        open_=False)

    s_features = section(
        "Features", "",
        field(cfg, "features.speaker_attribution", "Speaker attribution",
              "[You]/[Them] from channel energy", "bool")
        + field(cfg, "features.semantic_retrieval", "Semantic document search",
                "Ollama embeddings over your project docs", "bool")
        + field(cfg, "features.answer_suggestions", "Proactive reply suggestions",
                "Suggest an answer when someone asks you a question by name", "bool")
        + field(cfg, "features.post_meeting_digest", "Post-meeting digest",
                "Exhaustive summary appended to the transcript on quit", "bool")
        + field(cfg, "features.burst_recording", "Screen recording (r key)",
                "Content-driven screen capture with on-device OCR", "bool")
        + field(cfg, "features.prompt_caching", "Prompt caching",
                "Reuse the static project context across calls (cheaper)", "bool"),
        open_=False)

    body = s_project + s_speakers + s_ai + s_audio_setup + s_audio + s_whisper + s_features
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Hankscribe Settings</title><style>{PAGE_CSS}</style></head><body>
<div class="wrap">
  <h1>HANKSCRIBE SETTINGS<span class="sub">config.json — served locally, nothing leaves this Mac</span></h1>
  {body}
  <p class="note">Advanced values (recording tunables, retrieval caps, model efforts)
  aren't shown here — edit config.json directly for those; this page preserves them on save.</p>
</div>
<div class="bar"><div class="in">
  <button id="launch">Save &amp; Launch Hankscribe</button>
  <button id="save">Save</button>
  <button id="close">Close</button>
  <span id="status"></span>
</div></div>
<script>{PAGE_JS}</script></body></html>"""


# ── Server ─────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code=200, body=b"", ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(body=render_page().encode())
        else:
            self._send(404, b"not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/save":
            try:
                save_config(json.loads(raw))
                self._send(body=b"ok", ctype="text/plain")
                print("  ✓ config.json saved")
            except Exception as e:
                self._send(500, f"save failed: {e}".encode(), "text/plain")
                print(f"  ✗ save failed: {e}")
        elif self.path == "/launch":
            _exit_action["action"] = "launch"
            self._send(body=b"ok", ctype="text/plain")
            _shutdown.set()
        elif self.path == "/quit":
            self._send(body=b"ok", ctype="text/plain")
            _shutdown.set()
        else:
            self._send(404, b"not found")


def main():
    # Any free localhost port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Hankscribe Settings → {url}")
    print("  (local only — closes when you press Close or Save & Launch)\n")
    webbrowser.open(url)

    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()

    if _exit_action["action"] == "launch":
        print("  Launching Hankscribe…")
        # New Terminal window via START.command so the meeting UI gets its own
        # window; falls back to replacing this process with the app itself.
        start = os.path.join(BASE_DIR, "START.command")
        if os.path.exists(start):
            subprocess.run(["open", start])
        else:
            os.execv(sys.executable, [sys.executable, os.path.join(BASE_DIR, "hankscribe2.py")])


if __name__ == "__main__":
    main()
