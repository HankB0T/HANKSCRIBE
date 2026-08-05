#!/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/bin/python3.13
"""
Hankscribe 2.0 — Real-time meeting transcription + AI Q&A powered by Claude.

Controls:
  1  →  Answer questions from transcript (Claude-powered, uses project context)
  2  →  Ask your own question (type it in)
  3  →  Auto-summary of meeting so far
  4  →  Screenshot current screen for additional context (this session only)
  r  →  Burst recording: poll the screen, keep frames on meaningful change
  m  →  Toggle AI mode: FAST (Opus 5, medium effort) ↔ DEEP (Opus 5, xhigh effort, most thorough)
  s  →  Show session stats
  q  →  Quit (auto-saves transcript + post-meeting digest)

Architecture:
  - config.json for all paths/models/thresholds
  - VAD streaming: segments flush on speech pauses, not fixed chunks
  - Speaker attribution from channel energy: [You] = mic, [Them] = BlackHole
  - whisper-server kept loaded + health-checked (auto-restart)
  - Bedrock Claude Opus 5 (fast=medium effort / deep=xhigh effort) with prompt caching
  - Semantic retrieval via local Ollama embeddings (keyword fallback)
  - Proactive answer suggestions when someone asks YOU a question
  - Post-meeting digest appended to the saved transcript on quit
"""

import sounddevice as sd
import numpy as np
import wave, subprocess, tempfile, os, sys, signal
import threading, queue, collections, time
import urllib.request, json, tty, termios, re, base64
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_CONFIG = {
    "paths": {
        "project_dir": "~/Desktop/MyProject",
        "master_context": "~/Desktop/MyProject/Master_Context.md",
        "transcript_dir": "~/Desktop/MyProject/Transcripts",
        "context_index": "context-index.json",
    },
    "audio": {
        "preferred_devices": ["Aggregate Device", "BlackHole 2ch"],
        "sample_rate": 48000,
        "silence_threshold": 0.0006,
        "vad_pause_sec": 0.5,
        "vad_max_segment_sec": 8.0,
        "mic_channel": 2,
    },
    "whisper": {
        "model": "~/whisper-models/ggml-base.en.bin",
        "port": 8199,
        "threads": 4,
    },
    "ai": {
        "bedrock_region": "us-east-1",
        "fast_model": "us.anthropic.claude-opus-5",
        "fast_max_tokens": 1200,
        "fast_effort": "medium",
        "deep_model": "us.anthropic.claude-opus-5",
        "deep_max_tokens": 4000,
        "deep_effort": "xhigh",
        "utility_model": "us.anthropic.claude-opus-5",
        "utility_effort": "low",
        "ollama_model": "llama3.2:3b",
        "embed_model": "nomic-embed-text",
    },
    "features": {
        "speaker_attribution": True,
        "semantic_retrieval": True,
        "prompt_caching": True,
        "post_meeting_digest": True,
        "answer_suggestions": True,
        "burst_recording": True,
        "max_attached_screenshots": 3,
    },
    "recording": {
        "burst_poll_sec": 0.5,
        "burst_change_threshold": 0.001,
        "burst_diff_pixel_delta": 30,
        "burst_batch_size": 4,
        "burst_max_attached": 5,
    },
    "user": {
        "name": "Your Name",
        "role": "your role on this project",
        "name_variants": ["your name"],
    },
    "project": {
        "name": "the project",
        "description": "your project",
    },
}


def _merge(base, override):
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config():
    path = os.path.join(BASE_DIR, "config.json")
    user_cfg = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  ⚠ config.json invalid ({e}) — using defaults")
    return _merge(_DEFAULT_CONFIG, user_cfg)


CFG = load_config()

SAMPLE_RATE     = CFG["audio"]["sample_rate"]
WHISPER_RATE    = 16000
SILENCE_THRESH  = CFG["audio"]["silence_threshold"]
VAD_PAUSE_SEC   = CFG["audio"]["vad_pause_sec"]
VAD_MAX_SEC     = CFG["audio"]["vad_max_segment_sec"]
VAD_MIN_SPEECH  = 0.35      # discard blips shorter than this
VAD_PREROLL_SEC = 0.25      # audio kept before speech starts
MIC_CHANNEL     = CFG["audio"]["mic_channel"]
PREFERRED_DEVICES = CFG["audio"]["preferred_devices"]

WHISPER_MODEL   = os.path.expanduser(CFG["whisper"]["model"])
WHISPER_PORT    = CFG["whisper"]["port"]
WHISPER_THREADS = CFG["whisper"]["threads"]

AI_MODES = {
    "fast": {
        "label": "FAST (Opus 5, medium effort, ~4-8s)",
        "model": CFG["ai"]["fast_model"],
        "max_tokens": CFG["ai"]["fast_max_tokens"],
        "effort": CFG["ai"].get("fast_effort", "medium"),
        "thinking": False,
    },
    "deep": {
        "label": "DEEP (Opus 5, xhigh effort, ~15-40s, most thorough)",
        "model": CFG["ai"]["deep_model"],
        "max_tokens": CFG["ai"]["deep_max_tokens"],
        "effort": CFG["ai"].get("deep_effort", "xhigh"),
        "thinking": True,
    },
}
# Background/bulk calls (screenshot inventory, burst visual description,
# proactive suggestions): Opus 5 at low effort, thinking off — cheap + fast.
UTILITY_MODE = {
    "label": "UTILITY (Opus 5, low effort)",
    "model": CFG["ai"].get("utility_model", CFG["ai"]["fast_model"]),
    "max_tokens": 600,
    "effort": CFG["ai"].get("utility_effort", "low"),
    "thinking": False,
}
ai_mode        = "deep"     # default: Opus 5 xhigh (press m for faster medium effort)
ai_mode_lock   = threading.Lock()
BEDROCK_REGION = CFG["ai"]["bedrock_region"]
OLLAMA_MODEL   = CFG["ai"]["ollama_model"]
OLLAMA_URL     = "http://127.0.0.1:11434/api/generate"
EMBED_MODEL    = CFG["ai"]["embed_model"]
EMBED_URL      = "http://127.0.0.1:11434/api/embed"

PROJECT_DIR     = os.path.expanduser(CFG["paths"]["project_dir"])
MASTER_CONTEXT  = os.path.expanduser(CFG["paths"]["master_context"])
TRANSCRIPT_DIR  = os.path.expanduser(CFG["paths"]["transcript_dir"])
CONTEXT_FILE    = os.path.join(BASE_DIR, CFG["paths"].get("context_index", "context-index.json"))
CORRECTIONS_FILE = os.path.join(BASE_DIR, "transcription-corrections.json")
SCREENSHOT_DIR  = os.path.join(BASE_DIR, "screenshots")
EMBED_VEC_FILE  = os.path.join(BASE_DIR, "embeddings.npz")
EMBED_META_FILE = os.path.join(BASE_DIR, "embeddings-meta.json")

FEATURES  = CFG["features"]
USER_NAME = CFG["user"]["name"]
USER_ROLE = CFG["user"]["role"]
USER_NAME_VARIANTS = [v.lower() for v in CFG["user"]["name_variants"]]
PROJECT   = CFG.get("project", {})
PROJECT_NAME = PROJECT.get("name", "the project")
PROJECT_DESC = PROJECT.get("description", PROJECT_NAME)
MAX_ATTACHED_SCREENSHOTS = FEATURES["max_attached_screenshots"]

RECORDING            = CFG.get("recording", {})
BURST_POLL_SEC       = RECORDING.get("burst_poll_sec", 1.5)
BURST_PIXEL_DELTA    = RECORDING.get("burst_diff_pixel_delta", 30)
BURST_MAX_ATTACHED   = RECORDING.get("burst_max_attached", 5)
# Content-driven capture tunables:
# STATIC_EPSILON — below this pixel-change fraction the screen is treated as
#   idle and OCR is skipped (unless the periodic refresh is due). Kept low so
#   even small changes (a few typed characters) still trigger OCR.
BURST_STATIC_EPSILON     = RECORDING.get("burst_static_epsilon", 0.002)
# OCR_REFRESH_SEC — force an OCR pass this often even on a static screen, to
#   catch slow reveals that don't move pixels much.
BURST_OCR_REFRESH_SEC    = RECORDING.get("burst_ocr_refresh_sec", 8.0)
# OCR_DUP_SIMILARITY — frames whose OCR text is at least this similar to the
#   last kept frame are discarded as duplicates (JPEG never written).
BURST_OCR_DUP_SIMILARITY = RECORDING.get("burst_ocr_dup_similarity", 0.92)
BURST_THIN_OCR_CHARS     = RECORDING.get("burst_thin_ocr_chars", 50)
BURST_THIN_OCR_PIXEL     = RECORDING.get("burst_thin_ocr_pixel", 0.05)

# ── State ──────────────────────────────────────────────────────────────────────
running              = True
transcript_lines     = []       # (timestamp, speaker, text) — speaker: "You"/"Them"/""
transcript_lock      = threading.Lock()
audio_queue          = queue.Queue()
last_answered_index  = 0
answered_lock        = threading.Lock()
screenshot_memory    = []       # (timestamp, path, description)
screenshot_lock      = threading.Lock()
whisper_proc         = None
session_start        = datetime.now()
bedrock_available    = False
original_term        = None
last_suggestion_time = 0.0      # debounce for proactive suggestions
_last_non_terminal_app = None   # tracks the most recent non-terminal focused app

# ── Session metrics (live cost/storage counter) ──
class SessionMetrics:
    """Thread-safe tracker for API calls, tokens, storage, and estimated cost."""
    # Approximate pricing per 1M tokens (Bedrock on-demand, USD)
    PRICING = {
        "us.anthropic.claude-opus-5":                   {"input": 5.00, "output": 25.0},
        "us.anthropic.claude-opus-4-8":                 {"input": 5.00, "output": 25.0},
        "us.anthropic.claude-haiku-4-5-20251001-v1:0":  {"input": 0.80, "output": 4.00},
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.0},
    }

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.screenshots_bytes = 0
        self.burst_frames = 0

    def record_call(self, model_id, input_tok, output_tok):
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tok
            self.output_tokens += output_tok

    def record_storage(self, nbytes, is_burst=False):
        with self._lock:
            self.screenshots_bytes += nbytes
            if is_burst:
                self.burst_frames += 1

    @property
    def estimated_cost(self):
        # Conservative: bill all input/output at Haiku rate unless we can
        # attribute per-model (we track aggregate). For a tighter estimate
        # we'd need per-model tracking; this is an upper bound for fast mode
        # and lower bound for deep mode — good enough for a live indicator.
        with self._lock:
            # Blend: assume 80% calls are Haiku, 20% are Opus (rough heuristic)
            # Actually, track exact cost per call below
            return self._cost

    @property
    def _cost(self):
        return getattr(self, '_total_cost', 0.0)

    def record_call_with_cost(self, model_id, input_tok, output_tok):
        p = self.PRICING.get(model_id, {"input": 1.0, "output": 5.0})
        cost = (input_tok * p["input"] + output_tok * p["output"]) / 1_000_000
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tok
            self.output_tokens += output_tok
            self._total_cost = getattr(self, '_total_cost', 0.0) + cost

    def summary(self):
        with self._lock:
            cost = getattr(self, '_total_cost', 0.0)
            storage_mb = self.screenshots_bytes / (1024 * 1024)
            return (f"Calls: {self.calls} | Tokens: {self.input_tokens:,}in/{self.output_tokens:,}out | "
                    f"Cost: ${cost:.3f} | Storage: {storage_mb:.1f}MB | Burst frames: {self.burst_frames}")

    def title_bar(self):
        """Compact one-liner for the terminal title bar."""
        with self._lock:
            cost = getattr(self, '_total_cost', 0.0)
            duration = datetime.now() - session_start
            mins = int(duration.total_seconds() // 60)
            burst_tag = " 🔴REC" if burst_active else ""
            return (f"HS2 | {mins}m | ${cost:.3f} | {self.calls} calls | "
                    f"{self.input_tokens + self.output_tokens:,} tok{burst_tag}")


metrics = SessionMetrics()


def show_inline_stats():
    """No-op. Stats only shown on 's' keypress or at quit."""
    pass


def title_bar_updater():
    """Background thread: tracks the last non-terminal app for burst lock detection."""
    global _last_non_terminal_app
    TERMINAL_BUNDLES = {"com.apple.terminal", "com.googlecode.iterm2", "net.kovidgoyal.kitty"}
    while running:
        try:
            name, bundle = _get_frontmost_app()
            if bundle and bundle.lower() not in TERMINAL_BUNDLES:
                _last_non_terminal_app = (name, bundle)
        except Exception:
            pass
        time.sleep(3)

# ── Burst recording state ──
burst_recordings     = []       # completed + in-progress burst dicts (see below)
burst_lock           = threading.RLock()
burst_active         = False    # is a burst currently recording?
burst_stop_event     = None     # threading.Event to signal the poll loop to stop
burst_thread         = None     # the polling daemon thread
burst_session_count  = 0        # increments per burst started this session

# ── Output helpers ─────────────────────────────────────────────────────────────

ASK_PROMPT   = "  \033[1;33mAsk:\033[0m "
input_state  = {"active": False, "buffer": ""}
input_lock   = threading.Lock()


def out(text=""):
    text = text.replace('\r', '').replace('\n', '\r\n')
    with input_lock:
        if input_state["active"]:
            sys.stdout.write('\r\x1b[K' + text + '\r\n')
            sys.stdout.write(ASK_PROMPT + input_state["buffer"])
        else:
            sys.stdout.write(text + '\r\n')
        sys.stdout.flush()


def print_response(title, text):
    import textwrap
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('\r', '').strip()

    out()
    out(f"  \033[1;36m{title}\033[0m")
    out("  " + "─" * 56)
    for line in text.split('\n'):
        line = line.rstrip()
        if not line:
            out()
            continue
        for wrapped in textwrap.wrap(line, width=74,
                                     break_long_words=True,
                                     break_on_hyphens=False) or ['']:
            out("  " + wrapped)
    out("  " + "─" * 56)
    out()


# ── Audio device detection ─────────────────────────────────────────────────────

def find_audio_device():
    """Return (device_id, channels, sample_rate). The sample rate is the
    device's ACTUAL default rate, not a hardcoded assumption — different
    headphones create a different Aggregate Device that may run at 44100 Hz
    instead of 48000, and assuming the wrong rate makes Whisper hear
    pitch-shifted audio and transcribe garbage."""
    devices = sd.query_devices()
    for preferred in PREFERRED_DEVICES:
        for i, d in enumerate(devices):
            if preferred.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                ch = d['max_input_channels']
                sr = int(round(d.get('default_samplerate') or SAMPLE_RATE))
                out(f"  Audio: {d['name']} (ID {i}, {ch}ch, {sr}Hz)")
                return i, ch, sr

    out("\n  \033[1;31m✗ No suitable audio input device found.\033[0m")
    out("  Available input devices:")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            out(f"    ID {i}: {d['name']} ({d['max_input_channels']}ch)")
    out("\n  Fix: Open Audio MIDI Setup → create Aggregate Device")
    out("  with BlackHole 2ch + your microphone.\n")
    sys.exit(1)


# ── Whisper server management + health check ──────────────────────────────────

def _server_alive(timeout=2):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{WHISPER_PORT}/", timeout=timeout)
        return True
    except Exception:
        return False


def start_whisper_server(announce=True):
    global whisper_proc
    if _server_alive(1):
        if announce:
            out("  Whisper: server already running")
        return True

    if announce:
        out(f"  Whisper: starting server ({os.path.basename(WHISPER_MODEL)})...")
    whisper_proc = subprocess.Popen(
        ['whisper-server', '-m', WHISPER_MODEL, '--port', str(WHISPER_PORT),
         '-t', str(WHISPER_THREADS), '--no-timestamps'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(40):
        time.sleep(0.3)
        if _server_alive(1):
            if announce:
                out("  Whisper: server ready")
            return True
    out("  \033[1;31m✗ Whisper server failed to start\033[0m")
    return False


def stop_whisper_server():
    global whisper_proc
    if whisper_proc:
        try:
            whisper_proc.terminate()
            whisper_proc.wait(timeout=5)
        except Exception:
            pass
        whisper_proc = None


def whisper_health_monitor():
    """Restart whisper-server if it dies mid-session (instead of silently
    degrading to the 4x-slower per-chunk whisper-cli fallback)."""
    consecutive_failures = 0
    while running:
        time.sleep(20)
        if not running:
            break
        if _server_alive(3):
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        if consecutive_failures >= 2:       # ~40s dead — restart it
            out("  \033[33m⚠ Whisper server unresponsive — restarting...\033[0m")
            stop_whisper_server()
            if start_whisper_server(announce=False):
                out("  \033[32m✓ Whisper server restarted\033[0m")
            consecutive_failures = 0


# ── Transcription ──────────────────────────────────────────────────────────────

HALLUCINATIONS = {
    'you', 'you.', 'thank you.', 'thank you', 'thanks for watching!',
    'thanks for watching.', 'thanks for watching', 'thank you for watching.',
    'thank you for watching', 'bye.', 'bye', '.', 'the', 'so', 'uh', 'um',
    'okay.', 'okay', 'yeah.', 'yeah', 'hmm.', 'hmm', 'oh.', 'oh', 'and',
    'thank you very much.', 'thank you so much.', 'subtitles by the amara.org community',
    "we'll see you next time.", 'see you next time.', 'silence.', 'silence',
}


def is_hallucination(text):
    return text.strip().lower() in HALLUCINATIONS


def detect_speaker(segment):
    """[You] if the mic channel dominates the segment energy, [Them] if the
    BlackHole (meeting) channels dominate. '' when indeterminate/unavailable."""
    if not FEATURES["speaker_attribution"]:
        return ""
    if segment.ndim < 2 or segment.shape[1] <= MIC_CHANNEL:
        return ""
    energies = np.abs(segment).mean(axis=0)
    mic_e = energies[MIC_CHANNEL]
    others = np.delete(energies, MIC_CHANNEL)
    them_e = others.max() if len(others) else 0.0
    if mic_e > them_e * 1.8:
        return "You"
    if them_e > mic_e * 1.8:
        return "Them"
    return "You" if mic_e >= them_e else "Them"


def _resample_to_16k(mono, in_rate):
    """Resample a mono float32 signal from in_rate to WHISPER_RATE (16 kHz).
    Uses linear interpolation so any input rate works — integer-ratio rates
    (48000→16000) and non-integer ones (44100→16000) alike. Assuming an
    integer decimation factor was the cause of garbled transcription when a
    device ran at 44100 Hz instead of 48000."""
    if in_rate == WHISPER_RATE or len(mono) == 0:
        return mono.astype(np.float32)
    n_out = int(round(len(mono) * WHISPER_RATE / in_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    # Sample positions in the source signal for each output sample
    src_idx = np.linspace(0, len(mono) - 1, n_out)
    return np.interp(src_idx, np.arange(len(mono)), mono).astype(np.float32)


def transcribe_segment(audio_data):
    # Energy-weighted mono mix — silent channels don't dilute the voice
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        channel_energy = np.abs(audio_data).mean(axis=0)
        if channel_energy.max() > 0:
            weights = channel_energy / channel_energy.sum()
            mono = (audio_data * weights).sum(axis=1).astype(np.float32)
        else:
            mono = audio_data.mean(axis=1).astype(np.float32)
    else:
        mono = audio_data.flatten().astype(np.float32)

    mono_16k = _resample_to_16k(mono, SAMPLE_RATE)
    pcm = np.clip(mono_16k * 32768, -32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    with wave.open(tmp, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(WHISPER_RATE)
        wf.writeframes(pcm.tobytes())

    try:
        import http.client
        boundary = '----HankscribeBoundary'
        with open(tmp, 'rb') as f:
            file_data = f.read()

        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'
        ).encode() + file_data + (
            f'\r\n--{boundary}\r\n'
            f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
            f'json\r\n--{boundary}--\r\n'
        ).encode()

        conn = http.client.HTTPConnection("127.0.0.1", WHISPER_PORT, timeout=15)
        conn.request("POST", "/inference", body=body,
                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        resp = conn.getresponse()
        result = json.loads(resp.read())
        conn.close()

        text = result.get('text', '').strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = text.strip()
        if is_hallucination(text):
            return None
        return text if text and len(text) > 1 else None

    except Exception:
        # Per-segment fallback while the health monitor restarts the server
        try:
            result = subprocess.run(
                ['whisper-cli', '-m', WHISPER_MODEL, '-f', tmp,
                 '--no-timestamps', '-l', 'en', '-bs', '5'],
                capture_output=True, timeout=20
            )
            text = result.stdout.decode('utf-8', errors='ignore').strip()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'whisper_.*', '', text)
            text = re.sub(r'ggml_.*', '', text)
            text = text.strip()
            if is_hallucination(text):
                return None
            return text if text and len(text) > 1 else None
        except Exception:
            pass
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return None


# ── VAD audio segmentation ─────────────────────────────────────────────────────
# Speech segments flush on natural pauses (VAD_PAUSE_SEC of silence) or at
# VAD_MAX_SEC. No fixed chunks, no overlap → no duplicated words, and short
# utterances appear as soon as you stop speaking instead of at chunk boundary.

class VadSegmenter:
    def __init__(self):
        self.blocks = []            # accumulated audio blocks
        self.frames = 0
        self.speech_frames = 0
        self.silence_run = 0
        self.had_speech = False
        self.lock = threading.Lock()
        self.pause_frames = int(VAD_PAUSE_SEC * SAMPLE_RATE)
        self.max_frames = int(VAD_MAX_SEC * SAMPLE_RATE)
        self.min_speech_frames = int(VAD_MIN_SPEECH * SAMPLE_RATE)
        self.preroll_frames = int(VAD_PREROLL_SEC * SAMPLE_RATE)

    def feed(self, block):
        peak = np.abs(block).mean(axis=0).max() if block.ndim > 1 else np.abs(block).mean()
        speaking = peak >= SILENCE_THRESH

        with self.lock:
            self.blocks.append(block.copy())
            self.frames += len(block)

            if speaking:
                self.had_speech = True
                self.speech_frames += len(block)
                self.silence_run = 0
            else:
                self.silence_run += len(block)

            if not self.had_speech:
                # Pure silence — keep only a short pre-roll so speech onsets
                # aren't clipped, without growing the buffer forever
                while self.frames - len(self.blocks[0]) >= self.preroll_frames:
                    self.frames -= len(self.blocks[0])
                    self.blocks.pop(0)
                return None

            # Flush on pause or max length
            if self.silence_run >= self.pause_frames or self.frames >= self.max_frames:
                segment = None
                if self.speech_frames >= self.min_speech_frames:
                    segment = np.concatenate(self.blocks)
                self.blocks = []
                self.frames = 0
                self.speech_frames = 0
                self.silence_run = 0
                self.had_speech = False
                return segment
        return None


segmenter = VadSegmenter()


def audio_callback(indata, frames, time_info, status):
    segment = segmenter.feed(indata)
    if segment is not None:
        audio_queue.put(segment)


def transcription_worker():
    while running:
        try:
            segment = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        text = transcribe_segment(segment)
        if text:
            text = apply_corrections(text)
            speaker = detect_speaker(segment)
            timestamp = datetime.now().strftime("%H:%M:%S")
            with transcript_lock:
                transcript_lines.append((timestamp, speaker, text))
            record_burst_transcript_line(timestamp, speaker, text)
            tag = f"\033[1;32m[You]\033[0m " if speaker == "You" else \
                  f"\033[1;34m[Them]\033[0m " if speaker == "Them" else ""
            rec_dot = "\033[1;31m🔴\033[0m " if burst_active else ""
            out(f"  {rec_dot}\033[90m[{timestamp}]\033[0m {tag}{text}")
            live_append_transcript(timestamp, speaker, text)
            maybe_suggest_answer(speaker, text)

        audio_queue.task_done()


# ── Corrections ────────────────────────────────────────────────────────────────

def load_corrections():
    try:
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('corrections', {}), data.get('important_terms', [])
    except Exception:
        pass
    # No corrections file → run with none. Copy
    # transcription-corrections.example.json to transcription-corrections.json
    # and add your project's names/terms (see README).
    return {}, []


CORRECTIONS, IMPORTANT_TERMS = load_corrections()


def apply_corrections(text):
    result = text
    lower = text.lower()
    for wrong, right in CORRECTIONS.items():
        if wrong in lower:
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(right, result)
    return result


# ── AI Backend (Bedrock Claude + Ollama fallback) ─────────────────────────────

def _extract_text(result):
    """Pull the first text block out of a Bedrock Claude response. With
    thinking enabled, content[0] can be a thinking block (no 'text' key), so
    we scan rather than index blindly. Returns '' if no text block present."""
    for block in result.get("content", []):
        if block.get("type") == "text" and "text" in block:
            return block["text"].strip()
    # Fallback: some blocks carry text without an explicit type
    for block in result.get("content", []):
        if "text" in block:
            return block["text"].strip()
    return ""


def current_mode():
    with ai_mode_lock:
        return AI_MODES[ai_mode]


def toggle_ai_mode():
    global ai_mode
    with ai_mode_lock:
        ai_mode = "deep" if ai_mode == "fast" else "fast"
        label = AI_MODES[ai_mode]["label"]
    out(f"\n  \033[1;35mAI mode → {label}\033[0m\n")


def build_system_prompt():
    """Static per-session prefix — identical across calls so Bedrock's prompt
    cache can serve it at ~10% of the input cost after the first call."""
    master = load_master_context()
    persona = (
        f"You are {USER_NAME}'s private meeting co-pilot for {PROJECT_DESC}. "
        f"{USER_NAME} is {USER_ROLE}. You help during live meetings: "
        f"answering questions with project facts, summarizing, and suggesting replies. "
        f"Be specific and concise; when the facts aren't in the context, say so."
    )
    if master:
        cap = CFG["ai"].get("master_context_chars", 40000)
        return persona + f"\n\nPROJECT MASTER CONTEXT ({PROJECT_NAME}):\n{master[:cap]}"
    return persona


def ask_claude(prompt, max_tokens=None, image_paths=None, system=None, model_override=None):
    """Call Claude on Bedrock. Images attach as pixels; the system prompt is
    cache_control-marked so the static project context is billed once."""
    global bedrock_available
    mode = model_override or current_mode()
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                              config=Config(read_timeout=120))

        if image_paths:
            content = []
            for i, path in enumerate(image_paths):
                try:
                    with open(path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    media = "image/jpeg" if path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
                    content.append({"type": "text", "text": f"[Screenshot {i+1}]"})
                    content.append({"type": "image",
                                    "source": {"type": "base64", "media_type": media, "data": img_b64}})
                except OSError:
                    continue
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or mode["max_tokens"],
            "messages": [{"role": "user", "content": content}],
        }
        # Effort controls how hard Opus 5 thinks (low→max). Thinking on the
        # deep mode; off (disabled) on fast/utility to minimise latency —
        # HS2 uses no tools, so disabled thinking is safe here.
        effort = mode.get("effort")
        if effort:
            payload["output_config"] = {"effort": effort}
        if mode.get("thinking"):
            payload["thinking"] = {"type": "adaptive"}
        else:
            # disabled thinking is only valid at effort <= high on Opus 5
            payload["thinking"] = {"type": "disabled"}
        use_cache = bool(system) and FEATURES["prompt_caching"]
        if system:
            block = {"type": "text", "text": system}
            if use_cache:
                block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [block]

        try:
            resp = client.invoke_model(modelId=mode["model"], body=json.dumps(payload))
        except Exception:
            if not use_cache:
                raise
            # Retry without cache_control (model/region may not support it)
            payload["system"] = [{"type": "text", "text": system}]
            resp = client.invoke_model(modelId=mode["model"], body=json.dumps(payload))

        result = json.loads(resp["body"].read())
        bedrock_available = True
        usage = result.get("usage", {})
        metrics.record_call_with_cost(
            mode["model"],
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0)
        )
        text = _extract_text(result)
        return text or None
    except Exception:
        bedrock_available = False
        return None


def ask_claude_with_image(prompt, image_path, max_tokens=500):
    # Screenshot inventory always uses the fast model
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                              config=Config(read_timeout=60))
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        media = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "output_config": {"effort": UTILITY_MODE["effort"]},
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": img_b64}},
                {"type": "text", "text": prompt}
            ]}]
        })
        resp = client.invoke_model(modelId=UTILITY_MODE["model"], body=body)
        result = json.loads(resp["body"].read())
        usage = result.get("usage", {})
        metrics.record_call_with_cost(
            UTILITY_MODE["model"],
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0)
        )
        return _extract_text(result) or "[Screenshot analysis returned no text]"
    except Exception as e:
        return f"[Screenshot analysis failed: {e}]"


def ask_ollama(prompt, max_tokens=300):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens or 300, "temperature": 0.3, "top_p": 0.9}
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception:
        return None


def ask_ai(prompt, max_tokens=None, image_paths=None, system=None, model_override=None):
    answer = ask_claude(prompt, max_tokens, image_paths=image_paths,
                        system=system, model_override=model_override)
    if answer:
        return answer
    # Ollama fallback is text-only; fold the system prompt into the prompt
    full = f"{system}\n\n{prompt}" if system else prompt
    answer = ask_ollama(full, max_tokens)
    if answer:
        return answer
    return "[No AI backend available. Check Bedrock credentials or start Ollama.]"


# ── Screenshot memory + relevance selection ────────────────────────────────────

SCREEN_REFERENCE = re.compile(
    r'\b(screenshot|screen shot|on (the |my )?screen|this (page|window|tab|diagram|'
    r'table|chart|slide|board)|shared screen|what.{0,10}(showing|displayed))\b',
    re.IGNORECASE)


def select_relevant_screenshots(question_text):
    """Attach only screenshots whose description overlaps the question —
    each image costs ~1-1.5K tokens. Descriptions always ride along as text.
    Searches both manual screenshots (key 3) and burst-recording frames; the
    combined attachment count is capped at BURST_MAX_ATTACHED."""
    with screenshot_lock:
        shots = [(ts, path, desc) for ts, path, desc in screenshot_memory
                 if os.path.exists(path)]

    q_kws = keywords_only(apply_corrections(question_text))

    # --- Manual screenshots (original logic, unchanged) ---
    scored = []
    for idx, (ts, path, desc) in enumerate(shots):
        overlap = len(q_kws & keywords_only(desc))
        if overlap > 0:
            scored.append((overlap, idx, path))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [path for _, _, path in scored[:MAX_ATTACHED_SCREENSHOTS]]

    if SCREEN_REFERENCE.search(question_text) and shots:
        latest = shots[-1][1]
        if latest not in selected:
            selected = ([latest] + selected)[:MAX_ATTACHED_SCREENSHOTS]

    # --- Burst-recording frames (OCR text + visual descriptions) ---
    # Score against OCR text and visual_description. Frames only carry an image
    # if their JPEG still exists on disk (thin_ocr key frames + manual captures);
    # text-only matches still contribute their words but attach no image.
    burst_frames = []
    if FEATURES.get("burst_recording"):
        with burst_lock:
            for rec in burst_recordings:
                for fr in rec.get("frames", []):
                    if fr.get("duplicate"):
                        continue
                    searchable = " ".join(filter(None, [
                        fr.get("ocr_text", ""),
                        fr.get("visual_description", ""),
                        fr.get("description", ""),
                    ]))
                    burst_frames.append((fr.get("timestamp", ""),
                                         fr.get("path", ""),
                                         searchable))
    burst_scored = []
    for idx, (ts, path, searchable) in enumerate(burst_frames):
        overlap = len(q_kws & keywords_only(searchable))
        if overlap > 0:
            burst_scored.append((overlap, idx, path))
    burst_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Merge, keeping manual screenshots first, capped at the combined limit.
    # Only attach frames whose image is still on disk.
    for _, _, path in burst_scored:
        if path and os.path.exists(path) and path not in selected:
            selected.append(path)

    return selected[:BURST_MAX_ATTACHED]


def get_screenshot_context():
    with screenshot_lock:
        shots = screenshot_memory[:]
    if not shots:
        return ""
    parts = [f"[Screenshot {i+1} at {ts}]\n{desc}" for i, (ts, _, desc) in enumerate(shots)]
    return "\n\n".join(parts)


# ── Context loading ────────────────────────────────────────────────────────────

_master_context_cache = None
_master_context_mtime = 0


def load_master_context():
    global _master_context_cache, _master_context_mtime
    if not os.path.exists(MASTER_CONTEXT):
        return ""
    mtime = os.path.getmtime(MASTER_CONTEXT)
    if mtime != _master_context_mtime:
        with open(MASTER_CONTEXT, 'r', encoding='utf-8', errors='ignore') as f:
            _master_context_cache = f.read()
        _master_context_mtime = mtime
    return _master_context_cache


def context_is_stale():
    if not os.path.exists(CONTEXT_FILE):
        return True
    index_mtime = os.path.getmtime(CONTEXT_FILE)
    supported = {'.md', '.txt', '.vtt', '.docx', '.pdf', '.pptx', '.olm'}
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            if os.path.splitext(name)[1].lower() in supported:
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) > index_mtime:
                        return True
                except OSError:
                    pass
    return False


def rebuild_context_async():
    def _rebuild():
        script = os.path.join(BASE_DIR, "build-context.py")
        try:
            result = subprocess.run([sys.executable, script],
                                    capture_output=True, timeout=300)
            if result.returncode == 0:
                out("  \033[32m✓ Project context index refreshed\033[0m")
                build_embeddings_async()   # re-embed the fresh index
            else:
                out("  \033[33m⚠ Context rebuild failed — using existing index\033[0m")
        except Exception:
            out("  \033[33m⚠ Context rebuild error — using existing index\033[0m")
    threading.Thread(target=_rebuild, daemon=True).start()


def load_indexed_context():
    if not os.path.exists(CONTEXT_FILE):
        return None
    try:
        with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had','do','does',
    'did','will','would','could','should','may','might','shall','can','what',
    'who','where','when','why','how','that','this','these','those','it','its',
    'i','you','he','she','we','they','me','him','her','us','them','my','your',
    'his','our','their','not','no','so','as','if','then','than','there','here',
    'just','also','about','up','out','into','over','after','before','any','all'
}


def keywords_only(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in STOP_WORDS}


# ── Semantic retrieval (Ollama embeddings, keyword fallback) ───────────────────

_embed_index = {"ready": False, "built_at": None, "vecs": None, "chunks": None}
_embed_lock  = threading.Lock()


def _ollama_embed(texts):
    """Embed a list of texts via local Ollama. Returns list of vectors or None."""
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()).get("embeddings")
    except Exception:
        return None


def _chunk_context(context_data, chunk_chars=700):
    """Split indexed files into retrieval chunks. VTT/OLM get bigger chunks
    (low structure); structured docs keep paragraph granularity."""
    chunks = []
    for file_info in context_data.get('files', []):
        name = file_info['name']
        text = file_info['text']
        size = chunk_chars * 2 if file_info.get('type') in ('.vtt', '.olm') else chunk_chars
        lines = [l.strip() for l in text.splitlines() if l.strip()] or [text]
        cur = ""
        for line in lines:
            if len(cur) + len(line) > size and cur:
                chunks.append({"file": name, "text": cur})
                cur = line
            else:
                cur = (cur + "\n" + line) if cur else line
        if cur:
            chunks.append({"file": name, "text": cur})
    return chunks[:6000]


def build_embeddings_async():
    """Embed the context index in the background; semantic search activates
    when done. Cached on disk keyed by the index build timestamp."""
    if not FEATURES["semantic_retrieval"]:
        return

    def _build():
        context_data = load_indexed_context()
        if not context_data:
            return
        built_at = context_data.get('built_at')

        # Try disk cache first
        try:
            with open(EMBED_META_FILE, 'r') as f:
                meta = json.load(f)
            if meta.get('built_at') == built_at:
                vecs = np.load(EMBED_VEC_FILE)['vecs']
                with _embed_lock:
                    _embed_index.update(ready=True, built_at=built_at,
                                        vecs=vecs, chunks=meta['chunks'])
                out(f"  \033[32m✓ Semantic index loaded ({len(meta['chunks'])} chunks)\033[0m")
                return
        except Exception:
            pass

        chunks = _chunk_context(context_data)
        if not chunks:
            return
        all_vecs = []
        for i in range(0, len(chunks), 64):
            if not running:
                return
            batch = [c['text'][:2000] for c in chunks[i:i + 64]]
            vecs = _ollama_embed(batch)
            if vecs is None:
                out("  \033[33m⚠ Embeddings unavailable (Ollama down?) — keyword search active\033[0m")
                return
            all_vecs.extend(vecs)

        arr = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms

        try:
            np.savez_compressed(EMBED_VEC_FILE, vecs=arr)
            with open(EMBED_META_FILE, 'w') as f:
                json.dump({"built_at": built_at, "chunks": chunks}, f)
        except Exception:
            pass

        with _embed_lock:
            _embed_index.update(ready=True, built_at=built_at, vecs=arr, chunks=chunks)
        out(f"  \033[32m✓ Semantic index built ({len(chunks)} chunks)\033[0m")

    threading.Thread(target=_build, daemon=True).start()


def semantic_search(query, max_chars=None):
    # Wider retrieval than the original 8-chunk / 4K-char / 0.35-floor defaults:
    # Opus 5 has ample context, and the previous caps were dropping relevant
    # passages. All three are configurable under CFG["ai"].
    if max_chars is None:
        max_chars = CFG["ai"].get("retrieval_max_chars", 12000)
    top_k = CFG["ai"].get("retrieval_top_k", 20)
    floor = CFG["ai"].get("retrieval_floor", 0.30)
    with _embed_lock:
        if not _embed_index["ready"]:
            return None
        vecs = _embed_index["vecs"]
        chunks = _embed_index["chunks"]

    qv = _ollama_embed([apply_corrections(query)[:2000]])
    if not qv:
        return None
    q = np.array(qv[0], dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn == 0:
        return None
    q = q / qn

    sims = vecs @ q
    top = np.argsort(sims)[::-1][:top_k]

    result_parts, total, seen_files = [], 0, {}
    for idx in top:
        if sims[idx] < floor:           # relevance floor
            continue
        c = chunks[int(idx)]
        if total + len(c['text']) > max_chars:
            continue
        seen_files.setdefault(c['file'], []).append(c['text'])
        total += len(c['text'])
    for fname, texts in seen_files.items():
        result_parts.append(f"[{fname}]\n" + "\n".join(texts))

    return "\n\n".join(result_parts) if result_parts else None


def keyword_search(query):
    context_data = load_indexed_context()
    if not context_data:
        return ""

    query_kws = keywords_only(apply_corrections(query))
    if not query_kws:
        return ""

    scored_files = []
    for file_info in context_data.get('files', []):
        file_kws = keywords_only(file_info['text'])
        file_kws |= keywords_only(file_info['name'])
        file_kws |= {k.lower() for k in file_info.get('keywords', [])}

        overlap = query_kws & file_kws
        score = len(overlap)

        ext = file_info.get('type', '')
        if ext in ('.md', '.docx', '.pptx'):
            score = int(score * 1.5)
        elif ext == '.vtt':
            score = int(score * 0.5)

        if score > 0:
            scored_files.append((score, file_info))

    scored_files.sort(key=lambda x: x[0], reverse=True)

    result_parts = []
    total_chars = 0
    for _, file_info in scored_files[:3]:
        if total_chars >= 4000:
            break
        lines = [l.strip() for l in file_info['text'].splitlines() if l.strip()]
        line_kws = [(len(query_kws & keywords_only(l)), l) for l in lines]
        line_kws.sort(key=lambda x: x[0], reverse=True)
        passage_lines = []
        for score, line in line_kws:
            if score > 0 and total_chars + len(line) < 4000:
                passage_lines.append(line)
                total_chars += len(line)
        if passage_lines:
            result_parts.append(f"[{file_info['name']}]\n" + "\n".join(passage_lines[:15]))

    return "\n\n".join(result_parts)


def search_context(query):
    """Semantic when the embedding index is ready, keyword otherwise."""
    if FEATURES["semantic_retrieval"]:
        result = semantic_search(query)
        if result:
            return result
    return keyword_search(query)


# ── Question detection ─────────────────────────────────────────────────────────

QUESTION_WORDS = re.compile(
    r'\b(what|who|where|when|why|how|is|are|was|were|do|does|did|can|could|'
    r'would|should|will|have|has|had)\b', re.IGNORECASE)

FILLER_QUESTION = re.compile(
    r'^(can (you|everyone|we)|do you (hear|see|understand)|'
    r'are you (there|ready|okay|with me)|'
    r'does (that|this) (make sense|work)|okay\??$|right\??$|'
    r'you (know|understand|see))',
    re.IGNORECASE)

SPELLING_OUT = re.compile(r'\b([A-Za-z]-){2,}[A-Za-z]\b|spelled\b|spell\b', re.IGNORECASE)


def is_question(line):
    line = line.strip()
    if not line:
        return False
    if SPELLING_OUT.search(line):
        return False
    if FILLER_QUESTION.match(line):
        return False
    if line.endswith('?') and len(line.split()) >= 3:
        return True
    if QUESTION_WORDS.match(line) and len(line.split()) >= 4:
        return True
    return False


def find_questions(lines, start_from=0):
    search_lines = [t[-1] for t in lines[start_from:]]
    if not search_lines:
        return []

    questions = []
    non_q_streak = 0

    for line in reversed(search_lines):
        line = line.strip()
        if not line:
            continue
        if is_question(line):
            questions.append(line)
            non_q_streak = 0
        else:
            non_q_streak += 1
            if non_q_streak >= 4 and questions:
                break

    questions.reverse()

    deduped, seen = [], set()
    for q in questions:
        key = apply_corrections(q.lower())[:40]
        if key not in seen:
            deduped.append(q)
            seen.add(key)

    return deduped


# ── Proactive answer suggestions ───────────────────────────────────────────────

def is_directed_at_user(text):
    """A question from [Them] that names the user."""
    lower = text.lower()
    return any(v in lower for v in USER_NAME_VARIANTS)


def maybe_suggest_answer(speaker, text):
    """When someone else asks a question naming the user, proactively flash a
    suggested answer — no keypress needed. Debounced, always the fast model."""
    global last_suggestion_time
    if not FEATURES["answer_suggestions"]:
        return
    if speaker != "Them" or not is_question(text) or not is_directed_at_user(text):
        return
    now = time.time()
    if now - last_suggestion_time < 15:
        return
    last_suggestion_time = now

    def _suggest():
        with transcript_lock:
            lines = transcript_lines[:]
        recent = "\n".join(f"[{sp or '?'}] {tx}" for _, sp, tx in lines[-12:])
        doc_context = search_context(text)
        ctx = f"\n\nRELEVANT DOCUMENTS:\n{doc_context}" if doc_context else ""
        prompt = (
            f"Someone in the meeting just asked {USER_NAME} directly:\n\"{text}\"\n\n"
            f"RECENT TRANSCRIPT:\n{recent}{ctx}\n\n"
            f"Draft the reply {USER_NAME} should say out loud, in first person, "
            f"2-3 sentences, specific and confident. If the facts aren't available, "
            f"draft a graceful 'let me get back to you' that still shows command of the topic."
        )
        answer = ask_claude(prompt, max_tokens=250,
                            system=build_system_prompt(),
                            model_override=UTILITY_MODE)
        if answer:
            print_response("💡 SUGGESTED REPLY (they asked you)", answer)

    threading.Thread(target=_suggest, daemon=True).start()


# ── Key handlers ───────────────────────────────────────────────────────────────

def handle_key_1():
    global last_answered_index

    with transcript_lock:
        lines = transcript_lines[:]
    with answered_lock:
        answered_from = last_answered_index

    if not lines:
        out("\n  No transcript yet.\n")
        return

    questions = find_questions(lines, start_from=answered_from)
    if not questions:
        questions = find_questions(lines, start_from=max(0, len(lines) - 30))

    recent_text = "\n".join(f"[{sp or '?'}] {tx}" for _, sp, tx in lines[-20:])
    system = build_system_prompt()

    if not questions:
        out("\n  \033[33mNo obvious questions found — identifying discussion points...\033[0m")
        prompt = (
            f"Transcript ([You] = {USER_NAME} speaking, [Them] = other participants):\n"
            f"{recent_text}\n\n"
            f"Identify 1-3 implicit questions or topics people are discussing that "
            f"{USER_NAME} could clarify. For each, write the answer as {USER_NAME} would "
            f"say it in the meeting (first person, confident, citing specifics). Format:\n"
            f"Q: [question]\nA: [{USER_NAME}'s answer]\n"
        )
        answer = ask_ai(prompt, system=system)
        print_response("DISCUSSION POINTS", answer)
        with answered_lock:
            last_answered_index = len(lines)
        return

    out(f"\n  \033[33m{len(questions)} question(s) detected — {current_mode()['label']} thinking...\033[0m")

    doc_context = search_context(" ".join(questions))
    screen_ctx = get_screenshot_context()
    recordings_ctx = get_all_recordings_context(max_chars=8000)

    context_parts = []
    if doc_context:
        context_parts.append(f"RELEVANT DOCUMENTS:\n{doc_context}")
    if screen_ctx:
        context_parts.append(f"SCREENSHOTS CAPTURED THIS SESSION:\n{screen_ctx}")
    if recordings_ctx:
        context_parts.append(f"SCREEN RECORDINGS THIS SESSION (OCR of what was on screen):\n{recordings_ctx}")
    context_block = "\n\n".join(context_parts)

    q_list = "\n".join(f"  - {q}" for q in questions)

    prompt = (
        f"MEETING TRANSCRIPT (recent; [You] = {USER_NAME}, [Them] = others):\n{recent_text}\n\n"
        f"{context_block}\n\n"
        f"QUESTIONS ASKED IN THE MEETING:\n{q_list}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. First, consolidate: if multiple questions are about the same topic or one is a "
        f"follow-up/subset of another, merge them into a single comprehensive question. "
        f"If they're genuinely distinct, keep them separate.\n"
        f"2. For each consolidated question, provide {USER_NAME}'s answer — write as if "
        f"{USER_NAME} is answering in the meeting. Use first person (\"We have...\", "
        f"\"I confirmed...\", \"The plan is...\"). Be confident, specific, and cite project facts.\n"
        f"3. Keep each answer 2-4 sentences. Speak naturally, not like documentation.\n"
        f"4. If the docs don't cover something, say \"I'll need to check on that and get back to you.\"\n\n"
        f"Format:\n"
        f"Q: [consolidated question]\n"
        f"A: [{USER_NAME}'s answer in first person]\n"
    )

    relevant_shots = select_relevant_screenshots(" ".join(questions) + " " + recent_text[-500:])
    if relevant_shots:
        out(f"  \033[90m(attaching {len(relevant_shots)} relevant screenshot(s))\033[0m")
    answer = ask_ai(prompt, image_paths=relevant_shots, system=system)
    print_response(f"ANSWERS ({len(questions)} questions)", answer)

    with answered_lock:
        last_answered_index = len(lines)


def handle_key_2():
    with transcript_lock:
        lines = transcript_lines[:]

    if not lines:
        out("\n  No transcript yet.\n")
        return

    out(f"\n  \033[33mGenerating summary ({current_mode()['label']})...\033[0m")

    recent = "\n".join(f"[{ts}] [{sp or '?'}] {tx}" for ts, sp, tx in lines[-25:])
    screen_ctx = get_screenshot_context()
    extra = f"\n\nScreenshots captured during meeting:\n{screen_ctx}" if screen_ctx else ""

    prompt = (
        f"Summarize this meeting segment ([You] = {USER_NAME}, [Them] = other participants).\n\n"
        f"Transcript:\n{recent}{extra}\n\n"
        f"Provide a concise summary with:\n"
        f"1. KEY TOPICS (2-4 bullet points of what was discussed)\n"
        f"2. DECISIONS MADE (if any)\n"
        f"3. ACTION ITEMS (if any mentioned, format: [Person]: task)\n"
        f"4. OPEN QUESTIONS (unresolved items)\n\n"
        f"Be brief and specific. Use names when mentioned."
    )

    answer = ask_ai(prompt, system=build_system_prompt())
    print_response("MEETING SUMMARY", answer)


def answer_typed_question(question):
    with transcript_lock:
        lines = transcript_lines[:]
    recent_text = "\n".join(f"[{sp or '?'}] {tx}" for _, sp, tx in lines[-20:]) if lines else "(no transcript yet)"

    out(f"\n  \033[33mThinking ({current_mode()['label']})...\033[0m")

    doc_context = search_context(question)
    screen_ctx = get_screenshot_context()
    recordings_ctx = get_all_recordings_context(max_chars=8000)

    context_parts = []
    if doc_context:
        context_parts.append(f"RELEVANT DOCUMENTS:\n{doc_context}")
    if screen_ctx:
        context_parts.append(f"SCREENSHOTS CAPTURED THIS SESSION:\n{screen_ctx}")
    if recordings_ctx:
        context_parts.append(f"SCREEN RECORDINGS THIS SESSION (OCR of what was on screen):\n{recordings_ctx}")
    context_block = "\n\n".join(context_parts)

    prompt = (
        f"{USER_NAME} is asking this question privately (not heard by others in the meeting).\n\n"
        f"QUESTION: {question}\n\n"
        f"MEETING TRANSCRIPT (recent):\n{recent_text}\n\n"
        f"{context_block}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer as a knowledgeable colleague briefing {USER_NAME}\n"
        f"- Be specific, cite facts from the docs where possible\n"
        f"- If the answer has multiple parts, use short bullets\n"
        f"- If the docs don't cover it, say what you do know and what's missing\n"
        f"- If screenshots are attached, read them directly for any details asked about\n"
        f"- 3-5 sentences unless more detail is needed"
    )

    relevant_shots = select_relevant_screenshots(question)
    if relevant_shots:
        out(f"  \033[90m(attaching {len(relevant_shots)} relevant screenshot(s))\033[0m")
    answer = ask_ai(prompt, image_paths=relevant_shots, system=build_system_prompt())
    print_response("ANSWER", answer)


def display_under_cursor():
    try:
        import Quartz
        err, ids, cnt = Quartz.CGGetActiveDisplayList(10, None, None)
        pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        for i, did in enumerate(ids):
            b = Quartz.CGDisplayBounds(did)
            if (b.origin.x <= pos.x < b.origin.x + b.size.width and
                    b.origin.y <= pos.y < b.origin.y + b.size.height):
                return i + 1
    except Exception:
        pass
    return 1


def handle_key_3():
    display = display_under_cursor()
    out(f"\n  \033[33mCapturing screenshot (display {display})...\033[0m")

    session_dir = os.path.join(SCREENSHOT_DIR, session_start.strftime("%Y-%m-%d_%H%M"))
    os.makedirs(session_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(session_dir, f"screen_{timestamp}.png")

    result = subprocess.run(
        ['screencapture', '-x', '-C', '-D', str(display), filepath],
        capture_output=True, timeout=5
    )

    if result.returncode != 0 or not os.path.exists(filepath):
        out("  \033[1;31m✗ Screenshot failed\033[0m")
        return

    jpeg_path = filepath.replace('.png', '.jpg')
    comp = subprocess.run(
        ['sips', '-Z', '1600', '-s', 'format', 'jpeg',
         '-s', 'formatOptions', '80', filepath, '--out', jpeg_path],
        capture_output=True, timeout=10
    )
    if comp.returncode == 0 and os.path.exists(jpeg_path):
        os.unlink(filepath)
        filepath = jpeg_path

    file_size = os.path.getsize(filepath)
    metrics.record_storage(file_size)
    out(f"  Screenshot saved ({file_size // 1024} KB)")
    out("  Analyzing with Claude...")

    description = ask_claude_with_image(
        "Create a searchable inventory of this screen. List:\n"
        "1. Application and window/page title\n"
        "2. Every distinct name, ID, number, date, and status you can read "
        "(ticket IDs, server names, IP/CIDR ranges, percentages, deadlines)\n"
        "3. Table contents row by row if a table is visible\n"
        "4. One line on what this screen is about overall\n"
        "Be exhaustive with specifics — questions will be matched against these words.",
        filepath, max_tokens=500
    )

    if description and not description.startswith("[Screenshot analysis failed"):
        ts = datetime.now().strftime("%H:%M:%S")
        with screenshot_lock:
            screenshot_memory.append((ts, filepath, description))
            count = len(screenshot_memory)
        # If a burst is recording, link this manual capture into its timeline
        # (permanent, never deleted at finalize time).
        if burst_active:
            with burst_lock:
                if burst_active and burst_recordings:
                    burst_recordings[-1]["frames"].append({
                        "timestamp": ts,
                        "type": "manual_capture",
                        "path": filepath,
                        "description": description,
                        "ocr_text": _vision_ocr(filepath),
                    })
        print_response(f"SCREEN CONTEXT #{count} CAPTURED", description)
    else:
        out(f"  \033[1;31m✗ Analysis failed — screenshot saved at {filepath}\033[0m\n")


# ── Burst recording ────────────────────────────────────────────────────────────
# Press `r` to start/stop. While active, the screen is polled every
# BURST_POLL_SEC. A cheap thumbnail pixel-diff only skips truly-static frames;
# every non-static frame is OCR'd locally (macOS Vision, free/on-device) and the
# keep-vs-discard decision is made on the OCR TEXT that changed — so typed text
# (which barely moves pixels) is reliably captured. Duplicate-text frames are
# discarded without ever writing a JPEG. Text-sparse frames with a large visual
# change (diagrams/charts) are kept as thin_ocr keyframes and get a one-time
# Opus 5 visual description. On session end each burst is grouped into "screen
# states" in recording.json; only thin_ocr keyframes + manual captures keep their
# JPEGs.

def _burst_dir(n):
    session = session_start.strftime("%Y-%m-%d_%H%M")
    return os.path.join(SCREENSHOT_DIR, f"burst_{session}_{n}")


def _thumbnail_array(png_path):
    """Downsample a PNG to a 160x90 24-bit BMP and load its pixels as a numpy
    array (no PIL dependency). Returns None on failure."""
    try:
        tmp_bmp = png_path + ".thumb.bmp"
        r = subprocess.run(
            ['sips', '-z', '90', '160', '-s', 'format', 'bmp', png_path, '--out', tmp_bmp],
            capture_output=True, timeout=10
        )
        if r.returncode != 0 or not os.path.exists(tmp_bmp):
            return None
        with open(tmp_bmp, 'rb') as f:
            data = f.read()
        try:
            os.unlink(tmp_bmp)
        except OSError:
            pass
        if len(data) < 54 or data[:2] != b'BM':
            return None
        # Pixel data offset (little-endian, bytes 10-13)
        offset = int.from_bytes(data[10:14], 'little')
        pixels = np.frombuffer(data[offset:], dtype=np.uint8)
        # 160 wide * 3 bytes = 480 (4-byte aligned → no row padding)
        usable = (len(pixels) // 3) * 3
        return pixels[:usable].astype(np.int16)
    except Exception:
        return None


def _frames_differ(prev_arr, cur_arr):
    """Fraction of channel-samples that differ by more than BURST_PIXEL_DELTA."""
    if prev_arr is None or cur_arr is None:
        return 1.0
    n = min(len(prev_arr), len(cur_arr))
    if n == 0:
        return 1.0
    diff = np.abs(prev_arr[:n] - cur_arr[:n]) > BURST_PIXEL_DELTA
    return float(diff.mean())


def _compress_burst_frame(png_path, out_path):
    """Compress a full-res PNG to a small JPEG (quality 60, max 1200px)."""
    r = subprocess.run(
        ['sips', '-Z', '1200', '-s', 'format', 'jpeg',
         '-s', 'formatOptions', '60', png_path, '--out', out_path],
        capture_output=True, timeout=10
    )
    return r.returncode == 0 and os.path.exists(out_path)


def _vision_ocr(image_path):
    """Extract all visible text from an image using macOS Vision framework.
    Runs locally, no API cost. Returns extracted text string or ''."""
    try:
        import objc
        Vision = objc.loadBundle('Vision', bundle_path='/System/Library/Frameworks/Vision.framework',
                                 module_globals={})
        from Quartz import CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(image_path)
        source = CGImageSourceCreateWithURL(url, None)
        if not source:
            return ""
        image = CGImageSourceCreateImageAtIndex(source, 0, None)
        if not image:
            return ""

        # Need to get the classes from the loaded bundle
        VNImageRequestHandler = objc.lookUpClass('VNImageRequestHandler')
        VNRecognizeTextRequest = objc.lookUpClass('VNRecognizeTextRequest')

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(0)  # 0 = fast (good enough for screen text)

        result = handler.performRequests_error_([request], None)
        success = result[0] if isinstance(result, tuple) else result
        if not success:
            return ""

        results = request.results()
        if not results:
            return ""

        lines = []
        for obs in results:
            candidates = obs.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())
        return "\n".join(lines)
    except Exception:
        return ""


def _ocr_lines(text):
    """Normalized, non-empty lines of an OCR text blob."""
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _ocr_similarity(prev_text, cur_text):
    """Fraction of shared lines vs. all distinct lines across both frames
    (Jaccard on lines). 1.0 = identical text, 0.0 = nothing in common."""
    prev = set(_ocr_lines(prev_text))
    cur = set(_ocr_lines(cur_text))
    if not prev and not cur:
        return 1.0
    union = prev | cur
    if not union:
        return 1.0
    return len(prev & cur) / len(union)


def _ocr_diff(prev_text, cur_text):
    """Line-level diff between two OCR frames: '+' for lines that appeared,
    '-' for lines that disappeared."""
    prev_lines = _ocr_lines(prev_text)
    cur_lines = _ocr_lines(cur_text)
    prev_set = set(prev_lines)
    cur_set = set(cur_lines)
    added = [l for l in cur_lines if l not in prev_set]
    removed = [l for l in prev_lines if l not in cur_set]
    parts = [f"+{l}" for l in added] + [f"-{l}" for l in removed]
    return "\n".join(parts)


_FRONTMOST_SCRIPT = '''tell application "System Events"
    set frontApp to first application process whose frontmost is true
    return {name of frontApp, bundle identifier of frontApp}
end tell'''


def _get_frontmost_app():
    """Return (name, bundle_id) of the currently focused application."""
    try:
        r = subprocess.run(['osascript', '-e', _FRONTMOST_SCRIPT],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(', ')
            if len(parts) >= 2:
                return (parts[0], parts[1])
            elif parts[0]:
                return (parts[0], "")
    except Exception:
        pass
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return (app.localizedName(), app.bundleIdentifier() or "")
    except Exception:
        pass
    return ("", "")


def burst_poll_loop(rec, stop_event):
    """Content-driven screen recorder.

    Every poll we capture the screen and, unless it is byte-for-byte static,
    OCR it locally (macOS Vision — free, on-device). The keep-vs-discard
    decision is made on the *OCR text that changed*, NOT on pixel deltas. This
    is deliberate: typing a line into a chat box moves only a handful of pixels
    on a downscaled thumbnail — far below any pixel threshold — but it changes
    the on-screen text, so an OCR-driven decision always catches it. (The old
    pixel-diff gate is why typed content went uncaptured while OCR of the
    frames that *were* kept still listed text.)

    Flow per poll:
      1. Respect the app lock (skip when the locked app isn't frontmost).
      2. Cheap thumbnail pixel-diff — used ONLY to skip truly-static frames so
         we don't OCR the same idle screen forever. A periodic OCR refresh
         still fires on static screens to catch slow reveals.
      3. OCR the frame. If its text is ~identical to the last KEPT frame,
         discard it (and never write its JPEG). Otherwise keep it, recording
         the line-level text diff.
      4. Diagram/image frames (little text + large visual change) are kept as
         'thin_ocr' keyframes and get one Opus 5 visual description.
    If rec has a 'locked_app', only capture when that app is frontmost."""
    prev_thumb = None
    prev_kept_ocr = ""
    frame_count = 0
    tmp_png = os.path.join(rec["dir"], "_poll_tmp.png")

    locked_app = rec.get("locked_app", "")
    locked_bundle = rec.get("locked_bundle", "")
    last_ocr_time = 0.0

    while not stop_event.is_set():
        loop_start = time.time()
        try:
            # 1. If locked to an app, skip capture when that app isn't focused
            if locked_app or locked_bundle:
                cur_name, cur_bundle = _get_frontmost_app()
                is_match = False
                if locked_bundle and cur_bundle:
                    is_match = cur_bundle.lower() == locked_bundle.lower()
                elif locked_app:
                    is_match = cur_name.lower() == locked_app.lower()
                if not is_match:
                    elapsed = time.time() - loop_start
                    if elapsed < BURST_POLL_SEC:
                        stop_event.wait(BURST_POLL_SEC - elapsed)
                    continue

            display = display_under_cursor()
            r = subprocess.run(
                ['screencapture', '-x', '-D', str(display), tmp_png],
                capture_output=True, timeout=5
            )
            if r.returncode == 0 and os.path.exists(tmp_png):
                cur_thumb = _thumbnail_array(tmp_png)
                pixel_change = _frames_differ(prev_thumb, cur_thumb)

                # 2. Skip OCR only when the screen is essentially unchanged AND
                #    it isn't time for the periodic refresh. Any perceptible
                #    change (including a few typed characters) clears the gate.
                periodic_due = (time.time() - last_ocr_time) >= BURST_OCR_REFRESH_SEC
                if pixel_change <= BURST_STATIC_EPSILON and not periodic_due:
                    os.unlink(tmp_png) if os.path.exists(tmp_png) else None
                    elapsed = time.time() - loop_start
                    stop_event.wait(max(0.0, BURST_POLL_SEC - elapsed))
                    continue

                prev_thumb = cur_thumb
                last_ocr_time = time.time()

                # 3. OCR the full-res temp frame (better accuracy than the jpg)
                ocr_text = _vision_ocr(tmp_png)
                similarity = _ocr_similarity(prev_kept_ocr, ocr_text)
                is_duplicate = similarity >= BURST_OCR_DUP_SIMILARITY

                # 4. Diagram/image: little text extracted + big visual change
                thin_ocr = (len(ocr_text) < BURST_THIN_OCR_CHARS
                            and pixel_change > BURST_THIN_OCR_PIXEL)

                # Nothing new in the text and not a visual keyframe → drop it,
                # JPEG never written. This is the efficiency win: idle/duplicate
                # screens cost one OCR pass and no disk.
                if is_duplicate and not thin_ocr:
                    os.unlink(tmp_png) if os.path.exists(tmp_png) else None
                    elapsed = time.time() - loop_start
                    stop_event.wait(max(0.0, BURST_POLL_SEC - elapsed))
                    continue

                # KEEP: compress the frame to disk
                ts = datetime.now().strftime("%H:%M:%S")
                fname = f"frame_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                out_path = os.path.join(rec["dir"], fname)
                if _compress_burst_frame(tmp_png, out_path):
                    frame_count += 1
                    try:
                        metrics.record_storage(os.path.getsize(out_path), is_burst=True)
                    except OSError:
                        pass

                    text_diff = _ocr_diff(prev_kept_ocr, ocr_text)
                    frame = {
                        "timestamp": ts,
                        "path": out_path,
                        "ocr_text": ocr_text,
                        "text_diff": text_diff,
                        "duplicate": False,
                        "thin_ocr": thin_ocr,
                        "change_frac": pixel_change,
                    }

                    if thin_ocr:
                        desc = ask_claude_with_image(
                            "Describe this visual content in detail: layout, "
                            "connections, labels, and meaning.",
                            out_path, max_tokens=400
                        )
                        if desc and not desc.startswith("[Screenshot analysis failed"):
                            frame["visual_description"] = desc

                    with burst_lock:
                        rec["frames"].append(frame)
                    prev_kept_ocr = ocr_text

                if os.path.exists(tmp_png):
                    os.unlink(tmp_png)
        except Exception:
            pass

        # Sleep the remainder of the poll interval (responsive to stop_event)
        elapsed = time.time() - loop_start
        remaining = max(0.0, BURST_POLL_SEC - elapsed)
        stop_event.wait(remaining)

    rec["frame_count"] = frame_count


def start_burst(target_app_info=None):
    """Start burst recording. target_app_info is a (name, bundle_id) tuple from
    the hotkey callback. Only capture when that app is the frontmost window."""
    global burst_active, burst_stop_event, burst_thread, burst_session_count

    # Normalize: if target is a terminal, don't lock (user pressed local 'r')
    TERMINAL_BUNDLES = {"com.apple.terminal", "com.googlecode.iterm2", "net.kovidgoyal.kitty"}
    if target_app_info:
        name, bundle = target_app_info
        if bundle.lower() in TERMINAL_BUNDLES or name.lower() in ("terminal", "iterm2", "iterm"):
            target_app_info = None

    app_name = target_app_info[0] if target_app_info else ""
    app_bundle = target_app_info[1] if target_app_info else ""

    with burst_lock:
        if burst_active:
            return
        burst_session_count += 1
        n = burst_session_count
        rec_dir = _burst_dir(n)
        os.makedirs(rec_dir, exist_ok=True)
        rec = {
            "id": n,
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "end_time": None,
            "dir": rec_dir,
            "frames": [],
            "transcript_lines": [],
            "frame_count": 0,
            "_start_ts": time.time(),
            "locked_app": app_name,
            "locked_bundle": app_bundle,
        }
        burst_recordings.append(rec)
        burst_stop_event = threading.Event()
        burst_active = True
        burst_thread = threading.Thread(
            target=burst_poll_loop, args=(rec, burst_stop_event), daemon=True)
        burst_thread.start()

    app_note = f" → locked to \033[1m{app_name}\033[0m" if app_name else " (all screens)"
    out(f"\n  \033[1;31m🔴 RECORDING started{app_note} (press r/⌃⌥R to stop)\033[0m\n")


def stop_burst():
    global burst_active, burst_stop_event, burst_thread
    with burst_lock:
        if not burst_active:
            return
        rec = burst_recordings[-1]
        stop_event = burst_stop_event
        thread = burst_thread
        burst_active = False        # stop tagging new transcript lines / dot
    if stop_event:
        stop_event.set()
    out("\n  \033[33m⏹ Recording stopped — analyzing remaining frames...\033[0m")
    if thread:
        thread.join(timeout=120)
    with burst_lock:
        rec["end_time"] = datetime.now().strftime("%H:%M:%S")
        frames = rec.get("frame_count", len(rec["frames"]))
        duration = int(time.time() - rec.get("_start_ts", time.time()))
    with burst_lock:
        burst_stop_event = None
        burst_thread = None
    out(f"\n  \033[1;31m⏹ RECORDING stopped ({frames} frames captured in {duration} seconds)\033[0m")

    # Finalize immediately: write recording.json now so this recording's
    # content is available as context for later recordings and questions
    data = finalize_one_burst(rec)
    if data:
        n_states = len(data.get("screen_states", []))
        out(f"  \033[32m✓ recording.json saved ({n_states} screen states)\033[0m\n")
    else:
        out("")


def _detect_frontmost_for_hotkey():
    """Detect the frontmost app for the global hotkey context.
    Returns (name, bundle) or None."""
    result = _get_frontmost_app()
    if result and result[0]:
        return result
    return None


def _get_running_apps():
    """Return list of (name, bundle_id) for regular user apps (excludes terminals)."""
    try:
        from AppKit import NSWorkspace
        apps = NSWorkspace.sharedWorkspace().runningApplications()
        TERMINAL_BUNDLES = {"com.apple.terminal", "com.googlecode.iterm2", "net.kovidgoyal.kitty"}
        candidates = []
        for a in sorted(apps, key=lambda x: x.localizedName() or ""):
            if a.activationPolicy() == 0:
                bundle = a.bundleIdentifier() or ""
                if bundle.lower() not in TERMINAL_BUNDLES:
                    candidates.append((a.localizedName(), bundle))
        return candidates
    except Exception:
        return []


def _pick_app_inline():
    """Show a numbered app list and read the user's digit choice.
    MUST be called from the keyboard_listener thread (same stdin reader)."""
    candidates = _get_running_apps()
    if not candidates:
        out("  \033[33mCould not list apps — recording all screens\033[0m")
        return None

    out()
    out("  \033[1mRecord which app? (press number, or 0 for all):\033[0m")
    for i, (name, _) in enumerate(candidates[:9], 1):
        out(f"    \033[1m{i}\033[0m  {name}")
    out(f"    \033[1m0\033[0m  All screens (no lock)")
    out()

    # Read digit from same stdin (we're in the keyboard listener, already raw)
    ch = sys.stdin.read(1)
    if ch == '0' or not ch.isdigit():
        out("  → Recording all screens")
        return None
    idx = int(ch) - 1
    if 0 <= idx < len(candidates):
        name, bundle = candidates[idx]
        out(f"  → Locked to: {name}")
        return (name, bundle)
    out("  → Recording all screens")
    return None


def toggle_burst_recording(predetected_app_info=None):
    """Toggle burst on/off. predetected_app_info is a (name, bundle_id) tuple
    passed by the global hotkey callback."""
    if not FEATURES.get("burst_recording"):
        out("\n  \033[33mBurst recording is disabled in config.\033[0m\n")
        return
    if burst_active:
        stop_burst()
    else:
        start_burst(target_app_info=predetected_app_info)


def record_burst_transcript_line(timestamp, speaker, text):
    """Attach a transcript line to the currently-recording burst, if any."""
    if not burst_active:
        return
    with burst_lock:
        if burst_active and burst_recordings:
            burst_recordings[-1]["transcript_lines"].append((timestamp, speaker, text))


STATE_GROUP_SIMILARITY = RECORDING.get("state_group_similarity", 0.80)  # consecutive frames this similar → same screen state


def _build_screen_states(frames):
    """Group consecutive non-duplicate frames whose OCR text is >80% similar into
    'screen states'. Returns (screen_states, keep_files) where keep_files is the
    set of basenames whose JPEGs must survive finalize (thin_ocr key frames)."""
    screen_states = []
    keep_files = set()
    current = None

    def close(state):
        if state is None:
            return
        state.pop("_last_text", None)
        screen_states.append(state)

    for fr in frames:
        # Manual captures and duplicates are handled elsewhere / dropped
        if fr.get("type") == "manual_capture":
            continue
        if fr.get("duplicate"):
            continue

        ts = fr.get("timestamp", "")
        ocr_text = fr.get("ocr_text", "")
        is_thin = fr.get("thin_ocr", False)
        file_name = os.path.basename(fr.get("path", "")) if fr.get("path") else None

        # Thin-OCR (visual) frames always stand alone as their own state and keep the JPEG
        if is_thin:
            close(current)
            current = None
            if file_name:
                keep_files.add(file_name)
            state = {
                "start": ts,
                "end": ts,
                "frame_count": 1,
                "ocr_text": ocr_text,
                "thin_ocr": True,
                "key_frame": file_name,
                "changes": [],
                "_last_text": ocr_text,
            }
            if fr.get("visual_description"):
                state["visual_description"] = fr["visual_description"]
            close(state)
            current = None
            continue

        # Text frame: start a new state or extend the current one
        if current is None or _ocr_similarity(current["_last_text"], ocr_text) < STATE_GROUP_SIMILARITY:
            close(current)
            current = {
                "start": ts,
                "end": ts,
                "frame_count": 1,
                "ocr_text": ocr_text,
                "changes": [],
                "key_frame": None,
                "_last_text": ocr_text,
            }
        else:
            current["end"] = ts
            current["frame_count"] += 1
            # Keep the fullest OCR text as the representative text of the state
            if len(ocr_text) > len(current["ocr_text"]):
                current["ocr_text"] = ocr_text
            diff = (fr.get("text_diff") or "").strip()
            if diff:
                first_change = diff.splitlines()[0]
                current["changes"].append(f"{ts}: {first_change}")
            current["_last_text"] = ocr_text

    close(current)
    return screen_states, keep_files


def finalize_one_burst(rec):
    """Write recording.json for a single burst and clean up its JPEGs
    (keeping thin_ocr key frames + manual captures). Marks rec as finalized.
    Returns the recording data dict, or None on failure."""
    rec_dir = rec.get("dir")
    if not rec_dir or not os.path.isdir(rec_dir):
        return None
    if rec.get("_finalized"):
        # Already done — return the existing data
        try:
            with open(os.path.join(rec_dir, "recording.json"), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    frames = rec.get("frames", [])
    screen_states, keep_files = _build_screen_states(frames)

    manual_captures = []
    for fr in frames:
        if fr.get("type") == "manual_capture":
            fname = os.path.basename(fr.get("path", "")) if fr.get("path") else None
            if fname:
                keep_files.add(fname)
            manual_captures.append({
                "timestamp": fr.get("timestamp", ""),
                "file": fname,
                "description": fr.get("description", ""),
                "ocr_text": fr.get("ocr_text", ""),
            })

    data = {
        "id": rec["id"],
        "start_time": rec["start_time"],
        "end_time": rec.get("end_time"),
        "locked_app": rec.get("locked_app", ""),
        "locked_bundle": rec.get("locked_bundle", ""),
        "frame_count": rec.get("frame_count", len(frames)),
        "screen_states": screen_states,
        "manual_captures": manual_captures,
        "transcript_lines": [
            {"timestamp": ts, "speaker": sp, "text": tx}
            for ts, sp, tx in rec.get("transcript_lines", [])
        ],
    }
    try:
        with open(os.path.join(rec_dir, "recording.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        return None
    # Delete JPEGs/temp files EXCEPT thin_ocr key frames + manual captures
    for fname in os.listdir(rec_dir):
        if fname in keep_files:
            continue
        if fname.lower().endswith(('.jpg', '.png', '.bmp')):
            try:
                os.unlink(os.path.join(rec_dir, fname))
            except OSError:
                pass
    rec["_finalized"] = True
    rec["_recording_data"] = data
    return data


def finalize_bursts():
    """Session end: finalize any bursts not already finalized at stop time."""
    if not FEATURES.get("burst_recording"):
        return
    with burst_lock:
        recs = list(burst_recordings)
    for rec in recs:
        if not rec.get("_finalized"):
            finalize_one_burst(rec)


def get_all_recordings_context(max_chars=15000):
    """Combined context from all finalized recordings this session — used as
    context for later recordings, questions, and the post-meeting digest."""
    with burst_lock:
        recs = list(burst_recordings)
    parts = []
    total = 0
    for rec in recs:
        data = rec.get("_recording_data")
        if not data:
            continue
        header = (f"[Recording {data['id']}: {data['start_time']}–{data.get('end_time', '?')}"
                  f"{' | ' + data['locked_app'] if data.get('locked_app') else ''}]")
        state_texts = []
        for st in data.get("screen_states", []):
            txt = st.get("ocr_text", "")[:1500]
            vis = st.get("visual_description", "")
            entry = f"  {st.get('start')}–{st.get('end')}: {txt}"
            if vis:
                entry += f"\n  [VISUAL] {vis}"
            state_texts.append(entry)
        for mc in data.get("manual_captures", []):
            state_texts.append(f"  {mc.get('timestamp')} [MANUAL CAPTURE] {mc.get('description', '')[:800]}")
        block = header + "\n" + "\n".join(state_texts)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def cleanup_stale_bursts():
    """On startup: finalize any burst folders from previous sessions that still
    have JPEG files on disk (crashed/force-killed sessions that skipped cleanup)."""
    if not os.path.isdir(SCREENSHOT_DIR):
        return
    for name in os.listdir(SCREENSHOT_DIR):
        if not name.startswith("burst_"):
            continue
        burst_dir = os.path.join(SCREENSHOT_DIR, name)
        if not os.path.isdir(burst_dir):
            continue
        # If it already has recording.json but still has JPEGs, clean up
        has_json = os.path.exists(os.path.join(burst_dir, "recording.json"))
        jpgs = [f for f in os.listdir(burst_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        if jpgs:
            if has_json:
                # Already finalized — just leftover JPEGs
                for j in jpgs:
                    try:
                        os.unlink(os.path.join(burst_dir, j))
                    except OSError:
                        pass
            else:
                # Never finalized — write a minimal recording.json from filenames
                frames = [{"timestamp": f.split('_')[1][:6] if '_' in f else "",
                           "file": f, "description": "[unanalyzed — session crashed]"}
                          for f in sorted(jpgs)]
                data = {"id": 0, "start_time": "unknown", "end_time": "unknown",
                        "note": "recovered from crashed session",
                        "frame_count": len(jpgs), "frames": frames,
                        "transcript_lines": []}
                try:
                    with open(os.path.join(burst_dir, "recording.json"), 'w') as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
                for j in jpgs:
                    try:
                        os.unlink(os.path.join(burst_dir, j))
                    except OSError:
                        pass


# ── Transcript auto-save + post-meeting digest ─────────────────────────────────

def get_transcript_filepath():
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    date_str = session_start.strftime("%Y-%m-%d")
    time_str = session_start.strftime("%H%M")
    filename = f"{date_str}_{time_str}_meeting.txt"
    return os.path.join(TRANSCRIPT_DIR, filename)


def format_line(ts, speaker, text):
    tag = f"[{speaker}] " if speaker else ""
    return f"[{ts}] {tag}{text}"


def save_transcript():
    with transcript_lock:
        lines = transcript_lines[:]

    if not lines:
        return None

    filepath = get_transcript_filepath()
    duration = datetime.now() - session_start
    duration_str = f"{int(duration.total_seconds() // 60)}m {int(duration.total_seconds() % 60)}s"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Meeting Transcript\n")
        f.write(f"# Date: {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Duration: {duration_str}\n")
        f.write(f"# Lines: {len(lines)}\n")
        f.write(f"# Tool: Hankscribe 2.0\n")
        f.write(f"{'=' * 60}\n\n")
        for ts, sp, tx in lines:
            f.write(format_line(ts, sp, tx) + "\n")

    return filepath


def live_append_transcript(timestamp, speaker, text):
    filepath = get_transcript_filepath()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        if os.path.getsize(filepath) == 0:
            f.write(f"# Meeting Transcript — {session_start.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"{'=' * 60}\n\n")
        f.write(format_line(timestamp, speaker, text) + "\n")


def generate_digest(filepath):
    """Exhaustive post-meeting digest: full transcript + all recordings'
    screen content (OCR + visual descriptions) + manual screenshots,
    cross-referenced by timestamp. Uses Opus 5 (xhigh effort) for maximum accuracy."""
    with transcript_lock:
        lines = transcript_lines[:]
    if len(lines) < 8:
        return False

    full = "\n".join(format_line(ts, sp, tx) for ts, sp, tx in lines)[-25000:]

    # All screen content from recordings (OCR text, visual descriptions,
    # manual captures) — timestamped, so it interleaves with the transcript
    recordings_ctx = get_all_recordings_context(max_chars=20000)
    rec_block = (f"\n\nSCREEN RECORDINGS (timestamped OCR text + visual descriptions "
                 f"of what was on screen):\n{recordings_ctx}") if recordings_ctx else ""

    # Manual screenshots not tied to a recording
    screen_ctx = get_screenshot_context()
    manual_block = f"\n\nMANUAL SCREENSHOTS:\n{screen_ctx}" if screen_ctx else ""

    prompt = (
        f"The meeting has just ended. You have three timestamped sources:\n"
        f"1. The full audio transcript ([You] = {USER_NAME}, [Them] = others)\n"
        f"2. Screen recordings showing what was displayed during the meeting\n"
        f"3. Manual screenshots of key moments\n\n"
        f"TRANSCRIPT:\n{full}"
        f"{rec_block}"
        f"{manual_block}\n\n"
        f"Write an EXHAUSTIVE, precise post-meeting digest. Cross-reference the "
        f"screen content with what was being said at the same timestamps — e.g. if "
        f"a diagram was on screen while someone explained the architecture, connect "
        f"the two. Structure:\n"
        f"1. EXECUTIVE SUMMARY — one paragraph capturing the meeting's purpose and outcome\n"
        f"2. DETAILED CHRONOLOGY — walk through the meeting in order: what was "
        f"discussed, what was shown on screen, what was decided, with timestamps\n"
        f"3. KEY DECISIONS — every decision made, with who made it and the context\n"
        f"4. ACTION ITEMS — '- [ ] [Owner]: task (deadline)' — attribute owners "
        f"precisely; include items implied but not explicitly assigned (flag those)\n"
        f"5. SCREEN CONTENT HIGHLIGHTS — important information that appeared on "
        f"screen: numbers, IDs, dates, configurations, diagram contents\n"
        f"6. OPEN QUESTIONS — everything left unresolved\n"
        f"7. TOPICS FOR NEXT MEETING — deferred items\n\n"
        f"Be exhaustive and 100% faithful to the sources. Do not invent or assume "
        f"anything not present in the transcript or screen content. Use exact "
        f"names, numbers, and dates from the sources."
    )

    # Deep model for the digest — accuracy matters most here
    digest = ask_claude(prompt, max_tokens=3000,
                        system=build_system_prompt(),
                        model_override=AI_MODES["deep"])
    if not digest:
        # Fallback to a lighter Opus 5 pass if the deep xhigh call fails
        digest = ask_claude(prompt, max_tokens=1500,
                            system=build_system_prompt(),
                            model_override=AI_MODES["fast"])
    if not digest:
        # Last resort: cheap low-effort pass
        digest = ask_claude(prompt, max_tokens=1500,
                            system=build_system_prompt(),
                            model_override=UTILITY_MODE)
    if not digest:
        return False

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"# MEETING DIGEST (generated {datetime.now().strftime('%H:%M')})\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(digest + "\n")
    return True


# ── Keyboard listener ──────────────────────────────────────────────────────────

def read_question_pinned():
    with input_lock:
        input_state["active"] = True
        input_state["buffer"] = ""
        sys.stdout.write('\r\n' + ASK_PROMPT)
        sys.stdout.flush()

    try:
        while True:
            ch = sys.stdin.read(1)
            if ch in ('\r', '\n'):
                with input_lock:
                    question = input_state["buffer"].strip()
                    sys.stdout.write('\r\n')
                    sys.stdout.flush()
                return question
            elif ch == '\x1b':
                with input_lock:
                    sys.stdout.write('\r\x1b[K')
                    sys.stdout.flush()
                out("  \033[90m(cancelled)\033[0m")
                return ""
            elif ch == '\x03':
                os.kill(os.getpid(), signal.SIGINT)
                return ""
            elif ch in ('\x7f', '\x08'):
                with input_lock:
                    if input_state["buffer"]:
                        input_state["buffer"] = input_state["buffer"][:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
            elif ch.isprintable():
                with input_lock:
                    input_state["buffer"] += ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()
    finally:
        with input_lock:
            input_state["active"] = False
            input_state["buffer"] = ""


def global_hotkey_listener():
    try:
        from pynput import keyboard as pk
    except ImportError:
        out("  \033[33m⚠ pynput not installed — global hotkeys disabled\033[0m")
        return

    def make(handler):
        return lambda: threading.Thread(target=handler, daemon=True).start()

    try:
        from ApplicationServices import AXIsProcessTrusted
        if not AXIsProcessTrusted():
            out("  \033[33m⚠ Global hotkeys need Accessibility permission:\033[0m")
            out("    System Settings → Privacy & Security → Accessibility")
            out("    → add your terminal app (Terminal/iTerm), then restart Hankscribe")
            out("    (Terminal-focused keys 1/2/3/4/m still work fine)")
    except ImportError:
        pass

    try:
        hotkeys = pk.GlobalHotKeys({
            '<ctrl>+<alt>+1': make(handle_key_1),       # answer questions
            '<ctrl>+<alt>+3': make(handle_key_2),       # summary
            '<ctrl>+<alt>+4': make(handle_key_3),       # screenshot
            '<ctrl>+<alt>+r': lambda: threading.Thread(
                target=toggle_burst_recording,
                args=(_detect_frontmost_for_hotkey() or _last_non_terminal_app,),
                daemon=True).start(),
            '<ctrl>+<alt>+m': toggle_ai_mode,
        })
        hotkeys.start()
    except Exception as e:
        out(f"  \033[33m⚠ Global hotkeys unavailable: {e}\033[0m")


def keyboard_listener():
    global original_term
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return      # no interactive terminal (piped/pty-less) — hotkeys only
    original_term = old
    try:
        tty.setraw(fd)
        while running:
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                break   # stdin closed
            if not ch:
                break   # EOF
            if ch == '1':
                threading.Thread(target=handle_key_1, daemon=True).start()
            elif ch == '2':
                question = read_question_pinned()
                if question:
                    threading.Thread(target=answer_typed_question,
                                     args=(question,), daemon=True).start()
            elif ch == '3':
                threading.Thread(target=handle_key_2, daemon=True).start()
            elif ch == '4':
                threading.Thread(target=handle_key_3, daemon=True).start()
            elif ch == 'm':
                toggle_ai_mode()
            elif ch == 'r':
                if burst_active:
                    threading.Thread(target=toggle_burst_recording, daemon=True).start()
                else:
                    # Show app picker inline — reads the digit right here in
                    # the same stdin read loop (no thread race)
                    app_info = _pick_app_inline()
                    threading.Thread(target=toggle_burst_recording,
                                     args=(app_info,), daemon=True).start()
            elif ch == 's':
                out(f"\n  \033[1;33m📊 SESSION STATS:\033[0m {metrics.summary()}\n")
            elif ch in ('\x03', 'q'):
                os.kill(os.getpid(), signal.SIGINT)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global running, SAMPLE_RATE, segmenter

    out()
    out("  \033[1;36m━━━ HANKSCRIBE 2.0 ━━━\033[0m")
    out("  Real-time transcription + Claude-powered Q&A")
    out()

    cleanup_stale_bursts()

    if not start_whisper_server():
        out("  Falling back to whisper-cli (slower)...")

    device_id, channels, device_rate = find_audio_device()
    if device_rate and device_rate != SAMPLE_RATE:
        out(f"  \033[33m⚠ Device runs at {device_rate}Hz (config expected {SAMPLE_RATE}Hz) — adapting\033[0m")
        SAMPLE_RATE = device_rate
        # Rebuild the segmenter so its pause/preroll frame counts match the
        # real rate (they were computed from the old SAMPLE_RATE at import).
        segmenter = VadSegmenter()
    if FEATURES["speaker_attribution"] and channels <= MIC_CHANNEL:
        out(f"  \033[33m⚠ Speaker attribution off — device has {channels}ch, mic expected on ch{MIC_CHANNEL}\033[0m")

    if context_is_stale():
        out("  Context: project folder changed — refreshing index in background...")
        rebuild_context_async()
    else:
        out("  Context: project index up to date")
        build_embeddings_async()

    out("  AI: testing Bedrock connection...")
    test = ask_claude("Say OK", max_tokens=10)
    if test:
        out(f"  AI: \033[32mBedrock ready (prompt caching {'on' if FEATURES['prompt_caching'] else 'off'})\033[0m")
    else:
        out("  AI: \033[33mBedrock unavailable, using Ollama fallback\033[0m")

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    filepath = get_transcript_filepath()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Meeting Transcript — {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Tool: Hankscribe 2.0\n")
        f.write(f"{'=' * 60}\n\n")
    out(f"  Transcript: {filepath}")

    out()
    out("  \033[1mControls (terminal focused):\033[0m")
    out("    \033[1m1\033[0m → Answer questions from transcript (Claude + project context)")
    out("    \033[1m2\033[0m → Ask your own question (type it in)")
    out("    \033[1m3\033[0m → Auto-summary")
    out("    \033[1m4\033[0m → Screenshot for context (this session only)")
    out("    \033[1mr\033[0m → Burst recording: capture screen changes until pressed again")
    out("    \033[1mm\033[0m → Toggle AI mode: FAST (Opus 5 medium) ↔ DEEP (Opus 5 xhigh, most thorough)")
    out("    \033[1ms\033[0m → Show session stats (calls, cost, storage)")
    out("    \033[1mq\033[0m → Quit (auto-saves + digest)")
    out()
    out("  \033[1mGlobal hotkeys (work from Teams/Zoom/anywhere):\033[0m")
    out("    \033[1m⌃⌥1\033[0m answer   \033[1m⌃⌥3\033[0m summary   \033[1m⌃⌥4\033[0m screenshot   \033[1m⌃⌥R\033[0m record   \033[1m⌃⌥M\033[0m mode")
    out("    (needs Accessibility permission — see note above if shown)")
    out()
    out(f"  AI mode: {current_mode()['label']}")
    out()
    out("  \033[90mListening...\033[0m")
    out()

    def stop(sig, frame):
        global running, burst_active
        running = False
        # Stop any in-progress burst so its leftover frames get analyzed
        if burst_active and burst_stop_event is not None:
            burst_active = False
            burst_stop_event.set()
            if burst_thread is not None:
                burst_thread.join(timeout=60)
            with burst_lock:
                if burst_recordings and burst_recordings[-1].get("end_time") is None:
                    burst_recordings[-1]["end_time"] = datetime.now().strftime("%H:%M:%S")
        time.sleep(0.5)
        finalize_bursts()
        filepath = save_transcript()
        if filepath and FEATURES["post_meeting_digest"]:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
            except Exception:
                pass
            out()
            out("  Generating post-meeting digest...")
            if generate_digest(filepath):
                out("  \033[32m✓ Digest appended to transcript\033[0m")
        stop_whisper_server()
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
        except Exception:
            pass
        out()
        if filepath:
            out(f"  \033[32m✓ Transcript saved: {filepath}\033[0m")
        out(f"  \033[33m📊 {metrics.summary()}\033[0m")
        out("  Stopped.")
        sys.stdout.flush()
        # os._exit skips interpreter finalization — avoids the daemon-thread
        # stdout lock abort at shutdown
        os._exit(0)

    signal.signal(signal.SIGINT, stop)

    threading.Thread(target=transcription_worker, daemon=True).start()
    threading.Thread(target=keyboard_listener, daemon=True).start()
    threading.Thread(target=whisper_health_monitor, daemon=True).start()
    threading.Thread(target=title_bar_updater, daemon=True).start()
    global_hotkey_listener()

    try:
        with sd.InputStream(device=device_id, channels=channels, samplerate=SAMPLE_RATE,
                            dtype='float32', callback=audio_callback, blocksize=1024):
            while running:
                sd.sleep(100)
    except KeyboardInterrupt:
        stop(None, None)
    except Exception as e:
        out(f"  \033[1;31mError: {e}\033[0m")
        stop_whisper_server()
        sys.exit(1)


if __name__ == '__main__':
    main()
