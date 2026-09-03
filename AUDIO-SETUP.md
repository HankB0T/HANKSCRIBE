# Audio Setup — one time per Mac

Hankscribe hears two things: **your mic** and **the meeting's audio**. macOS
never gives an app the meeting audio directly, so it is routed through a free
virtual device (**BlackHole**). This takes ~5 minutes on a new Mac and never
needs touching again — *except* step 6, which every meeting app can silently
undo (that's the trap that produces empty transcripts).

> **Shortcut:** run `./SETTINGS.command` — the "Audio setup — new-Mac
> checklist" section scans your devices and shows ✓/✗ for steps 1–4 live.

---

## Step 1 — Install BlackHole

```bash
brew install blackhole-2ch
```

Log out and back in (or reboot) so macOS registers the device.

## Step 2 — Create the Aggregate Device (what Hankscribe listens to)

1. Open **Audio MIDI Setup** (Cmd-Space, type "Audio MIDI").
2. Click **+** (bottom-left) → **Create Aggregate Device**.
3. Tick, **in this order**:
   - ☑ **BlackHole 2ch** ← first (becomes channels 0–1 = the meeting)
   - ☑ **MacBook Pro Microphone** (or your headset mic) ← second (becomes channel 2 = you)
4. On the microphone's row, tick **Drift Correction**.
5. Leave the name as **"Aggregate Device"** (Hankscribe finds it by that name;
   if you rename it, add the new name to `audio.preferred_devices` in Settings).

> Ticked the mic first by mistake? Either re-order, or set **Mic channel = 0**
> in Settings → Audio. A wrong mic channel silently swaps [You] and [Them].

## Step 3 — Create the Multi-Output Device (so you still hear the meeting)

1. In Audio MIDI Setup: **+** → **Create Multi-Output Device**.
2. Tick **both**:
   - ☑ your headphones / speakers (what you listen with)
   - ☑ **BlackHole 2ch** (the copy Hankscribe transcribes)
3. Leave the name **"Multi-Output Device"**.

## Step 4 — System sound output

**System Settings → Sound → Output → Multi-Output Device.**

## Step 5 — Meeting app microphone

Teams/Zoom → Settings → Audio/Devices → **Microphone = your real mic**
(headset or built-in). Never BlackHole.

## Step 6 — Meeting app speaker ⚠ THE #1 TRAP

Teams/Zoom → Settings → Audio/Devices → **Speaker = "Multi-Output Device"**
(or **"Same as System"**).

This is the step that breaks silently: plug in a headset (Jabra, AirPods, …)
and Teams/Zoom helpfully switches its speaker straight to the headset —
bypassing BlackHole. You hear everything, Hankscribe hears **nothing** from
the meeting, and the transcript shows only [You] lines. **Re-check this
setting whenever you join with a different headset.** Hankscribe warns you
live ("No meeting audio has reached BlackHole") if this happens.

## Step 7 — Verify

Start Hankscribe, join any meeting (or play a video), and press **`n`**:

- it measures every audio channel for 3 seconds and prints the levels;
- meeting channels silent → step 6 (or 3/4) is wrong;
- mic channel silent while you speak → the mic channel number is wrong
  (Settings → Audio → Mic channel).

---

## Swapping headphones later

- **Listening side**: any headphones work — just keep the Multi-Output Device
  selected as the meeting app's speaker (step 6).
- **Mic side**: if your Aggregate Device uses the built-in Mac mic, nothing
  ever changes. If you add a Bluetooth headset's mic instead, expect lower
  audio quality (Bluetooth HFP mode) — the built-in mic usually transcribes
  better.
- Sample-rate differences (44.1 vs 48 kHz) are handled automatically.

## Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Only [You] lines, no [Them] | Meeting audio bypassing BlackHole | Step 6, then steps 3–4 |
| Only [Them] lines / [You] and [Them] swapped | Wrong mic channel | Settings → Audio → Mic channel (press `n` to confirm) |
| No lines at all | Wrong input device / Whisper down | Settings checklist; check startup output |
| You can't hear the meeting | Multi-Output missing your headphones | Step 3 |
| Garbled/chipmunk transcription | (Fixed in code — auto-resampled) | Update Hankscribe |
