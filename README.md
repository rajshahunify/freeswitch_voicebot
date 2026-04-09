# FreeSWITCH VoiceBot

An **automated IVR voicebot** that handles inbound telephone calls via [FreeSWITCH](https://freeswitch.com), transcribes caller speech in real-time, and navigates a JSON-defined conversation flow — playing pre-recorded audio responses for each step.

The system supports **multiple concurrent calls**, with per-call audio buffering, per-call VAD (Voice Activity Detection), and Redis-backed session management.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [End-to-End Call Flow (Traceback)](#end-to-end-call-flow-traceback)
- [Components](#components)
- [How mod\_audio\_fork Works](#how-mod_audio_fork-works)
- [The Custom Docker Image](#the-custom-docker-image)
- [IVR Flow Engine](#ivr-flow-engine)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
                         ┌─────────────────────────────────┐
                         │        SIP Phone / Caller        │
                         └──────────────┬──────────────────┘
                                        │ SIP INVITE (port 5060)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FreeSWITCH Docker Container                         │
│                                                                          │
│   ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────┐ │
│   │  SIP Stack   │───▶│  Call Routing     │───▶│  mod_audio_fork        │ │
│   │  (sofia-sip) │    │  (dialplan XML)   │    │  WebSocket audio fork  │ │
│   └─────────────┘    └──────────────────┘    └──────────┬─────────────┘ │
│   Port 5060                                              │               │
│   ┌─────────────┐                                        │               │
│   │  mod_event   │◀── ESL (port 8021) ──── agent.py      │               │
│   │  _socket     │                                       │               │
│   └─────────────┘    Sounds dir:                         │               │
│                      /usr/local/freeswitch/sounds/custom/ │               │
│                      ├── english_menu.wav                 │               │
│                      ├── thank_you.wav                    │               │
│                      ├── sorry.wav                        │               │
│                      └── ...                              │               │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                               WebSocket ws://host:8000/media
                               (16kHz mono PCM audio stream)
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Python VoiceBot Server (WSL/Linux)                      │
│                                                                          │
│   server_multicall.py (FastAPI + WebSocket)                               │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                    Per-Call Processing Pipeline                   │    │
│   │                                                                   │    │
│   │  ┌──────────┐   ┌─────────┐   ┌──────────────┐   ┌──────────┐  │    │
│   │  │ Raw Audio │──▶│ Silero  │──▶│ Audio Buffer │──▶│ DeepFilter│  │    │
│   │  │ Chunks   │   │ VAD     │   │ (accumulate) │   │ Net2 (NC) │  │    │
│   │  │ (32ms)   │   │ per-call│   │ per-call     │   │ utterance │  │    │
│   │  └──────────┘   └─────────┘   └──────────────┘   └─────┬────┘  │    │
│   │                                                          │       │    │
│   │  ┌──────────────────────────────────────────────────────┘       │    │
│   │  │                                                               │    │
│   │  ▼                                                               │    │
│   │  ┌──────────┐   ┌─────────────┐   ┌───────────────────────┐    │    │
│   │  │ STT API  │──▶│ Flow Engine │──▶│ Response Handler      │    │    │
│   │  │ (Whisper)│   │ (JSON IVR)  │   │ (uuid_broadcast via   │    │    │
│   │  │ HTTP POST│   │ fuzzy+semantic│  │  fs_cli)              │    │    │
│   │  └──────────┘   └─────────────┘   └───────────────────────┘    │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   ┌────────────┐                                                         │
│   │   Redis    │  (session state, locks, flow_state per call)            │
│   └────────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Call Flow (Traceback)

This is the exact sequence of events when a phone call comes in:

### Phase 1: Call Arrival

1. **SIP Phone** sends a SIP `INVITE` to FreeSWITCH on port `5060/udp`.
2. **FreeSWITCH** (running inside Docker) receives the call via `mod_sofia` (the SIP stack built on top of `sofia-sip`).
3. The FreeSWITCH **dialplan** (XML configuration in `conf/dialplan/`) routes the call. The default dialplan parks the call, triggering a `CHANNEL_PARK` event.

### Phase 2: Agent Picks Up

4. **`agent.py`** is connected to FreeSWITCH via **ESL** (Event Socket Library) on port `8021`. It runs an event loop using the `greenswitch` library.
5. When `agent.py` receives the `CHANNEL_PARK` event, it:
   - **Answers the call**: `api uuid_answer <uuid>`
   - **Forks the audio** to the Python WebSocket server: `api uuid_audio_fork <uuid> start ws://127.0.0.1:8000/media mono 16k`
   - **Plays a welcome message**: `api uuid_broadcast <uuid> <welcome.wav>`

### Phase 3: Audio Streaming

6. **`mod_audio_fork`** inside FreeSWITCH opens a WebSocket connection to `ws://127.0.0.1:8000/media`.
7. Audio from the caller's microphone (the "a-leg") is streamed in real-time as **binary WebSocket frames** — raw PCM, 16kHz, 16-bit, mono, in 32ms chunks (512 samples / 1024 bytes per chunk).
8. **`server_multicall.py`** accepts the WebSocket connection, looks up the active call UUID via `fs_cli show channels as json`, creates a Redis session, and initializes per-call state.

### Phase 4: Audio Processing Pipeline

For each 32ms audio chunk received:

9. **VAD Detection** (Silero VAD, per-call instance via `PerCallVADManager`):
   - Determines if the chunk contains speech (probability > 0.3 threshold).
   - Tracks speech start/end transitions based on minimum speech duration (200ms) and minimum silence duration (1500ms).

10. **Audio Buffering** (per-call `AudioBuffer`):
    - During speech: accumulates raw PCM chunks.
    - On `speech_end` event: releases the complete utterance as a single byte blob.
    - Safety: enforces minimum length (18KB ≈ 0.56s) and maximum length (320KB ≈ 10s).
    - Timeout: forces release after 10 seconds even without speech_end.

11. When a complete utterance is ready, it's sent to a **ThreadPoolExecutor** for processing:

### Phase 5: Utterance Processing (in thread pool)

12. **Noise Cancellation** (DeepFilterNet2, `ImprovedNoiseCanceller`):
    - Resamples to DeepFilter's native sample rate (48kHz).
    - Runs the full utterance through the neural network.
    - Normalizes output with configurable gain (4x by default).
    - Resamples back to 16kHz for STT.

13. **Speech-to-Text** (`STTHandler`):
    - HTTP POST to external Whisper API at `http://164.52.203.140:8890/transcribe`.
    - Sends raw PCM bytes with parameters: `sample_rate=16000, bit_depth=int16, language=en`.
    - Receives JSON response: `{"text": "I want to check my payment"}`.

14. **IVR Flow Engine** (`FlowEngine`):
    - Loads the current call's flow state from Redis (which step the caller is on).
    - Matches the transcribed text against the current step's choices using a hybrid strategy:
      - **Pass 1 — Fuzzy matching** (`fuzzywuzzy`): Fast string similarity (< 5ms). Includes built-in synonym expansion for yes/no variants.
      - **Pass 2 — Semantic matching** (`sentence-transformers`): Cosine similarity with `all-MiniLM-L6-v2` embeddings. Catches paraphrases that keywords miss.
    - Determines the next step in the JSON flow tree.
    - Returns: `(prompt_text, audio_filename, should_end, is_fallback)`.

15. **Response Playback** (`ResponseHandler`):
    - Plays the matched audio file via: `fs_cli -x "uuid_broadcast <uuid> /usr/local/freeswitch/sounds/custom/<filename> aleg"`
    - Sets a speaking lock (prevents processing user audio while bot is talking).
    - Gets audio duration via `ffprobe` and releases the lock after playback completes.

### Phase 6: Call End

16. When the WebSocket disconnects (caller hangs up):
    - Session lock is released in Redis.
    - Session is ended in Redis.
    - Per-call VAD instance is removed from `PerCallVADManager`.
    - Per-call audio buffer is removed from `CallAudioManager`.
    - Response handler state is cleaned up.

---

## Components

| Component | File | Description |
|---|---|---|
| **ESL Agent** | `agent.py` | Connects to FreeSWITCH ESL, answers calls, forks audio to WebSocket |
| **WebSocket Server** | `server_multicall.py` | FastAPI server, receives audio stream, runs processing pipeline |
| **Config** | `config.py` | Centralized settings with environment variable overrides |
| **Noise Canceller** | `audio_pipeline/improved_noise_canceller.py` | DeepFilterNet2 wrapper for full-utterance denoising |
| **VAD Manager** | `audio_pipeline/vad_detector.py` | Silero VAD with per-call state isolation (`PerCallVADManager`) |
| **Audio Buffer** | `audio_pipeline/audio_buffer.py` | Per-call audio accumulation with speech boundary detection |
| **Flow Engine** | `ivr/json_flow_engine.py` | JSON-driven IVR navigation with hybrid fuzzy+semantic matching |
| **Response Handler** | `ivr/response_handler.py` | Audio playback via `fs_cli uuid_broadcast` |
| **STT Handler** | `stt_handler.py` | HTTP client for Whisper STT API |
| **Session Manager** | `session_manager.py` | Redis-backed session state and locking |

---

## How mod_audio_fork Works

`mod_audio_fork` is a FreeSWITCH module that **forks (copies) the audio stream** from an active call and sends it over a **WebSocket connection** to an external server — in our case, the Python voicebot.

### What It Does

1. When activated via `uuid_audio_fork <uuid> start <ws_url> mono 16k`, the module:
   - Opens a WebSocket connection to the specified URL.
   - Hooks into FreeSWITCH's audio processing chain.
   - For each audio frame (32ms at 16kHz = 512 samples = 1024 bytes), sends the raw PCM data as a binary WebSocket message.

2. The audio is a **copy** — the original call audio continues to flow normally, so the caller can still hear audio played back to them via `uuid_broadcast`.

3. When the call ends, the WebSocket connection is automatically closed.

### Why a Custom Build?

The `mod_audio_fork` module is **not included** in the standard FreeSWITCH distribution. It was originally created by [drachtio](https://github.com/drachtio) for their real-time speech processing use case. Building it requires:

- FreeSWITCH source code (for headers).
- `libwebsockets` (specifically v3.2-stable, for WebSocket client support).
- Patches to resolve API incompatibilities between the module's code and the specific libwebsockets version.

---

## The Custom Docker Image

**Image**: `rajunify123/freeswitch-mod-audio-fork`

This is a **multi-stage Docker build** (see `Dockerfile` in `fs_new_docker/`):

### Stage 1: Builder (debian:11)

1. **Build dependencies**: gcc, g++, cmake, autoconf, pkg-config, etc.
2. **libwebsockets v3.2**: WebSocket client library (pinned to v3.2-stable for compatibility).
3. **Telephony libraries**: libks, signalwire-c, sofia-sip (SIP stack), spandsp (fax/modem).
4. **FreeSWITCH v1.10**: Built from source with `--prefix=/usr/local/freeswitch`.
   - `make samples` generates default configuration.
   - `mod_python3` enabled in `modules.conf`.
   - ESL configured to listen on `0.0.0.0:8021` with password `ClueCon`.
5. **mod_audio_fork**: Compiled separately from drachtio source with two patches:
   - **Patch 1** (`audio_pipe.cpp`): Removes the `lws_retry_bo_t` struct which doesn't exist in libwebsockets v3.2.
   - **Patch 2** (`parser.cpp`): Removes duplicate `parse_ws_uri` function (already defined in `lws_glue.cpp`).
   - Compiled as 4 object files → linked into `mod_audio_fork.so`.
   - Verified: must be > 500KB with > 10 AudioPipe symbols.

### Stage 2: Runtime (debian:11-slim)

- Copies only built artifacts (no build tools).
- Installs minimal runtime libraries.
- Exposes ports: `5060` (SIP), `8021` (ESL), `16384-16484` (RTP media).
- Healthcheck: verifies FreeSWITCH PID is alive.

### Ports

| Port | Protocol | Purpose |
|---|---|---|
| 5060 | UDP/TCP | SIP signaling |
| 5080 | UDP/TCP | SIP (external profile) |
| 5061 | UDP/TCP | SIP TLS |
| 8021 | TCP | Event Socket (ESL) — used by `agent.py` |
| 16384-16484 | UDP | RTP media (audio packets) |

---

## IVR Flow Engine

The IVR flow is defined in JSON files under `ivr/json_files/`. Each language has its own file (e.g., `en.json`).

### Flow Structure

```json
{
  "start": "en_flow_start",
  "steps": {
    "en_flow_start": {
      "prompt": "Your preferred language is english. Say 'yes' or 'no'",
      "audio": "en_flow_start.wav",
      "type": "choice",
      "choices": {
        "yes": "english_menu",
        "no": "language_select"
      }
    },
    "english_menu": {
      "prompt": "We offer the following services...",
      "audio": "english_menu.wav",
      "type": "choice",
      "choices": {
        "subscribe": "subscribe_fiber",
        "billing": "billing_info",
        "payment": "payment_options",
        ...
      }
    }
  }
}
```

### Step Types

| Type | Behavior |
|---|---|
| `choice` | Matches user speech against `choices` keys using fuzzy + semantic matching. On match, transitions to the mapped next step. |
| `input` | Accepts any speech (e.g., account number) and moves to `next`. |
| `action` | Has an `action` field (e.g., `transfer_agent`, `send_sms`). Currently stubbed — logs the action and moves to `next`. |
| `end` | Terminal step. Indicates the call flow is complete. |
| (auto-advance) | No `type` but has `next` — automatically transitions on any input. |

### Matching Strategy (Hybrid)

```
User says: "I want to pay my bill"

Pass 1 — Fuzzy (fuzzywuzzy):
  Compares against choice keys: ["subscribe", "billing", "payment", ...]
  "payment" → score 70 (below threshold 75) → MISS

Pass 2 — Semantic (sentence-transformers):
  Encodes "I want to pay my bill" with all-MiniLM-L6-v2
  Compares cosine similarity vs each choice key embedding
  "payment" → similarity 0.62 (above threshold 0.45) → HIT ✓

Result: Navigate to "payment_options" step, play payment_options.wav
```

---

## Audio Processing Pipeline

```
Raw 32ms chunk (1024 bytes @ 16kHz/16-bit/mono)
        │
        ▼
┌─────────────────────────────────────────┐
│  Silero VAD (per-call isolated state)   │
│  Determines: speech / silence / start / │
│  end transitions                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  AudioBuffer (per-call)                  │
│  Accumulates during speech, releases     │
│  complete utterance on speech_end        │
│  Min: 18KB (~0.56s) Max: 320KB (~10s)   │
└────────────────┬────────────────────────┘
                 │ Complete utterance (bytes)
                 ▼
┌─────────────────────────────────────────┐
│  DeepFilterNet2 (shared model)          │
│  Full-utterance noise cancellation       │
│  16kHz → 48kHz → denoise → 48kHz → 16kHz│
│  Normalize with 4x gain                 │
└────────────────┬────────────────────────┘
                 │ Clean audio
                 ▼
┌─────────────────────────────────────────┐
│  STT (HTTP POST to Whisper API)         │
│  Returns transcribed text               │
└────────────────┬────────────────────────┘
                 │ "I want to check payment"
                 ▼
┌─────────────────────────────────────────┐
│  FlowEngine.process_input()             │
│  Fuzzy match → Semantic fallback        │
│  Returns (text, audio_file, end, retry) │
└────────────────┬────────────────────────┘
                 │ ("payment_options.wav")
                 ▼
┌─────────────────────────────────────────┐
│  ResponseHandler.play_audio()           │
│  fs_cli → uuid_broadcast → FreeSWITCH  │
│  → Caller hears the response audio      │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
freeswitch_voicebot/
├── server_multicall.py          # Main WebSocket server (FastAPI)
├── agent.py                     # FreeSWITCH ESL agent
├── config.py                    # Centralized configuration
├── stt_handler.py               # Speech-to-text HTTP client
├── session_manager.py           # Redis-backed session management
├── requirements.txt             # Python dependencies
│
├── audio_pipeline/              # Audio processing modules
│   ├── __init__.py
│   ├── improved_noise_canceller.py  # DeepFilterNet2 wrapper
│   ├── vad_detector.py              # Silero VAD + PerCallVADManager
│   └── audio_buffer.py             # Per-call audio buffering
│
├── ivr/                         # IVR logic
│   ├── __init__.py
│   ├── json_flow_engine.py          # JSON-based flow navigation
│   ├── intent_matcher.py           # Legacy keyword matcher (kept for reference)
│   ├── response_handler.py         # Audio playback via fs_cli
│   └── json_files/                 # Flow definitions
│       └── en.json                 # English IVR flow
│
├── sounds/                      # Pre-recorded audio files
│   ├── english_menu.wav
│   ├── payment_options.wav
│   ├── sorry.wav
│   ├── thank_you.wav
│   └── ... (20 files)
│
├── docker-compose.yml           # Docker orchestration
├── logs/                        # Log output directory
├── debug_audio/                 # NC debug audio (before/after)
└── models/                      # Model cache directory
```

---

## Prerequisites

| Requirement | Purpose |
|---|---|
| **Docker** | Run the FreeSWITCH container |
| **WSL2 / Linux** | Python server (DeepFilterNet requires Linux, `fs_cli` is a Linux binary) |
| **Python 3.9+** | Server runtime |
| **Redis** | Session state management |
| **Network access** | STT API at `164.52.203.140:8890` must be reachable |

---

## Installation

### 1. Pull the FreeSWITCH Docker Image

```bash
docker pull rajunify123/freeswitch-mod-audio-fork
```

### 2. Copy Audio Files into the Container

The `sounds/` directory in this repo contains the pre-recorded IVR audio files. These need to be available inside the FreeSWITCH container at `/usr/local/freeswitch/sounds/custom/`:

```bash
# Start the container first (see step 4), then copy sounds in:
docker cp sounds/. voicebot-fs:/usr/local/freeswitch/sounds/custom/
```

Or mount as a volume in docker-compose:
```yaml
volumes:
  - ./sounds:/usr/local/freeswitch/sounds/custom
```

### 3. Start Redis

```bash
docker run -d --name voicebot-redis -p 6379:6379 redis:7-alpine
```

### 4. Start FreeSWITCH

```bash
# Using host network (recommended — standard ports):
docker run -d \
  --name voicebot-fs \
  --network host \
  rajunify123/freeswitch-mod-audio-fork
```

### 5. Verify FreeSWITCH

```bash
# Check FreeSWITCH is running
fs_cli -x "status"

# Verify mod_audio_fork is loaded
fs_cli -x "module_exists mod_audio_fork"
# Expected output: true
```

### 6. Install Python Dependencies (in WSL)

```bash
cd /mnt/c/Users/unify/freeswitch_voicebot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note**: `deepfilternet` and `torch` are large downloads (~2GB). The `sentence-transformers` model (`all-MiniLM-L6-v2`, ~90MB) is downloaded on first use.

---

## Running the System

### Start Order

You must start services in this order:

```
1. Redis          (session state backend)
2. FreeSWITCH     (telephony engine)
3. server_multicall.py   (audio processor + IVR)
4. agent.py       (call handler)
```

### Terminal 1: WebSocket Server

```bash
cd /mnt/c/Users/unify/freeswitch_voicebot
source venv/bin/activate
python3 server_multicall.py
```

The server starts on `0.0.0.0:8000` and listens for WebSocket connections at `/media`.

### Terminal 2: ESL Agent

```bash
cd /mnt/c/Users/unify/freeswitch_voicebot
source venv/bin/activate
python3 agent.py
```

The agent connects to FreeSWITCH ESL on `127.0.0.1:8021` and waits for incoming calls.

### Make a Test Call

Configure a SIP softphone (Zoiper, Linphone, MicroSIP) to register with FreeSWITCH:
- **SIP Server**: `<your-machine-ip>:5060`
- **Username**: `1000` (default FreeSWITCH user)
- **Password**: `1234` (default)

Dial any extension to trigger the voicebot.

---

## Configuration Reference

All settings are in `config.py`. Key settings support environment variable overrides:

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `FREESWITCH_HOST` | `FREESWITCH_HOST` | `127.0.0.1` | FreeSWITCH ESL host |
| `FREESWITCH_PORT` | `FREESWITCH_PORT` | `8021` | FreeSWITCH ESL port |
| `WS_PORT` | `WS_PORT` | `8000` | WebSocket server port |
| `STT_URL` | `STT_URL` | `http://164.52.203.140:8890/transcribe` | STT API endpoint |
| `REDIS_HOST` | `REDIS_HOST` | `127.0.0.1` | Redis host |
| `MAX_CONCURRENT_CALLS` | `MAX_CONCURRENT_CALLS` | `5` | Max simultaneous calls |
| `DF_USE_GPU` | `DF_USE_GPU` | `False` | Enable CUDA for DeepFilterNet |
| `LOG_LEVEL` | `LOG_LEVEL` | `INFO` | Logging verbosity |

### Audio Processing Tuning

| Setting | Default | Description |
|---|---|---|
| `VAD_THRESHOLD` | `0.3` | Speech probability threshold (lower = more sensitive) |
| `VAD_MIN_SILENCE_DURATION_MS` | `1500` | How long to wait after speech stops before processing |
| `DF_GAIN` | `4.0` | Post-NC volume normalization gain |
| `DF_ATTENUATION_LIMIT` | `6.0` | Max noise reduction in dB |
| `FUZZY_MATCH_THRESHOLD` | `75` | Minimum fuzzywuzzy score to accept a match |
| `SEMANTIC_MATCH_THRESHOLD` | `0.45` | Minimum cosine similarity for semantic matching |

---

## Troubleshooting

### "Could not find active call UUID"

The server couldn't query FreeSWITCH for active channels. Check:
- Is `fs_cli` accessible from the Python process?
- Is FreeSWITCH running? (`docker ps`)

### Audio plays but bot doesn't respond

- Check if the STT API is reachable: `curl http://164.52.203.140:8890/`
- Check `logs/voicebot.log` for STT timeout errors.
- Enable debug audio: set `DF_DEBUG_SAVE_DIR = "debug_audio"` in config and inspect the saved WAV files.

### VAD never detects speech end

- Lower `VAD_MIN_SILENCE_DURATION_MS` (default 1500ms may be too long for fast speakers).
- Lower `VAD_THRESHOLD` (default 0.3).

### "mod_audio_fork" not loaded

```bash
fs_cli -x "load mod_audio_fork"
# If it fails, check: fs_cli -x "module_exists mod_audio_fork"
```

### Redis connection refused

```bash
# Check Redis is running
docker ps | grep redis
# Or start it
docker run -d --name voicebot-redis -p 6379:6379 redis:7-alpine
```

---

## API Endpoints

The WebSocket server exposes REST endpoints for monitoring:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Component health check |
| `/stats` | GET | Detailed performance statistics |
| `/sessions` | GET | List all active call sessions |
| `/media` | WebSocket | Audio streaming endpoint (used by mod_audio_fork) |
