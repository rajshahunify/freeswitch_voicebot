# FreeSWITCH VoiceBot

[![Docker Pulls](https://img.shields.io/docker/pulls/rajunify123/freeswitch-voicebot?style=flat-square&logo=docker)](https://hub.docker.com/r/rajunify123/freeswitch-voicebot)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github&style=flat-square)](https://github.com/rajshahunify/freeswitch_voicebot)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square)](#)

An **automated IVR voicebot** that handles inbound telephone calls via [FreeSWITCH](https://freeswitch.com), transcribes caller speech in real-time, and navigates a JSON-defined conversation flow — playing pre-recorded audio responses for each step.

The system supports **multiple concurrent calls**, with per-call audio buffering, per-call VAD (Voice Activity Detection), and Redis-backed session management.

> [!TIP]
> **⚡ Want to get started quickly?** Check out the **[5-Minute Quick Start Guide](QUICKSTART.md)** for a step-by-step walkthrough.
>
> **🐳 Docker Hub Images:**
> - [`rajunify123/freeswitch-voicebot`](https://hub.docker.com/r/rajunify123/freeswitch-voicebot) — All-in-one VoiceBot container
> - [`rajunify123/freeswitch-mod-audio-fork`](https://hub.docker.com/r/rajunify123/freeswitch-mod-audio-fork) — Base FreeSWITCH image with mod_audio_fork

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
- [Installation & Running](#installation--running)
- [Making a Test Call (Zoiper Setup)](#-making-a-test-call-zoiper-setup)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [API Endpoints](#api-endpoints)

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
│   └─────────────┘                                        │               │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                               WebSocket ws://host:8000/media
                               (16kHz mono PCM audio stream)
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Python VoiceBot Server (Docker)                         │
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
│   │  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────┐  │    │
│   │  │ STT API  │──▶│ LLM Agent   │──▶│ Sentence     │──▶│ TTS      │  │    │
│   │  │ (Whisper)│   │ (Gemini/    │   │ Queue        │   │ Synthes- │  │    │
│   │  │ HTTP POST│   │ Groq/Ollama)│   │ (Streaming)  │   │ izer     │  │    │
│   │  └──────────┘   └─────────────┘   └──────────────┘   └─────┬────┘  │    │
│   │                                                            │       │    │
│   │                                                            ▼       │    │
│   │                                                   ┌──────────────┐ │    │
│   │                                                   │ Response     │ │    │
│   │                                                   │ Handler      │ │    │
│   │                                                   │ (fs_cli)     │ │    │
│   │                                                   └──────────────┘ │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   ┌────────────┐                                                         │
│   │   Redis    │  (session state, locks, conversation history, TTS cache)│
│   └────────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Call Flow (Traceback)

This is the exact sequence of events when a phone call comes in:

### Phase 1: Call Arrival

1. **SIP Phone** sends a SIP `INVITE` to FreeSWITCH on port `5060/udp`.
2. **FreeSWITCH** (running inside Docker) receives the call via `mod_sofia` (the SIP stack built on top of `sofia-sip`).
3. The FreeSWITCH **dialplan** (XML configuration in `conf/dialplan/`) routes the call. Call is immediately picked up (instant connection, no 10-second delay).

### Phase 2: Agent Picks Up

4. **`agent.py`** is connected to FreeSWITCH via **ESL** (Event Socket Library) on port `8021`. It runs an event loop using the `greenswitch` library.
5. When `agent.py` receives the `CHANNEL_PARK` event, it:
   - **Answers the call**: `api uuid_answer <uuid>`
   - **Forks the audio** to the Python WebSocket server: `api uuid_audio_fork <uuid> start ws://127.0.0.1:8000/media mono 16k`
   - Plays a greeting synthesized by TTS: `Hello! Welcome to Unified Reach Fiber. How can I help you today?`

### Phase 3: Audio Streaming

6. **`mod_audio_fork`** inside FreeSWITCH opens a WebSocket connection to `ws://127.0.0.1:8000/media`.
7. Audio from the caller's microphone (the "a-leg") is streamed in real-time as **binary WebSocket frames** — raw PCM, 16kHz, 16-bit, mono, in 32ms chunks (512 samples / 1024 bytes per chunk).
8. **`server_multicall.py`** accepts the WebSocket connection, looks up the active call UUID via `fs_cli show channels as json`, creates a Redis session, and initializes per-call state.

### Phase 4: Audio Processing Pipeline

For each 32ms audio chunk received:

9. **VAD Detection** (Silero VAD, per-call instance via `PerCallVADManager`):
   - Determines if the chunk contains speech (probability > 0.5 threshold).
   - Tracks speech start/end transitions based on minimum speech duration (200ms) and conversational silence duration (800ms).

10. **Audio Buffering** (per-call `AudioBuffer`):
    - During speech: accumulates raw PCM chunks.
    - On `speech_end` event: releases the complete utterance as a single byte blob.
    - Safety: enforces minimum length (18KB ≈ 0.3s) and maximum length (320KB ≈ 10s).
    - Timeout: forces release after 10 seconds even without speech_end.

11. When a complete utterance is ready, it's sent to a **ThreadPoolExecutor** for processing:

### Phase 5: Utterance Processing (in thread pool)

12. **Noise Cancellation** (DeepFilterNet2, `ImprovedNoiseCanceller` - if enabled):
    - Resamples to DeepFilter's native sample rate (48kHz).
    - Runs the full utterance through the neural network.
    - Normalizes output with configurable gain (4x by default).
    - Resamples back to 16kHz for STT.

13. **Speech-to-Text** (`STTHandler`):
    - HTTP POST to external Whisper API (or local `faster-whisper` model if configured).
    - Sends raw PCM bytes.
    - Receives transcribed text.

14. **Conversational LLM Agent** (`BaseLLMProvider` / `FallbackLLM`):
    - Passes the text + conversation history to the primary LLM provider (Gemini).
    - If Gemini fails (e.g. rate limit, daily quota limit), it automatically switches to the fallback provider (Groq or Ollama).
    - A background health check loop periodically pings Gemini to test its recovery, restoring Gemini as primary once available.
    - Streams tokens from the LLM and parses them into complete sentences.

15. **Sentence-Level TTS Synthesis** (`TTSSynthesizer`):
    - Submits sentences in real-time to a thread pool for synthesis via `edge-tts`.
    - MP3 data returned from Edge-TTS is converted to 16kHz mono WAV using `ffmpeg` (taking ~50ms) with a fallback to `torchaudio`.
    - Synthesized phrases are cached in Redis and local disk to avoid synthesis latency on repeat prompts.

16. **Response Playback & Queueing** (`ResponseHandler`):
    - Plays synthesized WAV files sequentially using `fs_cli uuid_broadcast`.
    - Enforces a speaking lock to prevent processing user audio while bot is speaking.
    - **Barge-In (User Interruption)**: If the user interrupts, the VAD detects speech, sends a `uuid_break <uuid> all` command to stop the bot playback, cancels pending TTS syntheses, and starts processing the new speech immediately.

### Phase 6: Call End

17. When the WebSocket disconnects (caller hangs up):
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
| **LLM Agent** | `ivr/llm_agent.py` | Conversational LLM wrapper supporting Gemini, Groq, and Ollama with fallbacks |
| **TTS Synthesizer** | `audio_pipeline/tts_synthesizer.py` | Edge-TTS synthesizer with ffmpeg MP3→WAV converter and Redis cache |
| **Response Handler** | `ivr/response_handler.py` | Playback queue, locks, and barge-in (interruption) controller |
| **Latency Tracker** | `latency_tracker.py` | Tracks phase durations (NC, STT, LLM, TTS, RTT) per turn |
| **STT Handler** | `stt_handler.py` | Switchable client for remote Whisper API or local Faster-Whisper |
| **Session Manager** | `session_manager.py` | Redis-backed session state, history storage, and locking |

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

This is a **multi-stage Docker build** (the Dockerfile for this base image is maintained separately — the pre-built image is available on [Docker Hub](https://hub.docker.com/r/rajunify123/freeswitch-mod-audio-fork)):

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

## Conversational LLM Engine

Rather than a static, JSON-defined flow tree, the VoiceBot uses a multi-provider Conversational LLM Agent to handle user requests dynamically. The LLM acts as an interactive customer service assistant, guided by a system instruction prompt.

### Supported Providers

- **Google Gemini** (`gemini-2.0-flash`): Primary cloud provider, offering rapid response times (~350ms for first sentence) and high quality.
- **Groq** (`llama-3.3-70b-versatile`): High-speed OpenAI-compatible cloud inference provider, yielding responses in 200–400ms.
- **Ollama** (`qwen2.5:0.5b` or custom models): Runs locally inside or alongside the container for fully offline, cost-free execution.

### Resilience and Automatic Fallback

To prevent call failures due to rate limits or API quota exhaustion:
1. **Fallback Chain**: If Gemini fails (such as returning a `429 Resource Exhausted` error), the agent automatically shifts to Groq (if `GROQ_API_KEY` is provided) or local Ollama.
2. **Background Health Checks**: Rather than testing connections on incoming calls, the server runs a background health check loop.
   - Pings the primary Gemini API every 60 seconds (or every 30 minutes if the daily quota is exhausted) to determine if it has recovered.
   - Automatically switches back to Gemini when healthy, preserving cloud quota and avoiding connection latency during active calls.

### Conversational Guidelines

The agent is instructed to follow specific telephony guidelines via the system prompt:
- **Conciseness**: Restricts responses to 1-3 sentences suitable for spoken conversations.
- **Format Filtering**: Strips markdown, bullet points, asterisks, URLs, and code formatting to produce clean text suitable for TTS playback.
- **Warm Tone**: Pre-configured as a professional support representative for "Unified Reach Fiber" helping with speed issues, billing, plan options, and outages.

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
│  Min: 18KB (~0.3s) Max: 320KB (~10s)    │
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
│  STT (HTTP POST or local Whisper)       │
│  Returns transcribed text               │
└────────────────┬────────────────────────┘
                 │ "I want to check payment"
                 ▼
┌─────────────────────────────────────────┐
│  Conversational LLM (Gemini/Groq/Ollama)│
│  Generates token stream in real-time,   │
│  splits tokens into full sentences      │
└────────────────┬────────────────────────┘
                 │ Sentence 1, Sentence 2...
                 ▼
┌─────────────────────────────────────────┐
│  TTS (Edge-TTS Synthesizer)             │
│  Converts text to WAV (ffmpeg/torchaudio)│
│  Caches results in Redis/disk           │
└────────────────┬────────────────────────┘
                 │ WAV file path
                 ▼
┌─────────────────────────────────────────┐
│  ResponseHandler (Playback Queue)       │
│  fs_cli → uuid_broadcast → FreeSWITCH  │
│  Handles barge-in (interruption)        │
└─────────────────────────────────────────┘
```

> [!NOTE]
> ### 🧠 Deep-Dive: Why VAD Runs BEFORE Noise Cancellation (NC)
> 
> In a high-concurrency production telephony pipeline, running **Voice Activity Detection (VAD) before Noise Cancellation (NC)** is a deliberate, highly optimized design choice. While it might seem intuitive to clean up the audio *before* detecting speech, that approach creates severe scalability bottlenecks and audio degradation in practice:
> 
> 1. **Ultra-Low Latency Streaming vs. Computational Complexity**
>    * **Silero VAD** is a lightweight, highly optimized recurrent neural network. It processes individual 32ms incoming frames in **less than 1ms**, contributing virtually zero latency to the streaming audio loop.
>    * **DeepFilterNet2** is a deep convolutional/recurrent neural network. Running NC on every 32ms frame continuously for multiple calls would completely saturate the host's CPU and degrade under high concurrency. By putting VAD first, we filter out silent periods and **only invoke the heavy NC model on active speech segments**.
> 
> 2. **Denoising Quality and Context Windows**
>    * Deep neural noise suppression models rely heavily on **temporal context** (historical spectral data over hundreds of milliseconds) to model background noise profiles accurately.
>    * Denoising tiny, isolated 32ms slices in real-time results in metallic voice distortion, high-frequency "speech bubbling" artifacts, and harsh clipping. 
>    * By buffering the raw speech frames and applying NC to the **entire combined utterance block**, DeepFilterNet2 has full access to the temporal structure of the speech, producing incredibly natural, crystal-clear denoised audio.
> 
> 3. **CPU Preservation & Multi-Call Capacity**
>    * Because users spend up to 70% of a call either listening to the bot or sitting in silence, running NC continuously is extremely wasteful. 
>    * Placing VAD at the front gate acts as a highly efficient guard, allowing the system to run on modest hardware and scale to many concurrent channels.

---

## Project Structure

```
freeswitch_voicebot/
├── README.md                        # This file
├── QUICKSTART.md                    # 5-minute Docker setup guide
├── CHANGELOG.md                     # Version history
│
├── Dockerfile                       # All-in-one Docker image
├── entrypoint.sh                    # Dynamic config injection at startup
├── docker-compose.yml               # Bridge networking (production)
├── docker-compose.dev.yml           # Local dev (volume mounts for live editing)
├── docker-compose.host.yml          # Linux host networking (high performance)
│
├── config.py                        # Centralized configuration
├── server_multicall.py              # Main WebSocket server (FastAPI)
├── agent.py                         # FreeSWITCH ESL agent
├── stt_handler.py                   # Speech-to-text HTTP client
├── session_manager.py               # Redis-backed session management
├── requirements.txt                 # Python dependencies
│
├── audio_pipeline/                  # Audio processing modules
│   ├── __init__.py                  # Package init + singleton accessors
│   ├── improved_noise_canceller.py  # DeepFilterNet2 wrapper (active)
│   ├── noise_canceller.py           # Legacy noise canceller (reference)
│   ├── vad_detector.py              # Silero VAD + PerCallVADManager
│   └── audio_buffer.py             # Per-call audio buffering
│
├── ivr/                             # IVR logic
│   ├── __init__.py                  # Package init
│   ├── json_flow_engine.py          # JSON-based flow navigation
│   ├── intent_matcher.py            # Hybrid fuzzy + semantic matching
│   ├── response_handler.py          # Audio playback via fs_cli
│   └── flows/                       # Flow definitions
│       └── en.json                  # English IVR flow
│
├── sounds/                          # Pre-recorded audio files (.wav)
│
├── docker/                          # Docker and FreeSWITCH configs
│   ├── supervisord.conf             # Process manager configuration
│   └── freeswitch-config/
│       ├── vars.xml
│       ├── sip_profiles/internal.xml
│       ├── dialplan/
│       └── autoload_configs/
│
├── docs/                            # Documentation
│   ├── DOCKERHUB.md                 # Docker Hub description content
│   ├── WALKTHROUGH.md               # Detailed debugging history
│   ├── UPGRADE_PLAN.md              # Future upgrade roadmap
│   ├── PROJECT_SUMMARY.md           # Architecture summary
│   └── archive/                     # Legacy bare-metal files
│
├── scripts/                         # Developer utilities
│   ├── test_components.py           # Component testing
│   └── test_redis.py                # Redis testing
│
├── logs/                            # Log output (gitignored)
├── debug_audio/                     # NC debug audio (gitignored)
└── models/                          # Model cache (gitignored)
```

---

## Prerequisites

| Requirement | Purpose |
|---|---|
| **Docker** | Run the all-in-one FreeSWITCH + VoiceBot container |
| **Network access** | STT API at `164.52.203.140:8890` must be reachable |
| **SIP Softphone** | Any SIP client: Zoiper, Linphone, MicroSIP, Grandstream Wave, hardware phones |

> [!NOTE]
> Everything (FreeSWITCH, Redis, Python, dependencies) runs inside Docker. No local Python installation or WSL2 is required for running the voicebot.

---

## Installation & Running

The entire VoiceBot system (FreeSWITCH, Redis, WebSocket Server, and ESL Agent) runs in a **single multi-service Docker container** managed by `supervisord`. All settings are configurable via environment variables in the compose file — no code changes needed.

We provide **three compose configurations**:

| File | Use Case | Networking |
|---|---|---|
| `docker-compose.yml` | Production / default | Bridge (works everywhere) |
| `docker-compose.dev.yml` | Local development | Bridge + volume mounts |
| `docker-compose.host.yml` | Linux high-performance | Host networking |

> [!IMPORTANT]
> **Platform Guide:** `docker-compose.yml` and `docker-compose.dev.yml` work on **all platforms** (Windows, macOS, Linux). The `docker-compose.host.yml` uses host networking and is **Linux only** — Docker Desktop on Windows/macOS does not support `network_mode: host`.

### Option A: Production / Default Setup

All configs are baked into the image. The `entrypoint.sh` script dynamically patches FreeSWITCH XML configs using environment variables at startup.

1. **Build and start the container:**
   ```bash
   docker compose build
   docker compose up -d
   ```

2. **Verify the container processes are running:**
   ```bash
   docker exec -it freeswitch-voicebot supervisorctl status
   ```
   *Expected output showing all four services active:*
   ```text
   freeswitch                       RUNNING   pid 12, uptime 0:01:00
   redis                            RUNNING   pid 10, uptime 0:01:00
   voicebot-agent                   RUNNING   pid 15, uptime 0:00:48
   voicebot-server                  RUNNING   pid 14, uptime 0:00:52
   ```

3. **Customize settings** by editing environment variables in `docker-compose.yml`:
   ```yaml
   environment:
     - VOICEBOT_EXTENSION=5000     # Change the dial extension
     - EXTERNAL_IP=127.0.0.1       # NAT IP for SDP
     - STT_URL=http://your-stt/transcribe
     - NC_ENABLED=true             # Enable noise cancellation
   ```

---

### Option B: Local Development Setup

Volume-mounts your local source code and FreeSWITCH configs into the container for instant live editing without rebuilds.

```bash
docker compose -f docker-compose.dev.yml up -d
```

After editing Python code:
```bash
docker exec freeswitch-voicebot supervisorctl restart voicebot-server voicebot-agent
```

After editing FreeSWITCH XML configs:
```bash
docker exec freeswitch-voicebot fs_cli -x "reloadxml"
```

---

### Option C: Linux Production Setup (High Performance)

For high-concurrency production deployments on Linux, uses native **host networking** (`network_mode: host`) to avoid Docker NAT overhead.

```bash
docker compose -f docker-compose.host.yml up -d
```

---

## 📞 Making a Test Call

The voicebot works with **any SIP-compliant softphone** (Zoiper, Linphone, MicroSIP, Grandstream Wave, hardware phones, etc.).

When testing locally, register your softphone to `127.0.0.1:5060`. On Windows, Docker Desktop uses WSL2 which can have hairpin NAT limitations — using the loopback address avoids this.

1. **Configure your softphone** (running on the same host machine):
   * **Domain / SIP Server**: `127.0.0.1:5060`
   * **Username / Extension**: `1000` *(Any extension `1000-1019` works)*
   * **Password**: `1234`
   * **Transport**: UDP

2. **Dial extension `5000`** (or whatever you set `VOICEBOT_EXTENSION` to):
   * The bot will answer the call and play the welcome audio greeting.
   * Speak into your microphone — the bot will process your audio and respond!

3. **Watch the live logs:**
   ```bash
   docker compose logs -f
   ```


---

## Configuration Reference

All settings are in `config.py`. Key settings support environment variable overrides:

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `FREESWITCH_HOST` | `FREESWITCH_HOST` | `127.0.0.1` | FreeSWITCH ESL host |
| `FREESWITCH_PORT` | `FREESWITCH_PORT` | `8021` | FreeSWITCH ESL port |
| `WS_PORT` | `WS_PORT` | `8000` | WebSocket server port |
| `STT_PROVIDER` | `STT_PROVIDER` | `remote` | `remote` (HTTP API) or `local` (Faster-Whisper) |
| `STT_URL` | `STT_URL` | `http://164.52.203.140:8890/transcribe` | STT API endpoint (when `remote`) |
| `LLM_PROVIDER` | `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, or `ollama` |
| `GEMINI_API_KEY` | `GEMINI_API_KEY` | (empty) | API key for Gemini |
| `GEMINI_MODEL` | `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GROQ_API_KEY` | `GROQ_API_KEY` | (empty) | API key for Groq |
| `GROQ_MODEL` | `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OLLAMA_URL` | `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `OLLAMA_MODEL` | `qwen2.5:0.5b` | Ollama model name |
| `TTS_VOICE` | `TTS_VOICE` | `en-US-GuyNeural` | Edge-TTS voice identifier |
| `REDIS_HOST` | `REDIS_HOST` | `127.0.0.1` | Redis host |
| `MAX_CONCURRENT_CALLS` | `MAX_CONCURRENT_CALLS` | `5` | Max simultaneous calls |
| `DF_USE_GPU` | `DF_USE_GPU` | `False` | Enable CUDA for DeepFilterNet |
| `LOG_LEVEL` | `LOG_LEVEL` | `INFO` | Logging verbosity |

### Audio Processing Tuning

| Setting | Default | Description |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold (lower = more sensitive, higher = filters noise) |
| `VAD_MIN_SILENCE_DURATION_MS` | `800` | How long to wait after speech stops before processing (800ms is conversational) |
| `DF_GAIN` | `4.0` | Post-NC volume normalization gain |
| `DF_ATTENUATION_LIMIT` | `6.0` | Max noise reduction in dB |
| `ALLOW_INTERRUPTIONS` | `true` | Enable user barge-in during bot playback |

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

- Lower `VAD_MIN_SILENCE_DURATION_MS` (default 800ms may be too long for fast speakers).
- Raise `VAD_THRESHOLD` (default 0.5) to filter out background noise causing false speech detection.

### "mod_audio_fork" not loaded

```bash
fs_cli -x "load mod_audio_fork"
# If it fails, check: fs_cli -x "module_exists mod_audio_fork"
```

### Redis connection refused

Redis runs **inside** the all-in-one container via `supervisord` — there is no separate Redis container to manage.

```bash
# Check if Redis is running inside the container
docker exec freeswitch-voicebot supervisorctl status redis

# Restart Redis if it's down
docker exec freeswitch-voicebot supervisorctl restart redis

# Verify Redis is responding
docker exec freeswitch-voicebot redis-cli ping
# Expected output: PONG
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
