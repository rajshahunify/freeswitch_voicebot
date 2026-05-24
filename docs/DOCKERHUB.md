<!-- ============================================================ -->
<!-- 🏷️ DOCKER HUB CATEGORY RECOMMENDATIONS                        -->
<!-- Choose up to 3 categories. We highly recommend selecting:     -->
<!-- 1. Machine learning & AI                                     -->
<!-- 2. Networking                                                -->
<!-- 3. Developer tools                                           -->
<!-- ============================================================ -->

<!-- ============================================================ -->
<!-- 📝 DOCKER HUB: SHORT DESCRIPTION (paste into "Short description" box) -->
<!-- 100 characters max, plain text only, no markdown               -->
<!-- ============================================================ -->
<!--
All-in-one FreeSWITCH AI VoiceBot with real-time VAD, Whisper STT, and smart semantic intent flows
-->


<!-- ============================================================ -->
<!-- 📖 DOCKER HUB: FULL DESCRIPTION (paste into "Full description" box) -->
<!-- Markdown supported, no character limit                          -->
<!-- Copy everything below this line ↓↓↓                            -->
<!-- ============================================================ -->

# 🤖 FreeSWITCH Automated AI VoiceBot

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github&style=flat-square)](https://github.com/rajshahunify/freeswitch_voicebot)
[![Docker Pulls](https://img.shields.io/docker/pulls/rajunify123/freeswitch-voicebot?style=flat-square)](https://hub.docker.com/r/rajunify123/freeswitch-voicebot)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square)](#)

An **automated, real-time IVR VoiceBot** that handles inbound telephone calls via **FreeSWITCH**, transcribes caller speech, and navigates a dynamic JSON-defined conversation flow. Powered by **Silero VAD**, **DeepFilterNet2 neural noise cancellation**, and **Whisper STT**—all in a single, ready-to-run container!

---

## 🌟 Why This VoiceBot?

This image is a complete, production-ready solution that turns any standard SIP/VoIP telephone line into an interactive AI voice conversation. It is designed to handle high-fidelity audio streams, clean background noise in real-time, and route callers through conversational trees with high semantic accuracy.

### 🚀 Key Technical Highlights

*   **📦 All-in-One Container:** Includes FreeSWITCH, Redis, FastAPI WebSocket Server, and an ESL (Event Socket Library) orchestrator agent, all pre-configured and managed cleanly by `supervisord`.
*   **⚡ Sub-Second Latency:** Uses **Silero VAD** for ultra-fast, local speech detection. No roundtrips to the cloud just to figure out if someone is talking!
*   **🧠 Deep Noise Cancellation:** Integrates **DeepFilterNet2** neural networks to filter out street noise, wind, and background static before sending audio for transcription.
*   **🗣️ Whisper STT Integration:** Streams chunked audio and issues ultra-fast transcription queries directly to a high-speed Whisper backend.
*   **🎯 Smart Router (Fuzzy + Semantic):** Custom JSON IVR Engine that matches user intent using hybrid matching:
    *   *Pass 1:* Fast fuzzy matching for keyword shortcuts and standard synonym expansions (e.g., yes/no).
    *   *Pass 2:* Semantic cosine-similarity matching (via sentence-transformers) to capture intent even when callers use different wording.
*   **📞 Enterprise Ready:** Supports concurrent channels (up to 5 in default settings) with dedicated per-call audio pipelines, sound buffers, and full state locking using Redis.

---

## 🏗️ How It Works

> 📞 **Caller** → FreeSWITCH (SIP) → WebSocket Audio Stream → **AI Pipeline** → Voice Response

```
+-----------------------------------------------------------+
|                   FreeSWITCH Container                     |
|                                                            |
|   +--------------+          +------------------------+     |
|   |  mod_sofia   |--------->|    mod_audio_fork      |     |
|   |  (SIP Stack) |          |  (WebSocket Streamer)  |     |
|   +--------------+          +-----------+------------+     |
+----------------------------------------|------------------+
                                         |
                                         v  WebSocket: 16kHz Mono PCM
+-----------------------------------------------------------+
|                   Python VoiceBot Engine                    |
|                                                            |
|   +---------------------------------------------------+    |
|   |            Per-Call Audio Pipeline                 |    |
|   |                                                   |    |
|   |   1. Silero VAD        - Speech detection         |    |
|   |   2. DeepFilterNet2    - Neural noise cancellation|    |
|   |   3. Whisper STT       - Speech-to-Text          |    |
|   |   4. JSON IVR Engine   - Intent matching         |    |
|   +-------------------------+-------------------------+    |
|                             |                              |
|                             v                              |
|                  [Broadcast Response Audio]                 |
+-----------------------------------------------------------+
```

---

## 🏃 Quick Start Guide

You can launch the complete stack using **Docker Compose** or as a **Standalone Container**.

### Option A: Using Docker Compose (Recommended)

1. Clone the project files:
   ```bash
   git clone https://github.com/rajshahunify/freeswitch_voicebot.git
   cd freeswitch_voicebot
   ```

2. Spin up the entire environment:
   ```bash
   docker compose up -d
   ```

### Option B: Running Standalone Container

Run the image directly with custom environment variables:
```bash
docker run -d \
  --name freeswitch-voicebot \
  -p 5060:5060/udp \
  -p 5080:5080/udp \
  -p 8021:8021 \
  -p 16384-16484:16384-16484/udp \
  -e VOICEBOT_EXTENSION=5000 \
  -e EXTERNAL_IP=127.0.0.1 \
  rajunify123/freeswitch-voicebot:latest
```

### 📞 Making Your First Test Call

1. Download and open any SIP softphone (e.g., **Zoiper**, **MicroSIP**, or **Linphone**).
2. Configure a new account and point it to your Docker host IP (e.g., `127.0.0.1:5060`).
3. Log in with **Username:** `1000` and **Password:** `1234`.
4. Dial **`5000`** (or your custom `VOICEBOT_EXTENSION`).
5. **Start talking!** The voice bot will answer immediately and engage in a real-time conversation.

---

## 🖥️ Choose Your Deployment Mode

This project ships with **three** Docker Compose files optimized for different platforms and use cases:

| Compose File | Platform | Use Case | Command |
| :--- | :--- | :--- | :--- |
| `docker-compose.yml` | ✅ Windows, macOS, Linux | **Default / Production** — Bridge networking, works everywhere | `docker compose up -d` |
| `docker-compose.dev.yml` | ✅ Windows, macOS, Linux | **Development** — Volume mounts for live code editing without rebuild | `docker compose -f docker-compose.dev.yml up -d` |
| `docker-compose.host.yml` | ⚠️ Linux only | **High-Performance** — Host networking, zero NAT overhead | `docker compose -f docker-compose.host.yml up -d` |

### 🪟 Windows / 🍎 macOS Users

Use the **default** `docker-compose.yml` (bridge mode). Docker Desktop on Windows and macOS does not support host networking.

```bash
docker compose up -d
```

> **Tip:** Set `EXTERNAL_IP=127.0.0.1` when testing locally from the same machine.

### 🐧 Linux Users

You have two options:

**Standard (bridge mode)** — same as Windows/macOS:
```bash
docker compose up -d
```

**High-performance (host networking)** — recommended for production servers with many concurrent calls:
```bash
docker compose -f docker-compose.host.yml up -d
```

> **Note:** With host networking, set `EXTERNAL_IP` to your server's public or LAN IP address.

### 🔧 Development Mode (Any Platform)

For active development with live code reloading (source code is volume-mounted into the container):
```bash
docker compose -f docker-compose.dev.yml up -d
```

After editing Python code, restart the services without rebuilding:
```bash
docker exec freeswitch-voicebot supervisorctl restart voicebot-server voicebot-agent
```

After editing FreeSWITCH XML configs:
```bash
docker exec freeswitch-voicebot fs_cli -x "reloadxml"
```

---

## ⚙️ Configuration Reference

Customize the behavior of the voicebot using these environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VOICEBOT_EXTENSION` | `5000` | The extension dialed on your softphone to trigger the bot. |
| `EXTERNAL_IP` | `127.0.0.1` | Public or host IP address for proper SDP negotiation. |
| `WEBSOCKET_URL` | `ws://127.0.0.1:8000/media` | Target websocket endpoint for streaming call audio. |
| `STT_URL` | `http://164.52.203.140:8890/transcribe` | Transcription endpoint of your Whisper instance. |
| `NC_ENABLED` | `false` | Set to `true` to enable DeepFilterNet2 noise cancellation. |
| `VAD_THRESHOLD` | `0.3` | Sensitivity threshold for speech detection (lower = more sensitive). |
| `MAX_CONCURRENT_CALLS` | `5` | Maximum number of active voice calls allowed simultaneously. |
| `LOG_LEVEL` | `INFO` | Output verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 🛠️ Deploying Custom Conversational Flows

To change how the voice bot answers, simply define your custom dialog steps in your JSON conversation file. You can mount your own custom configurations and audio soundboards into the container:

*   **Flow Configuration:** Mount your custom JSON flow file to `/app/ivr/flow.json`.
*   **Audio Assets:** Place your `.wav` prompts in the mounted FreeSWITCH sounds directory at `/usr/local/freeswitch/sounds/custom/`.

---

## 🔗 Code & Support

*   📂 **GitHub Repository:** [github.com/rajshahunify/freeswitch_voicebot](https://github.com/rajshahunify/freeswitch_voicebot)
*   🐛 **Issue Tracker:** [Submit bugs or feature requests here](https://github.com/rajshahunify/freeswitch_voicebot/issues)
*   📄 **License:** MIT License. Feel free to use, modify, and build upon this project.
