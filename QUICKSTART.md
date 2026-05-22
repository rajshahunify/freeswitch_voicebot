# 🚀 VoiceBot 5-Minute Quick Start Guide

Get your VoiceBot up and running instantly using our all-in-one Docker configuration.

---

## 🛠️ Step 1: Start the Container

The voicebot stack includes **FreeSWITCH**, **Redis**, a **FastAPI WebSocket Server**, and an **ESL Agent**, all managed by a single process manager inside Docker.

### Option A: Production / Default
All configs are baked into the image. Settings are configured via environment variables in `docker-compose.yml`.
```bash
# Build the image
docker compose build

# Start the stack
docker compose up -d
```

### Option B: Local Development
Volume-mounts your source code and FreeSWITCH configs for live editing without rebuilds:
```bash
docker compose -f docker-compose.dev.yml up -d
```

### Option C: Linux Production (Host Networking)
Uses high-performance host networking (highly recommended for high concurrency on Linux):
```bash
docker compose -f docker-compose.host.yml up -d
```

---

## 🔍 Step 2: Verify System Health

Verify that all four internal processes have started correctly:

```bash
docker exec -it freeswitch-voicebot supervisorctl status
```

**Expected Output:**
```text
freeswitch                       RUNNING   pid 12, uptime 0:01:00
redis                            RUNNING   pid 10, uptime 0:01:00
voicebot-agent                   RUNNING   pid 15, uptime 0:00:48
voicebot-server                  RUNNING   pid 14, uptime 0:00:52
```

To test the FastAPI server health directly, curl the host endpoint:
```bash
curl http://127.0.0.1:8000/health
```

---

## 📞 Step 3: Register Your Softphone & Call

Configure **any SIP-compliant softphone** (Zoiper, Linphone, MicroSIP, Grandstream Wave, etc.) to connect to the bot:

| Setting | Value |
|---|---|
| **Account Type** | SIP |
| **SIP Server / Domain** | `127.0.0.1:5060` *(registering to loopback avoids Windows hairpin UDP bugs)* |
| **Username / Extension** | `1000` *(Extensions 1000 to 1019 are pre-configured)* |
| **Password** | `1234` |
| **Transport** | UDP |

1. Wait for your softphone to show **"Registered"** status.
2. Dial **`5000`** (or whatever you set `VOICEBOT_EXTENSION` to in the compose file) and call.
3. You should hear the greeting audio message.
4. Speak into your microphone (e.g., say *"hello"* or *"billing"*) — the bot will process your audio and respond back!

---

## 📡 Step 4: Monitor Live Processing Logs

To watch the live audio pipeline, VAD transitions, STT transcriptions, and flow decisions, tail the Docker logs:

```bash
docker compose logs -f
```

### What You Should See in the Logs:
```text
freeswitch-voicebot  | 2026-05-22 12:00:00 - __main__ - INFO - 📞 NEW CALL STARTING
freeswitch-voicebot  | 2026-05-22 12:00:02 - audio_pipeline.vad_detector - INFO - 🎤 Speech START (prob: 0.94)
freeswitch-voicebot  | 2026-05-22 12:00:05 - audio_pipeline.vad_detector - INFO - 🎤 Speech END (silence detected)
freeswitch-voicebot  | 2026-05-22 12:00:06 - stt_handler - INFO - 🎯 STT: 'hello' (Duration: 320ms)
freeswitch-voicebot  | 2026-05-22 12:00:06 - ivr.json_flow_engine - INFO - 🎯 Match: 'hello' -> Playing 'english_menu.wav'
```

---

## ⚙️ Step 5: Customize Settings

All settings are configurable via environment variables in the compose file — no code changes needed:

```yaml
environment:
  - VOICEBOT_EXTENSION=5000           # Dial extension for the voicebot
  - EXTERNAL_IP=127.0.0.1            # NAT IP for SDP (set to server IP for remote access)
  - STT_URL=http://your-stt/transcribe  # Your STT API endpoint
  - NC_ENABLED=true                   # Enable DeepFilterNet2 noise cancellation
  - MAX_CONCURRENT_CALLS=10          # Max simultaneous calls
  - LOG_LEVEL=DEBUG                   # Logging verbosity
```

After changing compose settings:
```bash
docker compose up -d   # Recreates the container with new env vars
```

---

## 🐛 Troubleshooting

### 1. Softphone Shows "Registration Failed (408 Timeout)"
* Verify that you registered to `127.0.0.1:5060` and not your external Wi-Fi IP (due to Windows hairpin NAT bugs).
* Verify the container is running: `docker ps`.
* Check if a local FreeSWITCH instance is already running on your machine and blocking port 5060.

### 2. Connected but No Audio / Bot Doesn't Respond
* Verify your microphone is enabled in your softphone.
* For development mode, double-check that your active volume mounts match the ones in `docker-compose.dev.yml`.
* Check the logs for Whisper STT API timeouts.
