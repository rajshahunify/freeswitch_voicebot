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

### Option D: GPU Server (NVIDIA H100)
For shared GPU servers with port conflicts. See `.env.gpu.example` for config template:
```bash
cp .env.gpu.example .env.gpu   # Fill in your API keys
bash scripts/check_server.sh   # Run 7-step readiness check
docker compose -f docker-compose.gpu.yml up -d
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
freeswitch-voicebot  | 2026-06-03 12:00:00 - __main__ - INFO - 📞 NEW CALL STARTING
freeswitch-voicebot  | 2026-06-03 12:00:02 - audio_pipeline.vad_detector - INFO - 🎤 Speech START (prob: 0.94)
freeswitch-voicebot  | 2026-06-03 12:00:05 - audio_pipeline.vad_detector - INFO - 🎤 Speech END (speech=120, silence=805)
freeswitch-voicebot  | 2026-06-03 12:00:05 - event_emitter - INFO - [Transcription] User: 'hello' | Provider: remote | Latency: 320ms
freeswitch-voicebot  | 2026-06-03 12:00:06 - ivr.llm_agent - INFO - 🤖 LLM [gemini] first sentence in 350ms
freeswitch-voicebot  | 2026-06-03 12:00:06 - event_emitter - INFO - [BotResponse] Bot: 'Hello! How can I help you today?' | Provider: gemini | Latency: 420ms
```

---

## ⚙️ Step 5: Customize Settings

All settings are configurable via environment variables in the compose file or the `.env` file — no code changes needed:

```yaml
environment:
  - VOICEBOT_EXTENSION=5000           # Dial extension for the voicebot
  - EXTERNAL_IP=127.0.0.1            # NAT IP for SDP (set to server IP for remote access)
  - NC_ENABLED=false                  # Neural noise cancellation (CPU-intensive)
  - MAX_CONCURRENT_CALLS=5           # Max simultaneous calls
  - LOG_LEVEL=INFO                   # Logging verbosity
  
  # --- AI Pipeline Providers ---
  - STT_PROVIDER=remote              # "remote" (external API) or "local" (Faster-Whisper)
  - STT_URL=http://your-stt/transcribe
  
  - LLM_PROVIDER=gemini              # "gemini", "groq", or "ollama"
  - GEMINI_API_KEY=your-key          # Set in .env
  - GEMINI_MODEL=gemini-2.0-flash
  
  - GROQ_API_KEY=your-key            # Set in .env (for fallback or primary Groq)
  - GROQ_MODEL=llama-3.3-70b-versatile
  
  - OLLAMA_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=qwen2.5:0.5b
  
  - TTS_VOICE=en-US-AvaMultilingualNeural
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
