# The Final Plan v2 — Updated with Your Feedback

All 6 of your comments have been incorporated. Here's what changed:

> [!IMPORTANT]
> **Changes from v1:**
> 1. ✅ Ollama/Qwen is now a **first-class option**, not just a backup — switchable via env vars
> 2. ✅ External STT server **kept as a provider** alongside local Faster-Whisper
> 3. ✅ **Barge-in, streaming TTS, WebSocket event protocol** added to Phase 1
> 4. ✅ HTTP POST **not removed** from STT handler — switchable local/remote
> 5. ✅ Whisper model quality addressed — `small` recommended over `base`, with remote GPU path
> 6. ✅ **Round-trip time (RTT) tracking** added per request

---

## The Decisions (Updated)

| Question | Decision | Why |
|----------|----------|-----|
| **LLM** | **Switchable**: Gemini Flash API _or_ Ollama/Qwen — your choice via `LLM_PROVIDER` env var | Both are first-class. Gemini for speed+quality when you have internet. Ollama for fully offline/free-forever/in-house. |
| **LLM Model (Ollama)** | `qwen2.5:0.5b` (fast, ~400MB RAM) or `qwen2.5:3b` (better, ~3GB RAM) — your choice via `OLLAMA_MODEL` env var | 0.5B gives <1s on CPU. 3B gives much better answers. Pick based on your hardware. |
| **STT** | **Switchable**: External API (`http://164.52.203.140:8890/transcribe`) _or_ local Faster-Whisper — via `STT_PROVIDER` env var | External server works great today — keep using it. Local Whisper is fallback when the server goes down, or for fully offline use. |
| **STT Model (local)** | `small` (recommended, ~460MB RAM, better accuracy) instead of `base` | `base` has mediocre accuracy. `small` is 2x better with only ~200MB more RAM. `medium` and `large-v3` need GPU — use them on remote servers. |
| **TTS** | Edge-TTS with **streaming playback** (not wait for full file) | Cuts perceived latency by 50%+ — user hears first words while rest is still generating. |
| **Barge-in** | Yes, in Phase 1 | SiphonAI-style. User can interrupt the bot mid-speech. Essential for natural conversation. |
| **WebSocket Events** | Formalized JSON event protocol (SiphonAI-compatible) | Makes future migration or integration with SiphonAI possible. Low effort, high value. |
| **Cloud provider** | GCP (Phase 2) | $300 free credit. But we don't touch this until Phase 1 is complete. |
| **Architecture** | Start monolith → split into microservices | Don't over-engineer on day 1. |

---

## Provider Switching — How It Works

All providers are controlled via environment variables in `docker-compose.yml` or `.env`:

```yaml
# docker-compose.yml (environment section)
environment:
  # --- STT Provider ---
  STT_PROVIDER: "remote"           # "remote" (your external server) or "local" (Faster-Whisper)
  STT_REMOTE_URL: "http://164.52.203.140:8890/transcribe"
  STT_LOCAL_MODEL: "small"         # tiny, base, small, medium, large-v3
  STT_LOCAL_DEVICE: "cpu"          # "cpu" or "cuda"

  # --- LLM Provider ---
  LLM_PROVIDER: "gemini"           # "gemini" or "ollama"
  GEMINI_API_KEY: "your-key-here"
  GEMINI_MODEL: "gemini-2.0-flash"
  OLLAMA_URL: "http://host.docker.internal:11434"
  OLLAMA_MODEL: "qwen2.5:0.5b"    # 0.5b (fast) or 3b (smart) or 7b (GPU only)

  # --- TTS Provider ---
  TTS_VOICE: "en-US-GuyNeural"
  TTS_STREAMING: "true"            # Stream audio chunks as they arrive
```

> [!TIP]
> **To switch from Gemini to Ollama**: Just change `LLM_PROVIDER=ollama` and restart. No code changes.
> **To switch from remote STT to local**: Just change `STT_PROVIDER=local` and restart.
> **Mix and match freely** — e.g., remote STT + Ollama LLM for a fully free stack.

---

## Whisper Model Quality — The Real Story

Your concern is valid. Here's the honest breakdown:

| Model | Size | RAM | CPU Latency (3s audio) | Accuracy | When to Use |
|-------|------|-----|----------------------|----------|-------------|
| `tiny` | 75MB | ~150MB | ~200ms | Poor | Testing only |
| `base` | 142MB | ~300MB | ~400ms | Mediocre | Not recommended |
| **`small`** | **466MB** | **~460MB** | **~800ms** | **Good** | **← Recommended for CPU** |
| `medium` | 1.5GB | ~1.5GB | ~3-5s | Very good | Only with GPU or beefy CPU |
| `large-v3` | 3.1GB | ~3GB | ~10-15s | Excellent | GPU only (remote server) |

**Our approach**: Default to `small` locally. When you deploy to a GPU machine or use your external server, switch to `large-v3` via env var. The code supports all models — you just change `STT_LOCAL_MODEL`.

### Free GPU/VM Resources for Heavy Models

> [!NOTE]
> These are for when you want to run `large-v3` Whisper or `qwen2.5:7b+` — **not needed for Phase 1**.

| Platform | Free Tier | GPU | How to Use |
|----------|-----------|-----|-----------|
| **Oracle Cloud** | Always-free ARM VM (4 CPU, 24GB RAM) | No GPU, but ARM is fast for Whisper `small` | Deploy Faster-Whisper as a Docker container |
| **Google Colab** | Free T4 GPU (12GB VRAM, time-limited) | T4 | Run Whisper `large-v3` as a Colab notebook → expose via ngrok |
| **Kaggle Notebooks** | Free P100 GPU (30h/week) | P100 | Same as Colab but more hours |
| **Groq API** | Free tier — `whisper-large-v3` transcription | Cloud | `STT_PROVIDER=remote`, point URL to Groq. **Best free option for high-quality STT.** |
| **Hugging Face Spaces** | Free CPU (16GB) or paid GPU ($0.60/hr) | Optional | Deploy Faster-Whisper as a Gradio API |
| **Your existing server** | `http://164.52.203.140:8890` | Unknown | Already running — keep using it! |

> [!TIP]
> **Groq's free Whisper API** is probably the best short-term option for high-quality STT at zero cost. It runs `large-v3` on their hardware. We can add it as a third STT provider (`STT_PROVIDER=groq`) with minimal code.

---

## What Your Laptop Actually Runs (Updated Math)

```
Component                    RAM        CPU Latency      Notes
─────────────────────────────────────────────────────────────────
Docker Desktop               ~2.0 GB    —                Already running
FreeSWITCH container         ~200 MB    —                Already running
Python voicebot container    ~300 MB    —                Already running
Redis container              ~50 MB     —                Already running
Faster-Whisper (small model) ~460 MB    ~800ms           NEW — only if STT_PROVIDER=local
Edge-TTS client              ~30 MB     ~200ms           NEW — thin HTTP client
Gemini API client            ~10 MB     ~300ms           NEW — if LLM_PROVIDER=gemini
─────────────────────────────────────────────────────────────────
TOTAL (remote STT + Gemini)  ~2.6 GB    ~500ms           Lightest config
TOTAL (local STT + Gemini)   ~3.1 GB    ~1300ms          Offline STT
TOTAL (local STT + Ollama)   ~3.5-6 GB  ~2-4s            Fully offline
```

**If using remote STT + Gemini (lightest):**
`VAD (0ms) + Remote STT (200ms) + Gemini (300ms) + Edge-TTS stream (100ms first chunk) = ~600ms` ← Very fast

**If using local STT small + Gemini:**
`VAD (0ms) + Whisper small (800ms) + Gemini (300ms) + Edge-TTS stream (100ms) = ~1200ms` ← Good

**If fully offline (local STT + Ollama 0.5B):**
`VAD (0ms) + Whisper small (800ms) + Qwen 0.5B (500-1000ms) + Edge-TTS stream (100ms) = ~1400-1900ms` ← Acceptable

---

## The Architecture (Updated)

```
              YOUR 16GB LAPTOP
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌──────────────┐                                        │
│  │  FreeSWITCH  │──── mod_audio_fork ──────┐             │
│  │  Container   │                          │             │
│  │  (SIP+RTP)   │◄── uuid_broadcast ───┐   │             │
│  └──────────────┘     (streaming!)     │   │             │
│         ▲                              │   │             │
│         │ uuid_break (barge-in)        │   │             │
│         │                              │   ▼             │
│  ┌─────┴───────────────────────────────┴───────────────┐ │
│  │              Python Voicebot Container               │ │
│  │                                                     │ │
│  │  ┌─────────────────────────────────────────────┐    │ │
│  │  │ server_multicall.py (WebSocket + VAD)       │    │ │
│  │  │  + Barge-in detection                       │    │ │
│  │  │  + WebSocket Event Protocol (JSON events)   │    │ │
│  │  │  + RTT Latency Tracker                      │    │ │
│  │  └──────────────┬──────────────────────────────┘    │ │
│  │                 │                                   │ │
│  │                 ▼ (speech segment)                   │ │
│  │  ┌──────────────────────────────┐                   │ │
│  │  │ stt_handler.py              │                    │ │
│  │  │  ├─ RemoteSTT (HTTP POST)   │ ← your server     │ │
│  │  │  ├─ LocalWhisperSTT         │ ← Faster-Whisper  │ │
│  │  │  └─ GroqSTT (future)        │ ← Groq free API   │ │
│  │  │  [switchable via env var]   │                    │ │
│  │  └──────────┬───────────────────┘                   │ │
│  │             │ text                                  │ │
│  │             ▼                                       │ │
│  │  ┌──────────────────────────────┐                   │ │
│  │  │ llm_agent.py                │                    │ │
│  │  │  ├─ GeminiProvider          │ ← Gemini Flash     │ │
│  │  │  └─ OllamaProvider          │ ← Qwen 0.5B/3B    │ │
│  │  │  [switchable via env var]   │                    │ │
│  │  └──────────┬───────────────────┘                   │ │
│  │             │ response text                         │ │
│  │             ▼                                       │ │
│  │  ┌──────────────────────────────┐                   │ │
│  │  │ tts_synthesizer.py          │                    │ │
│  │  │  ├─ Edge-TTS (streaming)    │                    │ │
│  │  │  └─ Redis cache             │                    │ │
│  │  └──────────┬───────────────────┘                   │ │
│  │             │ .wav chunks (streamed)                │ │
│  │             ▼                                       │ │
│  │  ┌──────────────────────────────┐                   │ │
│  │  │ response_handler.py         │                    │ │
│  │  │  ├─ Streaming playback      │                    │ │
│  │  │  └─ Barge-in stop           │                    │ │
│  │  └──────────────────────────────┘                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────┐    ┌────────────┐                        │
│  │   Redis    │    │  Ollama    │ (optional)              │
│  │ Container  │    │ Container  │ qwen2.5:0.5b           │
│  └────────────┘    └────────────┘                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
              │
              │ (external calls — only when using remote providers)
              ▼
     ┌──────────────────┐     ┌──────────────────────┐
     │  Google Cloud     │     │  Your STT Server      │
     │  • Gemini API     │     │  164.52.203.140:8890  │
     │  • Edge-TTS CDN   │     │  (Faster-Whisper)     │
     └──────────────────┘     └──────────────────────┘
```

---

## SiphonAI-Inspired Improvements (Now in Phase 1)

### 1. Barge-In (User Interrupts Bot Mid-Speech)

**Current behavior**: When `ALLOW_INTERRUPTIONS=False`, audio from the caller is ignored while the bot speaks ([line 217](file:///c:/Users/unify/freeswitch_voicebot/server_multicall.py#L217)). The user must wait for the bot to finish.

**New behavior**: When the user starts speaking while the bot is playing audio:
1. VAD detects speech during bot playback
2. Send `uuid_break <call_uuid>` to FreeSWITCH (stops current audio instantly)
3. Buffer the new speech and process it normally
4. Emit a `speech_started` WebSocket event

**Implementation**:
```python
# In process_audio_segment() — modify the existing interruption check
if response_handler.is_speaking(call_uuid):
    if config.ALLOW_INTERRUPTIONS:
        # BARGE-IN: Stop the bot and process the new speech
        logger.info(f"[{call_uuid}] 🛑 BARGE-IN: User interrupted bot")
        subprocess.run(["fs_cli", "-x", f"uuid_break {call_uuid}"], 
                       capture_output=True, timeout=2)
        response_handler.mark_stopped(call_uuid)
        emit_event(call_uuid, "barge_in", {"timestamp": time.time()})
        # Continue processing the audio segment normally...
    else:
        return  # Ignore audio (old behavior)
```

### 2. Streaming LLM → Streaming TTS (Sentence-Level Pipeline)

**Current behavior**: Wait for full LLM response → wait for full TTS WAV → play. User waits ~1.5-2s.

**New behavior**: Stream LLM tokens → buffer until sentence boundary → pipe sentence to TTS immediately → play audio while LLM keeps generating next sentence. User hears first words within ~500ms of LLM starting.

```
Timeline comparison:
OLD: [----LLM full response (800ms)----][----TTS full audio (500ms)----][play]
NEW: [LLM sentence 1 (300ms)][TTS+play sentence 1] ← user hears audio HERE
     [LLM sentence 2 (200ms)][TTS+play sentence 2]  (overlapping with LLM)
     [LLM sentence 3...]
```

**Implementation approach**:
```python
# llm_agent.py — streaming mode
async def get_response_stream(self, user_text, history):
    """Yield complete sentences as they form from LLM stream"""
    buffer = ""
    async for token in self._stream_tokens(user_text, history):
        buffer += token
        # Check for sentence boundary: . ? ! or newline
        if re.search(r'[.!?]\s', buffer) or buffer.endswith(('.', '!', '?')):
            sentence = buffer.strip()
            buffer = ""
            yield sentence  # Send complete sentence to TTS immediately

# server_multicall.py — streaming pipeline
async for sentence in llm_agent.get_response_stream(text, history):
    # Each sentence goes to TTS immediately — don't wait for full response
    wav_path = await tts_synthesizer.synthesize(sentence)
    response_handler.queue_audio(call_uuid, wav_path)
```

Both **Gemini** and **Ollama** support streaming natively:
- Gemini: `model.generate_content(prompt, stream=True)`
- Ollama: `"stream": True` in the API request

> [!NOTE]
> Edge-TTS also streams its audio output. So we get **double streaming**: LLM→sentence→TTS→audio chunks→FreeSWITCH. The user hears the first words of sentence 1 while the LLM is still generating sentence 2. This is how production voicebots work.

> [!NOTE]
> FreeSWITCH's `uuid_broadcast` plays files sequentially when you queue them. We use `uuid_fileman` to queue the next chunk while the current one plays. The result: seamless streaming with no gap between chunks.

### 3. WebSocket Event Protocol (SiphonAI-Compatible)

**What**: Formalize the events between the Python server and any future WebSocket clients (dashboard, external integrations) using structured JSON messages. This matches SiphonAI's event spec.

**Events we emit**:
```json
// Speech started (user begins talking)
{"event": "speech_started", "call_uuid": "abc-123", "timestamp": 1706000000.0}

// Speech ended (VAD silence detected)
{"event": "speech_ended", "call_uuid": "abc-123", "timestamp": 1706000003.0, "duration_ms": 3000}

// Transcription complete
{"event": "transcription", "call_uuid": "abc-123", "text": "I want to check my balance", "stt_ms": 450, "provider": "remote"}

// Bot response generated
{"event": "bot_response", "call_uuid": "abc-123", "text": "Sure, let me look up your balance.", "llm_ms": 280, "provider": "gemini"}

// TTS playback started
{"event": "tts_started", "call_uuid": "abc-123", "text": "Sure, let me look up...", "tts_ms": 150}

// Barge-in detected
{"event": "barge_in", "call_uuid": "abc-123", "timestamp": 1706000005.0}

// DTMF detected (keypad press)
{"event": "dtmf", "call_uuid": "abc-123", "digit": "1"}

// Call metrics (per-turn summary)
{"event": "turn_metrics", "call_uuid": "abc-123", 
 "stt_ms": 450, "llm_ms": 280, "tts_ms": 150, 
 "pipeline_ms": 920, "rtt_ms": 1050, "provider_stt": "remote", "provider_llm": "gemini"}
```

**Implementation**: Add an `event_emitter.py` module that broadcasts these to all connected WebSocket clients. In Phase 1, these events go to logs. In Phase 2, the web dashboard subscribes to them.

---

## File Changes (Updated)

| File | Status | What happens |
|------|--------|-------------|
| [server_multicall.py](file:///c:/Users/unify/freeswitch_voicebot/server_multicall.py) | **MODIFY** | Replace `flow_engine.process_input()` with `llm_agent` call. Add barge-in logic. Add WebSocket event emitting. Add RTT tracking. Wire streaming TTS. |
| [stt_handler.py](file:///c:/Users/unify/freeswitch_voicebot/stt_handler.py) | **REWRITE** | Multi-provider STT: keep `RemoteSTT` (HTTP POST — existing logic), add `LocalWhisperSTT` (Faster-Whisper). Factory function picks provider from env var. Same `transcribe(audio_bytes) → str` interface. |
| `ivr/llm_agent.py` | **NEW** | Multi-provider LLM: `GeminiProvider` + `OllamaProvider`. Factory picks from env var. `get_response(user_text, history) → str`. |
| `audio_pipeline/tts_synthesizer.py` | **NEW** | Edge-TTS with streaming support. `synthesize(text) → wav_path` (sync) and `synthesize_stream(text, call_uuid)` (streaming chunks). Redis cache. |
| `event_emitter.py` | **NEW** | WebSocket event broadcaster. `emit_event(call_uuid, event_type, data)`. Logs events + broadcasts to connected clients. |
| `latency_tracker.py` | **NEW** | Per-request RTT + per-component timing. `LatencyTracker` class with `start()`, `mark(component_name)`, `finish()`. Logs full breakdown + RTT. |
| [response_handler.py](file:///c:/Users/unify/freeswitch_voicebot/ivr/response_handler.py) | **MODIFY** | Add `queue_audio()` for streaming TTS. Add `mark_stopped()` for barge-in. Accept dynamic wav paths. |
| [config.py](file:///c:/Users/unify/freeswitch_voicebot/config.py) | **MODIFY** | Add all new env vars: `STT_PROVIDER`, `LLM_PROVIDER`, `GEMINI_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`, `STT_LOCAL_MODEL`, `TTS_VOICE`, `TTS_STREAMING`, `ALLOW_INTERRUPTIONS=true`. |
| [requirements.txt](file:///c:/Users/unify/freeswitch_voicebot/requirements.txt) | **MODIFY** | Add: `faster-whisper`, `edge-tts`, `google-generativeai`. Keep existing deps. |
| [Dockerfile](file:///c:/Users/unify/freeswitch_voicebot/Dockerfile) | **MODIFY** | Add Faster-Whisper model download step (conditional, only if building for local STT). |
| [docker-compose.yml](file:///c:/Users/unify/freeswitch_voicebot/docker-compose.yml) | **MODIFY** | Add all env vars. Optionally add Ollama service. |
| `ivr/flows/`, `ivr/json_flow_engine.py`, `ivr/intent_matcher.py` | **KEEP (unused)** | Old IVR files stay for reference. |
| All `audio_pipeline/` files | **KEEP** | VAD, noise cancellation, audio buffer — unchanged. |
| [session_manager.py](file:///c:/Users/unify/freeswitch_voicebot/session_manager.py) | **KEEP** | Store LLM conversation history per call in existing Redis sessions. |

---

## Execution Order — Step by Step

### Step 1: Config + Provider Framework
**What**: Update [config.py](file:///c:/Users/unify/freeswitch_voicebot/config.py) with all new env vars. Create the `latency_tracker.py` and `event_emitter.py` utilities.

**Why first**: Every subsequent step depends on config values and these utilities.

**Changes**:
- Add STT, LLM, TTS config sections to [config.py](file:///c:/Users/unify/freeswitch_voicebot/config.py)
- Create `latency_tracker.py`:
  ```python
  class LatencyTracker:
      def __init__(self, call_uuid):
          self.call_uuid = call_uuid
          self.start_time = time.time()
          self.marks = {}
      
      def mark(self, name):
          """Mark start/end of a component"""
          self.marks[name] = time.time()
      
      def summary(self):
          """Return dict of component timings + total RTT"""
          # Calculate per-component times + total round-trip
  ```
- Create `event_emitter.py` with `emit_event()` function

**Test**: Import and verify config loads all env vars correctly.

---

### Step 2: Multi-Provider STT Handler
**What**: Rewrite [stt_handler.py](file:///c:/Users/unify/freeswitch_voicebot/stt_handler.py) with a provider pattern.

**Why second**: STT is the entry point. Everything depends on text.

**Changes**:
- Keep the existing `STTHandler` class as `RemoteSTT` (HTTP POST to external server) — **don't remove it**
- Add `LocalWhisperSTT` using `faster-whisper` library (replaces the placeholder `WhisperLocalSTT`)
- Add factory function:
  ```python
  def create_stt_handler(provider: str = "remote") -> STTHandler:
      if provider == "remote":
          return RemoteSTT(url=config.STT_REMOTE_URL, params=config.STT_PARAMS)
      elif provider == "local":
          return LocalWhisperSTT(model_size=config.STT_LOCAL_MODEL, device=config.STT_LOCAL_DEVICE)
      # Future: elif provider == "groq": return GroqSTT(api_key=...)
  ```
- All providers implement the same `transcribe(audio_bytes) → Optional[str]` interface
- Add automatic **fallback**: if remote STT fails 3 times consecutively, auto-switch to local (if available)

**Test**: Run container, make a call, verify transcription works with `STT_PROVIDER=remote` (existing server), then switch to `STT_PROVIDER=local`.

---

### Step 3: Edge-TTS Synthesizer (with Streaming)
**What**: Create `audio_pipeline/tts_synthesizer.py`.

**Changes**:
- `synthesize(text) → wav_file_path` — standard mode, generates full file first
- `synthesize_stream(text, call_uuid)` — streaming mode, plays audio as chunks arrive
- Redis caching: `hash(text + voice) → wav_path` (skip TTS for repeated phrases like "How can I help you?")
- Configurable voice via `TTS_VOICE` env var

**Test**: `synthesizer.synthesize("Hello!")` → verify valid WAV. Play via `uuid_broadcast`.

---

### Step 4: Multi-Provider LLM Agent
**What**: Create `ivr/llm_agent.py` — the brain.

**Changes**:
- `GeminiProvider`: uses `google-generativeai` SDK
  ```python
  import google.generativeai as genai
  genai.configure(api_key=config.GEMINI_API_KEY)
  model = genai.GenerativeModel(config.GEMINI_MODEL)
  ```
- `OllamaProvider`: uses HTTP POST to Ollama API
  ```python
  response = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
      "model": config.OLLAMA_MODEL,  # e.g., "qwen2.5:0.5b"
      "messages": conversation_history,
      "stream": False
  })
  ```
- Factory: `create_llm_agent(provider: str) → LLMAgent`
- Detailed system prompt for customer service agent (configurable via env var or file)
- Conversation history stored in Redis (via existing session_manager)
- Timeout handling: if LLM doesn't respond in 5s, return a canned fallback message

**Test**: Send text to both providers, verify coherent responses.

---

### Step 5: Wire Everything Together + Barge-In + Events
**What**: Modify [server_multicall.py](file:///c:/Users/unify/freeswitch_voicebot/server_multicall.py) to use the new pipeline.

**The key transformation in `process_audio_segment()`:**

```python
# OLD flow (lines 211-284):
# audio → NC → STT (HTTP POST) → FlowEngine → play hardcoded .wav

# NEW flow:
def process_audio_segment(audio_data: bytes, call_uuid: str):
    # Barge-in check (MODIFIED)
    if response_handler.is_speaking(call_uuid):
        if config.ALLOW_INTERRUPTIONS:
            subprocess.run(["fs_cli", "-x", f"uuid_break {call_uuid}"], ...)
            response_handler.mark_stopped(call_uuid)
            emit_event(call_uuid, "barge_in", {"timestamp": time.time()})
        else:
            return
    
    tracker = LatencyTracker(call_uuid)
    
    # Step 1: Noise Cancellation (unchanged)
    tracker.mark("nc_start")
    enhanced_audio = noise_canceller.process_utterance(audio_data) if noise_canceller else audio_data
    tracker.mark("nc_end")
    
    # Step 2: STT (switchable provider)
    tracker.mark("stt_start")
    text = stt_handler.transcribe(enhanced_audio)
    tracker.mark("stt_end")
    emit_event(call_uuid, "transcription", {"text": text, "provider": config.STT_PROVIDER})
    
    if not text:
        return
    
    # Step 3: LLM (switchable provider)
    tracker.mark("llm_start")
    session_meta = session_manager.get_session(call_uuid) or {}
    history = session_meta.get("conversation_history", [])
    response_text = llm_agent.get_response(text, history)
    tracker.mark("llm_end")
    emit_event(call_uuid, "bot_response", {"text": response_text, "provider": config.LLM_PROVIDER})
    
    # Update conversation history
    history.append({"role": "user", "text": text})
    history.append({"role": "assistant", "text": response_text})
    session_meta["conversation_history"] = history
    session_manager.update_session(call_uuid, session_meta)
    
    # Step 4: TTS (streaming)
    tracker.mark("tts_start")
    if config.TTS_STREAMING:
        asyncio.run(tts_synthesizer.synthesize_stream(response_text, call_uuid))
    else:
        wav_path = tts_synthesizer.synthesize(response_text)
        response_handler.play_audio(call_uuid, wav_path, response_text)
    tracker.mark("tts_end")
    
    # Step 5: Log full latency breakdown + RTT
    summary = tracker.summary()
    emit_event(call_uuid, "turn_metrics", summary)
    logger.info(
        f"[{call_uuid}] ⏱️  Pipeline: NC={summary['nc_ms']:.0f}ms, "
        f"STT={summary['stt_ms']:.0f}ms [{config.STT_PROVIDER}], "
        f"LLM={summary['llm_ms']:.0f}ms [{config.LLM_PROVIDER}], "
        f"TTS={summary['tts_ms']:.0f}ms, "
        f"Total={summary['total_ms']:.0f}ms, "
        f"RTT={summary['rtt_ms']:.0f}ms"  # ← Round-trip time
    )
```

**RTT vs Total**: `total_ms` measures just the pipeline components. `rtt_ms` measures wall-clock time from when audio arrives on the WebSocket to when the first TTS audio byte is sent to FreeSWITCH — this catches hidden overhead like thread scheduling, queue delays, etc.

**Test**: Make a real call. Hear the AI respond. Check logs for latency breakdown. Test barge-in by interrupting the bot.

---

### Step 6: Polish & Edge Cases
**What**: Harden everything.

**Changes**:
- Empty/failed STT → respond with "Sorry, I didn't catch that. Could you repeat?"
- LLM timeout (5s) → "I'm having a moment, could you say that again?"
- TTS failure → play pre-recorded fallback WAV
- STT auto-fallback: if remote server fails 3 times, switch to local Whisper automatically
- Test 2 concurrent calls
- Write down actual latency numbers for each provider combination

---

## After Phase 1 — What Comes Next

**Phase 2** (Week 3-4): Deploy to GCP VM + Web Dashboard
- GCP account ($300 free credit)
- e2-standard-2 VM → `docker compose up` → voicebot is live
- Web dashboard subscribes to WebSocket events (the protocol we built in Step 5)

**Phase 3** (Week 5-6): Microservices + Kubernetes
- Split containers: FreeSWITCH, Voicebot, STT, TTS, Redis
- Kubernetes manifests + Helm chart
- HPA auto-scaling

**Phase 4** (Week 7-8): Advanced
- Gemini Multimodal Live API (voice-to-voice, no STT/TTS needed)
- Prometheus + Grafana monitoring (the event protocol feeds metrics)
- Deploy STT/Ollama on free GPU (Oracle, Colab, or Groq)

---

## What You Need Before We Start

1. **A Gemini API key** (if using Gemini) — [Google AI Studio](https://aistudio.google.com/), free, 30 seconds, no credit card.
2. **Ollama installed** (if using Ollama) — `docker run -d --name ollama ollama/ollama`, then `docker exec ollama ollama pull qwen2.5:0.5b`.
3. **That's it.** Your existing external STT server works as default. Edge-TTS needs no key.

---

## Verification Plan

### Phase 1 Verification (Local Development)
- [ ] Config loads all new env vars correctly
- [ ] `STT_PROVIDER=remote` → transcribes via your external server (existing behavior)
- [ ] `STT_PROVIDER=local` → transcribes via Faster-Whisper `small` model
- [ ] STT auto-fallback: remote fails → switches to local automatically
- [ ] Edge-TTS generates intelligible WAV files
- [ ] Edge-TTS streaming plays audio chunks seamlessly
- [ ] `LLM_PROVIDER=gemini` → returns conversational responses
- [ ] `LLM_PROVIDER=ollama` → returns conversational responses with Qwen
- [ ] End-to-end: SIP call → hear AI respond naturally
- [ ] Barge-in: interrupt bot mid-speech → bot stops and processes new input
- [ ] WebSocket events logged correctly for each turn
- [ ] Latency breakdown logged: NC, STT, LLM, TTS, Total, RTT
- [ ] RTT correctly captures overhead beyond component times
- [ ] 2 concurrent calls work without interference
- [ ] LLM timeout fallback works (disconnect network mid-call)
- [ ] TTS failure fallback works (corrupt text)
