# Changelog

All notable changes to the FreeSWITCH VoiceBot project.

## [1.1.0] — 2026-06-03

### 🤖 Conversational LLM Engine (Replaces Legacy IVR Flow)
- **Multi-Provider LLM Integration**: Support for Google Gemini (`gemini-2.0-flash`), Groq (`llama-3.3-70b-versatile`), and local Ollama (`qwen2.5:0.5b` or custom models).
- **Proactive Fallback & Health Checks**: Automatically falls back from Gemini to Groq/Ollama on API quota exhaustion or rate limits. Background loop checks availability every 60 seconds (or 30 minutes for daily limits) and reverts to Gemini when available.
- **Concise & Voice-Optimized System Prompt**: System instructions ensure response length stays under 3 sentences with natural, conversational voice formatting (stripping Markdown, links, and code formatting).

### 🗣️ Streaming & Low-Latency Text-to-Speech (TTS)
- **Edge-TTS Synthesizer**: Uses high-quality, free Edge-TTS voices (e.g., `en-US-AvaMultilingualNeural` or `en-US-GuyNeural`) requiring no API keys.
- **Concurrent Streaming Pipeline**: Streams tokens from LLM, aggregates them into sentences, and submits them to a thread pool for TTS synthesis in real-time.
- **ffmpeg Conversion**: Converts MP3 to WAV format in ~50ms using `ffmpeg` subprocess with fallback to `torchaudio` if needed.
- **Redis & Local Disk Cache**: Caches synthesized WAV files based on voice-text hashes, bypassing TTS synthesis entirely for repeat phrases.

### 🛑 Real-Time Barge-In & Pipeline Optimizations
- **Barge-In (User Interruption)**: If the user speaks during bot playback, the VAD detects speech start, interrupts playout via `uuid_break <uuid> all`, cancels any queued TTS synthesis futures, and processes the new utterance immediately.
- **Conversational VAD Defaults**: Lowered default `VAD_MIN_SILENCE_DURATION_MS` to `800` (down from `1500`) and raised `VAD_THRESHOLD` to `0.5` (up from `0.3`) for faster speech-end detection and background noise filtering.
- **Latency Tracker & Performance Logs**: Tracks and outputs metrics for each pipeline stage (`NC`, `STT`, `LLM`, `TTS`, `Playback RTT`). Logs actual provider used (`[groq]`, `[gemini]`, `[ollama]`) rather than static configuration variables.
- **10s Dialplan Delay Removed**: Fixed dialplan and Docker entrypoint configurations to remove the 10-second startup ring sleep for instant call pickup.

## [1.0.0] — 2026-05-22

### 🏗️ Architecture
- **All-in-one Docker image**: FreeSWITCH + Redis + Python WebSocket Server + ESL Agent managed by `supervisord` in a single container
- **Dynamic `entrypoint.sh`**: Container configuration (external IP, dial extension, WebSocket URL) is now fully configurable via Docker environment variables — zero code changes needed per deployment
- **Three compose files**: `docker-compose.yml` (bridge/production), `docker-compose.dev.yml` (local dev with volume mounts), `docker-compose.host.yml` (Linux host networking)

### 🎙️ Audio Processing Pipeline
- **Silero VAD** (Voice Activity Detection): Per-call isolated instances process raw 32ms chunks in real-time (<1ms latency)
- **AudioBuffer**: Accumulates speech chunks and releases complete utterances on speech-end detection
- **DeepFilterNet2** (Noise Cancellation): Full-utterance neural denoising with 48kHz resampling — runs only on buffered speech segments for CPU efficiency
- **Whisper STT**: HTTP-based speech-to-text transcription

### 📞 IVR Flow Engine
- JSON-driven conversation flow with `choice`, `input`, `action`, and `end` step types
- Hybrid matching: fuzzy (fuzzywuzzy) → semantic (sentence-transformers) fallback
- Per-call session state via Redis

### 🐳 Docker & Deployment
- Custom `mod_audio_fork` FreeSWITCH module for WebSocket audio streaming
- Restricted RTP port range (16384-16394) for efficient bridge networking
- NAT-safe SIP profile with `local-network-acl=none` for Docker bridge compatibility
- Configurable dial extension via `VOICEBOT_EXTENSION` environment variable
- Works with any SIP-compliant softphone (Zoiper, Linphone, MicroSIP, hardware phones)

### 📁 Repository Structure
- Organized into `docs/`, `scripts/`, `audio_pipeline/`, `ivr/`, `docker/`
- Comprehensive README with architecture diagrams and end-to-end call flow
- Detailed debugging walkthrough documenting all 6 phases of Docker telephony troubleshooting
