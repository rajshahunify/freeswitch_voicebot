# Changelog

All notable changes to the FreeSWITCH VoiceBot project.

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
