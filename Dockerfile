# =============================================================================
# FreeSWITCH VoiceBot - All-in-One Docker Image
# Base: rajunify123/freeswitch-mod-audio-fork:v1 (Debian + FreeSWITCH + mod_audio_fork)
# Adds: Python3, Redis, supervisord, voicebot application code
# =============================================================================

FROM rajunify123/freeswitch-mod-audio-fork:v1

LABEL maintainer="unify" \
      description="FreeSWITCH VoiceBot with mod_audio_fork, Python AI pipeline, and Redis"

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# 1. SYSTEM DEPENDENCIES
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    supervisor \
    redis-server \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/supervisor

# =============================================================================
# 2. FREESWITCH CONFIGURATION OVERLAYS
# =============================================================================

# Dialplan: voicebot extension 5000 in default context
COPY docker/freeswitch-config/dialplan/default/01_voicebot.xml \
     /usr/local/freeswitch/conf/dialplan/default/01_voicebot.xml

# Dialplan: public context route for extension 5000
COPY docker/freeswitch-config/dialplan/public/01_voicebot_route.xml \
     /usr/local/freeswitch/conf/dialplan/public/01_voicebot_route.xml

# ESL event socket configuration
COPY docker/freeswitch-config/autoload_configs/event_socket.conf.xml \
     /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml

# Custom SIP Profile internal.xml (handles NAT and bridge network gateway ACL translation)
COPY docker/freeswitch-config/sip_profiles/internal.xml \
     /usr/local/freeswitch/conf/sip_profiles/internal.xml

# Ensure mod_audio_fork is enabled in modules.conf.xml
# (The base image should have it, but this guarantees it)
RUN if ! grep -q 'mod_audio_fork' /usr/local/freeswitch/conf/autoload_configs/modules.conf.xml; then \
      sed -i '/<\/modules>/i\    <load module="mod_audio_fork"/>' \
        /usr/local/freeswitch/conf/autoload_configs/modules.conf.xml; \
      echo ">>> Injected mod_audio_fork into modules.conf.xml"; \
    else \
      echo ">>> mod_audio_fork already present in modules.conf.xml"; \
    fi

# Ensure fs_cli is in PATH
RUN ln -sf /usr/local/freeswitch/bin/fs_cli /usr/local/bin/fs_cli || true

# Restrict RTP port range to 16384-16394 to make Docker bridge port mapping lightweight
RUN sed -i 's|<!-- <param name="rtp-start-port" value="16384"/> -->|<param name="rtp-start-port" value="16384"/>|' /usr/local/freeswitch/conf/autoload_configs/switch.conf.xml && \
    sed -i 's|<!-- <param name="rtp-end-port" value="32768"/> -->|<param name="rtp-end-port" value="16394"/>|' /usr/local/freeswitch/conf/autoload_configs/switch.conf.xml

# Rename conflicting default ivr_demo destination number from 5000 to 9999
# Also remove the sleep(10000) in the default_password warning block — it delays every call by 10s
RUN sed -i 's|expression="^5000$"|expression="^9999$"|g' /usr/local/freeswitch/conf/dialplan/default.xml && \
    sed -i '/sleep.*10000/d' /usr/local/freeswitch/conf/dialplan/default.xml

# Configure FreeSWITCH to use the EXTERNAL_IP environment variable for NAT SIP and RTP IPs
RUN sed -i 's|<X-PRE-PROCESS cmd="stun-set" data="external_rtp_ip=stun:stun.freeswitch.org"/>|<X-PRE-PROCESS cmd="set" data="external_rtp_ip=$${env(EXTERNAL_IP)}"/>|g' /usr/local/freeswitch/conf/vars.xml && \
    sed -i 's|<X-PRE-PROCESS cmd="stun-set" data="external_sip_ip=stun:stun.freeswitch.org"/>|<X-PRE-PROCESS cmd="set" data="external_sip_ip=$${env(EXTERNAL_IP)}"/>|g' /usr/local/freeswitch/conf/vars.xml



# FreeSWITCH compiled default looks for config at /usr/local/freeswitch/etc/freeswitch/
# but the base image has it at /usr/local/freeswitch/conf/. Create symlink.
RUN mkdir -p /usr/local/freeswitch/etc && \
    ln -sf /usr/local/freeswitch/conf /usr/local/freeswitch/etc/freeswitch && \
    mkdir -p /usr/local/freeswitch/log /usr/local/freeswitch/db /usr/local/freeswitch/run

# =============================================================================
# 3. CUSTOM SOUND FILES
# =============================================================================
COPY sounds/ /usr/local/freeswitch/sounds/custom/

# =============================================================================
# 4. PYTHON APPLICATION
# =============================================================================
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download Silero VAD model to avoid first-call latency
RUN python3 -c "\
import torch; \
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False); \
print('✓ Silero VAD model pre-cached')" \
    || echo "⚠ VAD model pre-download skipped (will download on first call)"

# Copy application code
COPY config.py agent.py server_multicall.py stt_handler.py session_manager.py ./
COPY latency_tracker.py event_emitter.py ./
COPY audio_pipeline/ ./audio_pipeline/
COPY ivr/ ./ivr/

# Create runtime directories (tts_cache is mounted as a volume)
RUN mkdir -p logs models debug_audio tts_cache

# =============================================================================
# 5. SUPERVISOR CONFIGURATION
# =============================================================================
COPY docker/supervisord.conf /etc/supervisor/conf.d/voicebot.conf

# =============================================================================
# 5b. ENTRYPOINT SCRIPT (dynamic config injection at startup)
# =============================================================================
COPY entrypoint.sh /entrypoint.sh
# Fix Windows \r\n line endings → Linux \n (prevents "exec: no such file or directory")
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# =============================================================================
# 6. PORTS
# =============================================================================
# 5060  = SIP signaling (UDP + TCP)
# 5080  = SIP external profile
# 8000  = WebSocket server (Python)
# 8021  = FreeSWITCH ESL
EXPOSE 5060/udp 5060/tcp 5080/udp 5080/tcp 8000 8021

# =============================================================================
# 7. ENVIRONMENT DEFAULTS
# =============================================================================
ENV FREESWITCH_HOST=127.0.0.1 \
    FREESWITCH_PORT=8021 \
    FREESWITCH_PASSWORD=ClueCon \
    WEBSOCKET_URL=ws://127.0.0.1:8000/media \
    STT_URL=http://164.52.203.140:8890/transcribe \
    WS_PORT=8000 \
    LOG_LEVEL=INFO \
    REDIS_HOST=127.0.0.1 \
    REDIS_PORT=6379 \
    REDIS_REQUIRED=false \
    MAX_CONCURRENT_CALLS=5 \
    DF_USE_GPU=false \
    NC_ENABLED=false \
    EXTERNAL_IP=127.0.0.1 \
    VOICEBOT_EXTENSION=5000 \
    STT_PROVIDER=remote \
    LLM_PROVIDER=gemini \
    GEMINI_MODEL=gemini-2.0-flash \
    OLLAMA_URL=http://ollama:11434 \
    OLLAMA_MODEL=qwen2.5:0.5b \
    TTS_VOICE=en-US-GuyNeural \
    TTS_CACHE_DIR=/app/tts_cache \
    ALLOW_INTERRUPTIONS=true \
    STT_LOCAL_MODEL=small \
    STT_LOCAL_DEVICE=cpu

# =============================================================================
# 8. ENTRYPOINT
# =============================================================================
# entrypoint.sh dynamically patches FreeSWITCH XML configs using env vars
# (EXTERNAL_IP, VOICEBOT_EXTENSION, WEBSOCKET_URL) then starts supervisord.
ENTRYPOINT ["/entrypoint.sh"]
