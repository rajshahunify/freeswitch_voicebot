#!/bin/bash
# =============================================================================
# FreeSWITCH VoiceBot — Dynamic Entrypoint Script
# =============================================================================
# This script runs BEFORE supervisord starts. It dynamically patches
# FreeSWITCH XML configurations using Docker environment variables,
# making the image fully self-contained and configurable without
# volume mounts or source code changes.
#
# Configurable variables (set via docker-compose or docker run -e):
#   EXTERNAL_IP         — NAT SIP/RTP IP for SDP (default: 127.0.0.1)
#   VOICEBOT_EXTENSION  — Dial extension number for the voicebot (default: 5000)
#   WEBSOCKET_URL       — WebSocket URL for mod_audio_fork (default: ws://127.0.0.1:8000/media)
# =============================================================================

set -e

# ---- Defaults ---------------------------------------------------------------
EXTERNAL_IP="${EXTERNAL_IP:-127.0.0.1}"
VOICEBOT_EXTENSION="${VOICEBOT_EXTENSION:-5000}"
WEBSOCKET_URL="${WEBSOCKET_URL:-ws://127.0.0.1:8000/media}"

FS_CONF="/usr/local/freeswitch/conf"

echo "============================================================"
echo "  FreeSWITCH VoiceBot — Entrypoint Configuration"
echo "============================================================"
echo "  EXTERNAL_IP:        ${EXTERNAL_IP}"
echo "  VOICEBOT_EXTENSION: ${VOICEBOT_EXTENSION}"
echo "  WEBSOCKET_URL:      ${WEBSOCKET_URL}"
echo "============================================================"

# ---- Helper for safe sed on mounted files (prevents busy device errors) ------
safe_sed() {
    local pattern="$1"
    local file="$2"
    sed "$pattern" "$file" > "$file.tmp"
    cat "$file.tmp" > "$file"
    rm "$file.tmp"
}

# ---- 1. Patch vars.xml: external_rtp_ip / external_sip_ip ------------------
# The Dockerfile already converted the stun-set directives to $${env(EXTERNAL_IP)}.
# That works when supervisord passes env vars. But as a safety net, we also do
# a direct sed replacement so it works even if the env() preprocessor fails.
echo ">>> Patching vars.xml with EXTERNAL_IP=${EXTERNAL_IP}"
safe_sed "s|\(external_rtp_ip=\)[^\"]*|\1${EXTERNAL_IP}|g" "${FS_CONF}/vars.xml"
safe_sed "s|\(external_sip_ip=\)[^\"]*|\1${EXTERNAL_IP}|g" "${FS_CONF}/vars.xml"

# ---- 2. Patch dialplan: voicebot extension number ---------------------------
# Update the default context voicebot extension
echo ">>> Patching dialplan extension to ${VOICEBOT_EXTENSION}"
VOICEBOT_DIALPLAN="${FS_CONF}/dialplan/default/01_voicebot.xml"
if [ -f "${VOICEBOT_DIALPLAN}" ]; then
    safe_sed "s|expression=\"^[0-9]*$\"|expression=\"^${VOICEBOT_EXTENSION}$\"|g" "${VOICEBOT_DIALPLAN}"
fi

# Update the public context route
PUBLIC_ROUTE="${FS_CONF}/dialplan/public/01_voicebot_route.xml"
if [ -f "${PUBLIC_ROUTE}" ]; then
    safe_sed "s|expression=\"^[0-9]*$\"|expression=\"^${VOICEBOT_EXTENSION}$\"|g" "${PUBLIC_ROUTE}"
    safe_sed "s|data=\"[0-9]* XML default\"|data=\"${VOICEBOT_EXTENSION} XML default\"|g" "${PUBLIC_ROUTE}"
fi

# Ensure the built-in demo IVR doesn't conflict with our chosen extension
# (the Dockerfile already moved it to 9999, but if someone picks 9999 we move it to 9998)
DEFAULT_DIALPLAN="${FS_CONF}/dialplan/default.xml"
if [ -f "${DEFAULT_DIALPLAN}" ]; then
    if grep -q "expression=\"^${VOICEBOT_EXTENSION}$\"" "${DEFAULT_DIALPLAN}" 2>/dev/null; then
        echo ">>> Moving conflicting built-in demo IVR extension away from ${VOICEBOT_EXTENSION}"
        safe_sed "s|expression=\"^${VOICEBOT_EXTENSION}$\"|expression=\"^99999$\"|g" "${DEFAULT_DIALPLAN}"
    fi
fi

# ---- 3. Patch dialplan: WebSocket URL for mod_audio_fork --------------------
echo ">>> Patching audio_fork WebSocket URL to ${WEBSOCKET_URL}"
if [ -f "${VOICEBOT_DIALPLAN}" ]; then
    safe_sed "s|ws://[^\"]*|${WEBSOCKET_URL}|g" "${VOICEBOT_DIALPLAN}"
fi

echo "============================================================"
echo "  ✓ Configuration complete. Starting supervisord..."
echo "============================================================"

# ---- Hand off to supervisord ------------------------------------------------
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
