#!/bin/bash
# =============================================================================
# FreeSWITCH VoiceBot — GPU Server Readiness Check
# Run this on the server:  ssh user@10.32.28.29 'bash -s' < scripts/check_server.sh
# Or copy to server and run:  chmod +x check_server.sh && ./check_server.sh
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASS="${GREEN}✅ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"
WARN="${YELLOW}⚠️  WARN${NC}"
INFO="${CYAN}ℹ️  INFO${NC}"

echo ""
echo "============================================================"
echo "  FreeSWITCH VoiceBot — Server Readiness Check"
echo "  Server: $(hostname -I | awk '{print $1}') | $(date)"
echo "============================================================"

# =============================================================================
# 1. OPERATING SYSTEM
# =============================================================================
echo ""
echo -e "${CYAN}[1/7] SYSTEM INFO${NC}"
echo "  OS      : $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "  Kernel  : $(uname -r)"
echo "  Uptime  : $(uptime -p)"
echo "  CPU     : $(nproc) cores | $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
FREE_MEM=$(free -h | awk '/^Mem:/ {print $7}')
TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
echo "  Memory  : $FREE_MEM free of $TOTAL_MEM"
FREE_DISK=$(df -h / | awk 'NR==2 {print $4}')
echo "  Disk    : $FREE_DISK free on /"

# =============================================================================
# 2. GPU CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[2/7] GPU CHECK${NC}"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  $PASS NVIDIA GPU detected"
    echo "  GPU     : $GPU_NAME"
    echo "  VRAM    : $GPU_MEM_FREE free of $GPU_MEM_TOTAL"
    echo "  Util    : $GPU_UTIL"
    
    # Check CUDA
    if command -v nvcc &>/dev/null; then
        echo -e "  $PASS CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ,)"
    else
        echo -e "  ${WARN} nvcc not found (CUDA toolkit not installed — Docker can still use GPU)"
    fi
    
    # Check Docker GPU support
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo -e "  $PASS Docker NVIDIA runtime detected"
    else
        echo -e "  ${WARN} Docker NVIDIA runtime NOT found. Run: sudo apt install nvidia-container-toolkit"
    fi
else
    echo -e "  ${WARN} nvidia-smi not found — no GPU detected or drivers not installed"
fi

# =============================================================================
# 3. DOCKER CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[3/7] DOCKER CHECK${NC}"
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ,)
    echo -e "  $PASS Docker installed: $DOCKER_VER"
    
    if docker info &>/dev/null; then
        echo -e "  $PASS Docker daemon running"
    else
        echo -e "  $FAIL Docker daemon NOT running — run: sudo systemctl start docker"
    fi
    
    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        COMPOSE_VER=$(docker compose version --short 2>/dev/null || echo "unknown")
        echo -e "  $PASS Docker Compose: $COMPOSE_VER"
    else
        echo -e "  $FAIL Docker Compose not found"
    fi
else
    echo -e "  $FAIL Docker NOT installed"
fi

# =============================================================================
# 4. OLLAMA CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[4/7] OLLAMA CHECK${NC}"

# Check if ollama process is running
if pgrep -x ollama &>/dev/null; then
    echo -e "  $PASS Ollama process is running (PID: $(pgrep -x ollama))"
elif systemctl is-active --quiet ollama 2>/dev/null; then
    echo -e "  $PASS Ollama systemd service is active"
else
    echo -e "  ${WARN} Ollama is NOT running (or not installed)"
fi

# Check if ollama binary exists
if command -v ollama &>/dev/null; then
    echo -e "  $PASS Ollama binary found: $(which ollama)"
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    echo "  Version : $OLLAMA_VER"
else
    echo -e "  ${WARN} Ollama binary not in PATH"
fi

# Test Ollama API on default port
echo ""
echo "  Testing Ollama API endpoints:"
for port in 11434; do
    if curl -sf --max-time 3 "http://localhost:$port/api/tags" > /tmp/ollama_tags.json 2>/dev/null; then
        echo -e "  $PASS Ollama API responding on port $port"
        MODEL_COUNT=$(cat /tmp/ollama_tags.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "?")
        echo "  Models  : $MODEL_COUNT model(s) already pulled"
        if [ "$MODEL_COUNT" -gt 0 ] 2>/dev/null; then
            cat /tmp/ollama_tags.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('models', []):
    size_gb = m.get('size', 0) / 1e9
    print(f'    - {m[\"name\"]} ({size_gb:.1f} GB)')
" 2>/dev/null
        fi
    else
        echo -e "  $FAIL Ollama API NOT responding on port $port"
        echo "       To start Ollama: ollama serve &"
        echo "       Or install: curl -fsSL https://ollama.com/install.sh | sh"
    fi
done

# Check if Ollama is bound to all interfaces (needed for Docker access)
echo ""
OLLAMA_LISTEN=$(ss -tlnp 2>/dev/null | grep 11434 | awk '{print $4}')
if echo "$OLLAMA_LISTEN" | grep -qE "^0\.0\.0\.0|^\*|^\[::\]"; then
    echo -e "  $PASS Ollama listening on all interfaces ($OLLAMA_LISTEN) — Docker can reach it"
elif echo "$OLLAMA_LISTEN" | grep -q "127.0.0.1"; then
    echo -e "  ${WARN} Ollama bound to 127.0.0.1 only — Docker containers CANNOT reach it!"
    echo "       Fix: Set OLLAMA_HOST=0.0.0.0 before starting ollama serve"
    echo "       Or:  export OLLAMA_HOST=0.0.0.0 && ollama serve"
elif [ -z "$OLLAMA_LISTEN" ]; then
    echo -e "  ${INFO} Could not detect Ollama bind address (may not be running)"
else
    echo "  Ollama listening on: $OLLAMA_LISTEN"
fi

# =============================================================================
# 5. REQUIRED PORTS CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[5/7] PORT AVAILABILITY CHECK${NC}"
echo "  (Checking if ports needed by voicebot are FREE)"
echo ""

declare -A PORT_NAMES=(
    [5060]="SIP (FreeSWITCH)"
    [5080]="SIP Outbound (FreeSWITCH)"
    [8000]="WebSocket (VoiceBot Python)"
    [8021]="ESL (FreeSWITCH Event Socket)"
    [6379]="Redis"
    [11434]="Ollama LLM API"
)

ALL_PORTS_OK=true
for port in 5060 5080 8000 8021 6379 11434; do
    NAME="${PORT_NAMES[$port]}"
    if ss -tlnp 2>/dev/null | grep -q ":$port " || ss -ulnp 2>/dev/null | grep -q ":$port "; then
        PROCESS=$(ss -tlnp 2>/dev/null | grep ":$port " | awk '{print $NF}' | head -1)
        echo -e "  ${WARN} Port $port ($NAME) — IN USE by: $PROCESS"
        ALL_PORTS_OK=false
    else
        echo -e "  $PASS Port $port ($NAME) — FREE"
    fi
done

if [ "$ALL_PORTS_OK" = true ]; then
    echo ""
    echo -e "  ${GREEN}All required ports are available!${NC}"
fi

# Check RTP port range
echo ""
echo "  Checking RTP port range (16384-16394):"
RTP_USED=$(ss -ulnp 2>/dev/null | awk -F: '{print $2}' | awk '{print $1}' | grep -E '^1638[4-9]$|^1639[0-4]$' | wc -l)
if [ "$RTP_USED" -gt 0 ]; then
    echo -e "  ${WARN} $RTP_USED RTP ports in use (may cause audio issues)"
else
    echo -e "  $PASS RTP ports 16384-16394 all FREE"
fi

# =============================================================================
# 6. NETWORK CONNECTIVITY CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[6/7] NETWORK CONNECTIVITY CHECK${NC}"
echo "  (Testing external services the voicebot uses)"
echo ""

# STT Server
echo -n "  STT Server (164.52.203.140:8890) ... "
if curl -sf --max-time 5 "http://164.52.203.140:8890/transcribe" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|400|422"; then
    echo -e "$PASS reachable"
elif curl -sf --max-time 5 "http://164.52.203.140:8890/" -o /dev/null 2>/dev/null; then
    echo -e "$PASS reachable (got response)"
else
    # Try TCP connect
    if timeout 5 bash -c 'cat < /dev/null > /dev/tcp/164.52.203.140/8890' 2>/dev/null; then
        echo -e "${WARN} TCP port open but HTTP check failed"
    else
        echo -e "$FAIL NOT reachable (check firewall/VPN)"
    fi
fi

# Gemini API
echo -n "  Gemini API (generativelanguage.googleapis.com) ... "
if curl -sf --max-time 5 "https://generativelanguage.googleapis.com" -o /dev/null 2>/dev/null; then
    echo -e "$PASS reachable"
else
    echo -e "${WARN} Not reachable (needs internet access)"
fi

# Groq API
echo -n "  Groq API (api.groq.com) ... "
if curl -sf --max-time 5 "https://api.groq.com" -o /dev/null 2>/dev/null; then
    echo -e "$PASS reachable"
else
    echo -e "${WARN} Not reachable (needs internet access)"
fi

# Microsoft Edge TTS
echo -n "  Edge TTS (speech.microsoft.com) ... "
if curl -sf --max-time 5 "https://speech.microsoft.com" -o /dev/null 2>/dev/null; then
    echo -e "$PASS reachable"
else
    echo -e "${WARN} Not reachable (TTS may fail)"
fi

# =============================================================================
# 7. FIREWALL CHECK
# =============================================================================
echo ""
echo -e "${CYAN}[7/7] FIREWALL STATUS${NC}"
if command -v ufw &>/dev/null; then
    UFW_STATUS=$(ufw status 2>/dev/null | head -1)
    echo "  UFW: $UFW_STATUS"
    if echo "$UFW_STATUS" | grep -q "active"; then
        echo -e "  ${WARN} UFW is active — ensure ports 5060, 5080, 8000, 8021 are allowed:"
        echo "       sudo ufw allow 5060/udp && sudo ufw allow 5060/tcp"
        echo "       sudo ufw allow 5080/udp && sudo ufw allow 5080/tcp"
        echo "       sudo ufw allow 8000/tcp && sudo ufw allow 8021/tcp"
        echo "       sudo ufw allow 16384:16394/udp"
    fi
elif command -v firewall-cmd &>/dev/null; then
    FW_STATE=$(firewall-cmd --state 2>/dev/null)
    echo "  firewalld: $FW_STATE"
    if [ "$FW_STATE" = "running" ]; then
        echo -e "  ${WARN} firewalld is active — check that required ports are open"
    fi
else
    echo -e "  $INFO No UFW or firewalld detected (iptables may still be active)"
    iptables -L INPUT -n --line-numbers 2>/dev/null | head -20
fi

# =============================================================================
# SUMMARY & RECOMMENDED .env SETTINGS
# =============================================================================
echo ""
echo "============================================================"
echo "  SUMMARY & RECOMMENDED SETTINGS FOR THIS SERVER"
echo "============================================================"
SERVER_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "  Server IP: $SERVER_IP"
echo ""
echo "  Add/update these settings in your .env file:"
echo ""
echo "  # Point to this server's IP (not host.docker.internal)"
echo "  OLLAMA_URL=http://${SERVER_IP}:11434"
echo ""
echo "  # For GPU-accelerated noise cancellation:"
echo "  DF_USE_GPU=true"
echo "  STT_LOCAL_DEVICE=cuda"
echo ""
echo "  # In docker-compose.yml, set EXTERNAL_IP to this server's IP"
echo "  # EXTERNAL_IP=${SERVER_IP}"
echo ""
if command -v nvidia-smi &>/dev/null; then
    echo "  # GPU detected — for local STT with GPU:"
    echo "  STT_PROVIDER=local"
    echo "  STT_LOCAL_MODEL=medium   # or large-v3 for best accuracy"
    echo "  STT_LOCAL_DEVICE=cuda"
    echo "  STT_LOCAL_COMPUTE_TYPE=float16"
    echo ""
fi
echo "  To start Ollama visible to Docker containers:"
echo "  export OLLAMA_HOST=0.0.0.0 && ollama serve"
echo ""
echo "============================================================"
