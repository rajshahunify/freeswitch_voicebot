#!/bin/bash

#===============================================================================
# FreeSWITCH VoiceBot - Installation Script
# For WSL Ubuntu 24.04
#===============================================================================

set -e  # Exit on error

echo "========================================"
echo "FreeSWITCH VoiceBot - Installation"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on WSL
if ! grep -qi microsoft /proc/version; then
    print_warning "Not running on WSL. This script is optimized for WSL."
fi

# Update system
print_status "Updating system packages..."
apt-get update

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    build-essential \
    git

# Install FreeSWITCH CLI tools (if not already installed)
if ! command -v fs_cli &> /dev/null; then
    print_warning "fs_cli not found. Please ensure FreeSWITCH is installed."
    print_warning "Installation guide: https://freeswitch.org/confluence/display/FREESWITCH/Installation"
else
    print_status "FreeSWITCH CLI found"
fi

# Create virtual environment (optional but recommended)
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python packages..."
pip install -r requirements.txt --break-system-packages || {
    print_warning "Installing without --break-system-packages..."
    pip install -r requirements.txt
}

# Download models (they will auto-download on first run, but we can pre-download)
print_status "Pre-downloading AI models..."
python3 -c "
import torch
import logging
logging.basicConfig(level=logging.INFO)

print('Downloading Silero VAD...')
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    print('✓ Silero VAD downloaded')
except Exception as e:
    print(f'Warning: Could not pre-download Silero VAD: {e}')

print('DeepFilterNet will download on first use (~200MB)')
"

# Create directories
print_status "Creating directories..."
mkdir -p logs
mkdir -p models

# Create systemd service file (optional)
print_status "Creating systemd service file..."
cat > voicebot-server.service << EOF
[Unit]
Description=FreeSWITCH VoiceBot Server
After=network.target freeswitch.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

cat > voicebot-agent.service << EOF
[Unit]
Description=FreeSWITCH VoiceBot Agent
After=network.target freeswitch.service voicebot-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python agent.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service files created (not installed)"

# Configuration check
print_status "Checking configuration..."
python3 -c "
import config
print(f'STT URL: {config.STT_URL}')
print(f'Audio Path: {config.AUDIO_BASE_PATH}')
print(f'WebSocket: {config.WS_HOST}:{config.WS_PORT}')
" || print_error "Configuration error - please check config.py"

# Test FreeSWITCH connection
print_status "Testing FreeSWITCH connection..."
fs_cli -x "status" > /dev/null 2>&1 && \
    print_status "FreeSWITCH is running" || \
    print_warning "FreeSWITCH is not running or not accessible"

echo ""
echo "========================================"
print_status "Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review config.py and adjust settings"
echo "  2. Ensure FreeSWITCH is running: systemctl status freeswitch"
echo "  3. Start the server: python3 server.py"
echo "  4. Start the agent: python3 agent.py"
echo ""
echo "Optional: Install as system service"
echo "  cp voicebot-*.service /etc/systemd/system/"
echo "  systemctl daemon-reload"
echo "  systemctl enable --now voicebot-server voicebot-agent"
echo ""
echo "Logs will be in: logs/voicebot.log"
echo ""
