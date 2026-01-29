# FreeSWITCH VoiceBot with Noise Cancellation and VAD

A production-ready voicebot system integrating FreeSWITCH with advanced audio processing, including DeepFilterNet2 noise cancellation and Silero VAD (Voice Activity Detection).

## 🎯 Features

- **🎤 Real-time Noise Cancellation**: DeepFilterNet2 removes background noise before processing
- **🗣️ Voice Activity Detection**: Silero VAD accurately detects speech segments
- **📞 Multi-call Support**: Handles multiple concurrent calls independently
- **🔄 Modular Architecture**: Easy to modify and extend components
- **📊 Performance Monitoring**: Comprehensive logging and timing metrics
- **🎛️ Configurable**: Single config file for all settings
- **🔌 IVR Integration**: Intent-based audio responses

## 📋 System Architecture

```
┌─────────────────┐
│   FreeSWITCH    │ ← Incoming calls
└────────┬────────┘
         │ (uuid_audio_fork)
         ↓
┌─────────────────┐
│  WebSocket      │ ← Audio stream (16kHz mono)
│  Server         │
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│        AUDIO PROCESSING PIPELINE        │
│                                         │
│  1. Noise Cancellation (DeepFilterNet2)│
│         ↓                               │
│  2. VAD Detection (Silero)             │
│         ↓                               │
│  3. Buffer Management                   │
│         ↓                               │
│  4. Speech-to-Text (Whisper API)       │
│         ↓                               │
│  5. Intent Matching                     │
│         ↓                               │
│  6. Audio Response Playback             │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
freeswitch_voicebot/
├── config.py                 # Central configuration
├── server.py                 # Main WebSocket server
├── agent.py                  # FreeSWITCH ESL handler
├── stt_handler.py           # Speech-to-text processing
│
├── audio_pipeline/          # Audio processing modules
│   ├── __init__.py
│   ├── noise_canceller.py   # DeepFilterNet2 wrapper
│   ├── vad_detector.py      # Silero VAD wrapper
│   └── audio_buffer.py      # Buffer management
│
├── ivr/                     # IVR logic
│   ├── __init__.py
│   ├── intent_matcher.py    # Keyword matching
│   └── response_handler.py  # Audio playback
│
├── logs/                    # Log files
├── models/                  # Downloaded AI models
├── requirements.txt         # Python dependencies
├── install.sh              # Installation script
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Ubuntu 24.04 (WSL or native)
- FreeSWITCH installed and running
- Python 3.10+
- 2GB+ RAM
- (Optional) CUDA for GPU acceleration

### Quick Install

```bash
# Clone or extract the project
cd freeswitch_voicebot

# Run installation script
chmod +x install.sh
./install.sh

# Or manual installation:
pip install -r requirements.txt --break-system-packages
```

### Verify Installation

```bash
# Check if models download correctly
python3 -c "from audio_pipeline import get_noise_canceller, get_vad_detector"

# Test configuration
python3 -c "import config; print('Config OK')"
```

## ⚙️ Configuration

Edit `config.py` to customize:

### FreeSWITCH Settings
```python
FREESWITCH_HOST = '127.0.0.1'
FREESWITCH_PORT = 8021
FREESWITCH_PASSWORD = 'ClueCon'
```

### Audio Processing
```python
# Noise Cancellation
DF_MODEL = 'DeepFilterNet2'  # or 'DeepFilterNet3'
DF_USE_GPU = False           # Set True if CUDA available

# VAD Settings
VAD_THRESHOLD = 0.5          # Speech probability (0.0-1.0)
VAD_MIN_SPEECH_DURATION_MS = 250
VAD_MIN_SILENCE_DURATION_MS = 300
```

### IVR Responses
```python
INTENT_KEYWORDS = {
    "hello": "english_menu.wav",
    "internet": "internet_inquiries.wav",
    "payment": "payment_options.wav",
    # Add your own mappings...
}
```

## 🏃 Running

### Method 1: Manual Start

```bash
# Terminal 1: Start WebSocket server
python3 server.py

# Terminal 2: Start FreeSWITCH agent
python3 agent.py
```

### Method 2: System Service

```bash
# Install services
sudo cp voicebot-*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl enable --now voicebot-server
sudo systemctl enable --now voicebot-agent

# Check status
sudo systemctl status voicebot-server
sudo systemctl status voicebot-agent

# View logs
sudo journalctl -u voicebot-server -f
```

## 📊 Monitoring

### Log Files
```bash
# View real-time logs
tail -f logs/voicebot.log

# Search for errors
grep ERROR logs/voicebot.log
```

### Health Check
```bash
# Check server health
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats
```

### Performance Metrics

The system logs detailed timing information:

```
⏱️  Pipeline: NC=35ms, STT=250ms, Intent=2ms, Response=15ms, Total=302ms
```

- **NC**: Noise Cancellation time
- **STT**: Speech-to-Text transcription time
- **Intent**: Intent matching time
- **Response**: Audio playback setup time

## 🔧 Customization

### Adding New Intents

Edit `config.py`:

```python
INTENT_KEYWORDS = {
    "new_keyword": "new_audio_file.wav",
    # ...
}
```

Place audio files in: `/usr/local/freeswitch/sounds/custom/`

### Changing VAD Sensitivity

In `config.py`:

```python
# More sensitive (detect softer speech)
VAD_THRESHOLD = 0.3

# Less sensitive (require clearer speech)
VAD_THRESHOLD = 0.7
```

### Adjusting Noise Cancellation

```python
# Stronger noise reduction
DF_ATTENUATION_LIMIT = 150  # dB

# Lighter processing (faster)
DF_POST_FILTER = False
```

### Using Different STT Service

Edit `stt_handler.py` or implement your own STT class.

## 🐛 Troubleshooting

### Issue: "Cannot connect to FreeSWITCH"

**Solution:**
```bash
# Check if FreeSWITCH is running
sudo systemctl status freeswitch

# Test ESL connection
fs_cli -x "status"

# Check ESL password in config.py matches FreeSWITCH
```

### Issue: "Model download fails"

**Solution:**
```bash
# Ensure internet connection
ping google.com

# Download manually
python3 -c "
import torch
torch.hub.load('snakers4/silero-vad', 'silero_vad')
"
```

### Issue: "Audio not processing"

**Solution:**
```bash
# Check WebSocket connection
tail -f logs/voicebot.log | grep "NEW CALL"

# Verify FreeSWITCH audio fork
fs_cli -x "show calls"

# Test STT endpoint
curl -X POST http://164.52.203.140:8890/transcribe
```

### Issue: "High latency"

**Solutions:**
- Enable GPU: `DF_USE_GPU = True`
- Use smaller model: `DF_MODEL = 'DeepFilterNet2'`
- Reduce VAD window: `VAD_WINDOW_SIZE = 512`
- Increase worker threads: `MAX_WORKERS = 8`

## 📈 Performance Optimization

### For WSL

```bash
# Increase WSL memory (in .wslconfig)
[wsl2]
memory=4GB
processors=4
```

### For GPU Acceleration

```bash
# Install CUDA toolkit
# Then in config.py:
DF_USE_GPU = True
```

### For Lower Latency

```python
# config.py
VAD_MIN_SILENCE_DURATION_MS = 200  # Faster response
CHUNK_DURATION_MS = 20             # Smaller chunks
```

## 🧪 Testing

### Test Individual Components

```python
# Test noise cancellation
from audio_pipeline import denoise_audio
denoised = denoise_audio(audio_bytes)

# Test VAD
from audio_pipeline import detect_speech
is_speech, prob = detect_speech(audio_chunk)

# Test intent matching
from ivr import IntentMatcher
matcher = IntentMatcher(config.INTENT_KEYWORDS)
intent = matcher.match_intent("hello")
```

### Load Testing

Use multiple concurrent calls to test stability:

```bash
# Generate test calls (requires SIP client)
sipp -sn uac -d 10000 -s <destination> <freeswitch_ip>
```

## 📚 Technical Details

### Audio Pipeline Processing

1. **Input**: 16kHz mono PCM audio (int16)
2. **Noise Cancellation**: 32ms chunks processed by DeepFilterNet2
3. **VAD Detection**: 32ms windows analyzed by Silero
4. **Buffering**: Speech accumulated until VAD detects end
5. **STT**: Complete utterance sent to Whisper API
6. **Response**: Intent matched and audio played

### Concurrency Model

- **FastAPI WebSocket**: Async I/O for multiple connections
- **ThreadPoolExecutor**: CPU-intensive tasks (NC, VAD, STT)
- **Per-call State**: Each call has independent buffers and state
- **Thread-safe**: All shared components are thread-safe

### Memory Usage

- **Baseline**: ~500MB (models loaded)
- **Per-call overhead**: ~10-20MB
- **Peak usage**: Depends on concurrent calls

## 🔐 Security Considerations

- FreeSWITCH ESL password should be changed from default
- Consider firewall rules for WebSocket port
- Validate input in production deployments
- Use HTTPS/WSS for production

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Add tests for new features
- Update documentation

## 📧 Support

For issues and questions:
- Check logs: `logs/voicebot.log`
- Review FreeSWITCH logs: `/var/log/freeswitch/`
- GitHub Issues: [Your Repo]

## 🙏 Acknowledgments

- **DeepFilterNet**: Noise suppression by Schröter et al.
- **Silero VAD**: Voice activity detection by Silero Team
- **FreeSWITCH**: Open-source telephony platform
- **FastAPI**: Modern web framework

---

**Version**: 1.0.0  
**Last Updated**: January 2026
