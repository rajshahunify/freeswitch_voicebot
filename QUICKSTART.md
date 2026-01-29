# 🚀 Quick Start Guide

Get your voicebot running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd freeswitch_voicebot
./install.sh
```

Or manually:
```bash
pip install -r requirements.txt --break-system-packages
```

## Step 2: Configure Settings

Edit `config.py`:

```python
# Verify these settings match your FreeSWITCH setup
FREESWITCH_HOST = '127.0.0.1'
FREESWITCH_PORT = 8021
FREESWITCH_PASSWORD = 'ClueCon'  # Change if different

# Check STT endpoint is correct
STT_URL = "http://164.52.203.140:8890/transcribe"

# Verify audio file path
AUDIO_BASE_PATH = "/usr/local/freeswitch/sounds/custom"
```

## Step 3: Start the Server

**Option A: Development Mode** (recommended for testing)

```bash
# Terminal 1
python3 server.py

# Terminal 2
python3 agent.py
```

**Option B: Production Mode** (systemd services)

```bash
sudo cp voicebot-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voicebot-server voicebot-agent
```

## Step 4: Test a Call

1. Call your FreeSWITCH number
2. You should hear the welcome message
3. Say something like "hello" or "internet"
4. Bot should respond with appropriate audio

Watch the logs:
```bash
tail -f logs/voicebot.log
```

## Step 5: Verify It's Working

### Check Server Health
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "components": {
    "noise_canceller": "loaded",
    "vad_detector": "loaded",
    ...
  }
}
```

### Monitor Logs

You should see:
```
📞 NEW CALL STARTING
✓ Connected to call abc-123-def
🎤 Speech START (prob: 0.85)
🎯 STT: 'hello' (250ms, RTF: 0.25x)
🎯 Exact match: 'hello' → english_menu.wav
▶️  Playing 3.5s audio
```

## 🎛️ Common Adjustments

### Make VAD More/Less Sensitive

```python
# config.py
VAD_THRESHOLD = 0.3  # More sensitive (detects softer speech)
VAD_THRESHOLD = 0.7  # Less sensitive (clearer speech required)
```

### Allow User Interruptions

```python
# config.py
ALLOW_INTERRUPTIONS = True  # User can interrupt bot
```

### Change Noise Cancellation Strength

```python
# config.py
DF_ATTENUATION_LIMIT = 150  # Stronger (default: 100)
DF_POST_FILTER = False      # Faster processing
```

### Add New Response Keywords

```python
# config.py
INTENT_KEYWORDS = {
    "your_keyword": "your_audio_file.wav",
    # ...
}
```

Place `your_audio_file.wav` in `/usr/local/freeswitch/sounds/custom/`

## 🐛 Quick Troubleshooting

### "Cannot connect to FreeSWITCH"
```bash
sudo systemctl status freeswitch
fs_cli -x "status"
```

### "No audio processing"
```bash
# Check if audio fork is working
fs_cli -x "show calls"

# Verify WebSocket connection
tail -f logs/voicebot.log | grep "Connected to call"
```

### "STT not working"
```bash
# Test STT endpoint
curl -X POST http://164.52.203.140:8890/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test_audio.raw
```

### "Models not downloading"
```bash
# Pre-download models
python3 -c "
import torch
torch.hub.load('snakers4/silero-vad', 'silero_vad')
print('VAD model downloaded')
"

# DeepFilterNet downloads on first use
```

## 📊 Monitor Performance

```bash
# Real-time logs
tail -f logs/voicebot.log

# Filter for performance metrics
tail -f logs/voicebot.log | grep "Pipeline:"

# Get statistics
curl http://localhost:8000/stats | python3 -m json.tool
```

## 🎯 Next Steps

1. **Customize Intents**: Add your own keywords and audio files
2. **Tune Performance**: Adjust VAD and NC settings for your use case
3. **Monitor**: Set up log rotation and monitoring
4. **Scale**: Add more workers if handling many concurrent calls

## 📚 More Help

- Full documentation: `README.md`
- Configuration reference: `config.py` (inline comments)
- Logs location: `logs/voicebot.log`

---

**Need help?** Check the logs first:
```bash
tail -50 logs/voicebot.log
```

Most issues are configuration-related. Verify:
- FreeSWITCH is running
- Audio file paths are correct
- STT endpoint is accessible
- No firewall blocking ports
