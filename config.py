"""
FreeSWITCH VoiceBot Configuration
Centralized configuration for all components
"""

import os

# =============================================================================
# FREESWITCH CONFIGURATION
# =============================================================================
FREESWITCH_HOST = os.getenv("FREESWITCH_HOST", '127.0.0.1')
FREESWITCH_PORT = int(os.getenv("FREESWITCH_PORT", 8021))
FREESWITCH_PASSWORD = os.getenv("FREESWITCH_PASSWORD", 'ClueCon')

# WebSocket URL where FreeSWITCH sends audio
# FreeSWITCH runs INSIDE Docker, so it cannot use 127.0.0.1 to reach the host.
# Choose the correct option for your network setup:
#   Option A (Linux, bridge network): ws://172.17.0.1:8000/media
#   Option B (Docker Desktop Mac/Win): ws://host.docker.internal:8000/media
#   Option C (--network host mode):   ws://127.0.0.1:8000/media
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://172.17.0.1:8000/media")  # Linux bridge (default)

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000  # Hz - DO NOT CHANGE (required for DF2, Silero, Whisper)
CHANNELS = 1  # Mono
BIT_DEPTH = 16  # bits
CHUNK_DURATION_MS = 32  # milliseconds (512 samples at 16kHz)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 512 samples

# =============================================================================
# STT (SPEECH-TO-TEXT) CONFIGURATION
# =============================================================================
# Provider: "remote" (HTTP API) or "local" (Faster-Whisper)
STT_PROVIDER = os.getenv("STT_PROVIDER", "remote")

# Remote STT (existing external Faster-Whisper server)
STT_URL = os.getenv("STT_REMOTE_URL", os.getenv("STT_URL", "http://164.52.203.140:8890/transcribe"))
STT_PARAMS = {
    "sid": "live",
    "sample_rate": SAMPLE_RATE,
    "bit_depth": f"int{BIT_DEPTH}",
    "language": "en"
}
STT_TIMEOUT = 5  # seconds

# Local STT (Faster-Whisper) — only loaded when STT_PROVIDER=local
STT_LOCAL_MODEL = os.getenv("STT_LOCAL_MODEL", "large-v3") # tiny, base, small, medium, large-v3
STT_LOCAL_DEVICE = os.getenv("STT_LOCAL_DEVICE", "cpu")        # "cpu" or "cuda"
STT_LOCAL_COMPUTE_TYPE = os.getenv("STT_LOCAL_COMPUTE_TYPE", "int8")  # int8 for CPU, float16 for GPU
STT_FALLBACK_ENABLED = os.getenv("STT_FALLBACK_ENABLED", "true").lower() == "true"
STT_FALLBACK_THRESHOLD = int(os.getenv("STT_FALLBACK_THRESHOLD", 3))  # consecutive failures before fallback

# =============================================================================
# LLM (LARGE LANGUAGE MODEL) CONFIGURATION
# =============================================================================
# Provider: "gemini", "groq", or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# Groq configuration (fast cloud LLM — free tier, ~200ms latency)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# LLM behavior
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 10))  # seconds before fallback
LLM_MAX_HISTORY = int(os.getenv("LLM_MAX_HISTORY", 20))  # max conversation turns to keep
LLM_SYSTEM_PROMPT_FILE = os.getenv("LLM_SYSTEM_PROMPT_FILE", "")  # optional: path to custom system prompt
LLM_FALLBACK_ENABLED = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
LLM_FALLBACK_THRESHOLD = int(os.getenv("LLM_FALLBACK_THRESHOLD", 1))

# =============================================================================
# TTS (TEXT-TO-SPEECH) CONFIGURATION
# =============================================================================
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
TTS_CACHE_DIR = os.getenv("TTS_CACHE_DIR", "/app/tts_cache")
TTS_FALLBACK_WAV = os.getenv("TTS_FALLBACK_WAV", "ivr/sorry_try_again.wav")  # played on TTS failure

# =============================================================================
# DEEPFILTERNET2 (NOISE CANCELLATION) CONFIGURATION
# =============================================================================
# Model options: 'DeepFilterNet2' or 'DeepFilterNet3'
DF_MODEL = 'DeepFilterNet2'

# Processing settings
DF_FRAME_SIZE = CHUNK_SIZE  # Process in 32ms chunks
DF_COMPENSATE_DELAY = True  # Compensate for processing delay
DF_ATTENUATION_LIMIT = 6.0  # dB - REDUCED from 100 to preserve voice quality

# Set to False to skip DeepFilterNet entirely (no git, no torchaudio needed for NC)
# Audio goes straight to STT — good for CPU-only deployments
NC_ENABLED = os.getenv("NC_ENABLED", "true").lower() == "true"

# Performance
DF_USE_GPU = os.getenv("DF_USE_GPU", "False").lower() == "true"
DF_POST_FILTER = False  # Additional perceptual enhancement (disabled for speed)

# Volume boost to compensate for NC reducing amplitude
DF_GAIN = 4.0  # 4x volume boost after noise cancellation

# Debug settings - Enable to save before/after audio files
DF_DEBUG_RMS = True  # Log RMS values to see volume changes
DF_DEBUG_SAVE_DIR = "debug_audio"  # Directory to save debug audio files (set to None to disable)

# =============================================================================
# SILERO VAD (VOICE ACTIVITY DETECTION) CONFIGURATION
# =============================================================================
# Thresholds
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))  # Speech probability threshold (0.0-1.0). 0.5 = balanced, 0.3 = too sensitive
VAD_MIN_SPEECH_DURATION_MS = 200  # Minimum speech duration to consider (reduced for faster start)
VAD_MIN_SILENCE_DURATION_MS = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", 800))  # Silence before speech end (800ms = conversational)
VAD_SPEECH_PAD_MS = 30  # Padding before/after speech

# Window size for VAD (512 or 1024 samples recommended)
VAD_WINDOW_SIZE = 512  # 32ms at 16kHz
VAD_SAMPLE_RATE = SAMPLE_RATE  # Must be 8000 or 16000

# =============================================================================
# BUFFER MANAGEMENT
# =============================================================================
# Audio buffering
MIN_AUDIO_LENGTH_BYTES = 18000  # ~0.3 second at 16kHz mono int16
MAX_AUDIO_LENGTH_BYTES = 320000  # ~10 seconds (prevent memory issues)

# Buffer timeout
BUFFER_TIMEOUT_SECONDS = 10  # Force processing after this time

# =============================================================================
# IVR (INTERACTIVE VOICE RESPONSE) CONFIGURATION
# =============================================================================
# Audio file paths (inside FreeSWITCH container)
AUDIO_BASE_PATH = "/usr/local/freeswitch/sounds/custom"

# JSON Flow engine configuration (LEGACY — kept for reference)
IVR_FLOW_DIR = "ivr/flows"
IVR_DEFAULT_LANG = "en"
IVR_MAX_RETRIES = 3

# Intent matching thresholds (LEGACY — replaced by LLM)
SEMANTIC_MATCH_THRESHOLD = 0.45  # Minimum similarity score for sentence transformers
FUZZY_MATCH_THRESHOLD = 75       # Minimum similarity score for fuzzywuzzy fallback

# Set to True only if sentence-transformers is installed AND you have enough RAM/CPU.
# When False, only fuzzy matching is used (faster, works well for menu-style IVR).
USE_SEMANTIC_MATCHING = False

# =============================================================================
# PLAYBACK SETTINGS
# =============================================================================
ALLOW_INTERRUPTIONS = os.getenv("ALLOW_INTERRUPTIONS", "true").lower() == "true"
BOT_SPEAKING_TIMEOUT = 30  # Max seconds to hold speaking lock
CALL_END_DISCONNECT_DELAY = 2.5  # Seconds to wait after final audio before hanging up
TRANSFER_DESTINATION = os.getenv("TRANSFER_DESTINATION", "1001")  # Extension to transfer to when action triggered

# =============================================================================
# WEBSOCKET SERVER CONFIGURATION
# =============================================================================
WS_HOST = "0.0.0.0"
WS_PORT = int(os.getenv("WS_PORT", 8000))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/voicebot.log"
ENABLE_TIMING_LOGS = True  # Performance metrics

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================
# Thread pool settings
MAX_WORKERS = 4  # For CPU-intensive tasks (NC, VAD)

# Cache settings
CACHE_MODELS = True  # Keep models in memory between calls

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", '127.0.0.1')
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_TTL = 3600
# Set to False to allow server to start without Redis (useful for single-call testing)
REDIS_REQUIRED = os.getenv("REDIS_REQUIRED", "False").lower() == "true"

# Worker Configuration
WORKER_ID = f"worker-{os.getpid()}"
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", 5))
SESSION_CLEANUP_INTERVAL = 300

# =============================================================================
# PATHS
# =============================================================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs", exist_ok=True)
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Create debug audio directory if debugging enabled
if DF_DEBUG_SAVE_DIR:
    os.makedirs(DF_DEBUG_SAVE_DIR, exist_ok=True)