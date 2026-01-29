# """
# FreeSWITCH VoiceBot Configuration
# Centralized configuration for all components
# """

# import os

# # =============================================================================
# # FREESWITCH CONFIGURATION
# # =============================================================================
# FREESWITCH_HOST = '127.0.0.1'
# FREESWITCH_PORT = 8021
# FREESWITCH_PASSWORD = 'ClueCon'

# # WebSocket URL where FreeSWITCH sends audio
# WEBSOCKET_URL = "ws://127.0.0.1:8000/media"

# # =============================================================================
# # AUDIO CONFIGURATION
# # =============================================================================
# SAMPLE_RATE = 16000  # Hz - DO NOT CHANGE (required for DF2, Silero, Whisper)
# CHANNELS = 1  # Mono
# BIT_DEPTH = 16  # bits
# CHUNK_DURATION_MS = 32  # milliseconds (512 samples at 16kHz)
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 512 samples

# # =============================================================================
# # STT (SPEECH-TO-TEXT) CONFIGURATION
# # =============================================================================
# STT_URL = "http://164.52.203.140:8890/transcribe"
# STT_PARAMS = {
#     "sid": "live",
#     "sample_rate": SAMPLE_RATE,
#     "bit_depth": f"int{BIT_DEPTH}",
#     "language": "en"
# }
# STT_TIMEOUT = 5  # seconds

# # =============================================================================
# # DEEPFILTERNET2 (NOISE CANCELLATION) CONFIGURATION
# # =============================================================================
# # Model options: 'DeepFilterNet2' or 'DeepFilterNet3'
# DF_MODEL = 'DeepFilterNet2'

# # Processing settings
# DF_FRAME_SIZE = CHUNK_SIZE  # Process in 32ms chunks
# DF_COMPENSATE_DELAY = True  # Compensate for processing delay
# DF_ATTENUATION_LIMIT = 100  # dB - maximum noise reduction

# # Performance
# DF_USE_GPU = False  # Set to True if CUDA available in WSL
# DF_POST_FILTER = True  # Additional perceptual enhancement

# # =============================================================================
# # SILERO VAD (VOICE ACTIVITY DETECTION) CONFIGURATION
# # =============================================================================
# # Thresholds
# VAD_THRESHOLD = 0.2  # Speech probability threshold (0.0 - 1.0) - LOWERED for better detection
# VAD_MIN_SPEECH_DURATION_MS = 100  # Minimum speech duration to consider - REDUCED
# VAD_MIN_SILENCE_DURATION_MS = 500  # Silence duration before speech end - INCREASED for full utterances
# VAD_SPEECH_PAD_MS = 30  # Padding before/after speech

# # Window size for VAD (512 or 1024 samples recommended)
# VAD_WINDOW_SIZE = 512  # 32ms at 16kHz
# VAD_SAMPLE_RATE = SAMPLE_RATE  # Must be 8000 or 16000

# # =============================================================================
# # BUFFER MANAGEMENT
# # =============================================================================
# # Audio buffering
# MIN_AUDIO_LENGTH_BYTES = 32000  # ~1 second at 16kHz mono int16
# MAX_AUDIO_LENGTH_BYTES = 320000  # ~10 seconds (prevent memory issues)

# # Buffer timeout
# BUFFER_TIMEOUT_SECONDS = 10  # Force processing after this time

# # =============================================================================
# # IVR (INTERACTIVE VOICE RESPONSE) CONFIGURATION
# # =============================================================================
# # Audio file paths
# AUDIO_BASE_PATH = "/usr/local/freeswitch/sounds/custom"

# # Intent to audio file mapping
# INTENT_KEYWORDS = {
#     "hello": "english_menu.wav",
#     "hi": "english_menu.wav", 
#     "menu": "english_menu.wav",
#     "bye": "thank_you.wav",
#     "thanks": "thank_you.wav",
#     "thank": "thank_you.wav",
#     "internet": "internet_inquiries.wav",
#     "data": "internet_inquiries.wav",
#     "mpesa": "mpesa.wav",
#     "payment": "payment_options.wav",
#     "location": "shop_location.wav",
#     "default": "sorry.wav"
# }

# # Intent matching settings
# FUZZY_MATCH_THRESHOLD = 70  # Minimum similarity score (0-100)

# # =============================================================================
# # PLAYBACK SETTINGS
# # =============================================================================
# ALLOW_INTERRUPTIONS = False  # Allow user to interrupt bot speech
# BOT_SPEAKING_TIMEOUT = 30  # Max seconds to hold speaking lock

# # =============================================================================
# # WEBSOCKET SERVER CONFIGURATION
# # =============================================================================
# WS_HOST = "0.0.0.0"
# WS_PORT = 8000

# # =============================================================================
# # LOGGING CONFIGURATION
# # =============================================================================
# LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR - Set to DEBUG for troubleshooting
# LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# LOG_FILE = "logs/voicebot.log"
# ENABLE_TIMING_LOGS = True  # Performance metrics

# # =============================================================================
# # PERFORMANCE OPTIMIZATION
# # =============================================================================
# # Thread pool settings
# MAX_WORKERS = 4  # For CPU-intensive tasks (NC, VAD)

# # Cache settings
# CACHE_MODELS = True  # Keep models in memory between calls

# # =============================================================================
# # PATHS
# # =============================================================================
# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs", exist_ok=True)



"""
FreeSWITCH VoiceBot Configuration
Centralized configuration for all components
"""

import os

# =============================================================================
# FREESWITCH CONFIGURATION
# =============================================================================
FREESWITCH_HOST = '127.0.0.1'
FREESWITCH_PORT = 8021
FREESWITCH_PASSWORD = 'ClueCon'

# WebSocket URL where FreeSWITCH sends audio
WEBSOCKET_URL = "ws://127.0.0.1:8000/media"

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
STT_URL = "http://164.52.203.140:8890/transcribe"
STT_PARAMS = {
    "sid": "live",
    "sample_rate": SAMPLE_RATE,
    "bit_depth": f"int{BIT_DEPTH}",
    "language": "en"
}
STT_TIMEOUT = 5  # seconds

# =============================================================================
# DEEPFILTERNET2 (NOISE CANCELLATION) CONFIGURATION
# =============================================================================
# Model options: 'DeepFilterNet2' or 'DeepFilterNet3'
DF_MODEL = 'DeepFilterNet2'

# Processing settings
DF_FRAME_SIZE = CHUNK_SIZE  # Process in 32ms chunks
DF_COMPENSATE_DELAY = True  # Compensate for processing delay
DF_ATTENUATION_LIMIT = 100  # dB - maximum noise reduction

# Performance
DF_USE_GPU = False  # Set to True if CUDA available in WSL
DF_POST_FILTER = True  # Additional perceptual enhancement

# =============================================================================
# SILERO VAD (VOICE ACTIVITY DETECTION) CONFIGURATION
# =============================================================================
# Thresholds
# VAD_THRESHOLD = 0.15  # Speech probability threshold (VERY LOW - no NC preprocessing)
VAD_THRESHOLD = 0.3 # Speech probability threshold (VERY LOW - no NC preprocessing)
VAD_MIN_SPEECH_DURATION_MS = 400  # Minimum speech duration to consider
VAD_MIN_SILENCE_DURATION_MS = 800  # Silence duration before speech end (longer for complete utterances)
VAD_SPEECH_PAD_MS = 30  # Padding before/after speech

# Window size for VAD (512 or 1024 samples recommended)
VAD_WINDOW_SIZE = 512  # 32ms at 16kHz
VAD_SAMPLE_RATE = SAMPLE_RATE  # Must be 8000 or 16000

# =============================================================================
# BUFFER MANAGEMENT
# =============================================================================
# Audio buffering
# MIN_AUDIO_LENGTH_BYTES = 32000  # ~1 second at 16kHz mono int16
MIN_AUDIO_LENGTH_BYTES = 18000  # ~0.3 second at 16kHz mono int16
MAX_AUDIO_LENGTH_BYTES = 320000  # ~10 seconds (prevent memory issues)

# Buffer timeout
BUFFER_TIMEOUT_SECONDS = 10  # Force processing after this time

# =============================================================================
# IVR (INTERACTIVE VOICE RESPONSE) CONFIGURATION
# =============================================================================
# Audio file paths
AUDIO_BASE_PATH = "/usr/local/freeswitch/sounds/custom"

# Intent to audio file mapping
INTENT_KEYWORDS = {
    "hello": "english_menu.wav",
    "hi": "english_menu.wav", 
    "menu": "english_menu.wav",
    "bye": "thank_you.wav",
    "thanks": "thank_you.wav",
    "thank": "thank_you.wav",
    "internet": "internet_inquiries.wav",
    "data": "internet_inquiries.wav",
    "mpesa": "mpesa.wav",
    "payment": "payment_options.wav",
    "location": "shop_location.wav",
    "default": "sorry.wav"
}

# Intent matching settings
FUZZY_MATCH_THRESHOLD = 70  # Minimum similarity score (0-100)

# =============================================================================
# PLAYBACK SETTINGS
# =============================================================================
ALLOW_INTERRUPTIONS = False  # Allow user to interrupt bot speech
BOT_SPEAKING_TIMEOUT = 30  # Max seconds to hold speaking lock

# =============================================================================
# WEBSOCKET SERVER CONFIGURATION
# =============================================================================
WS_HOST = "0.0.0.0"
WS_PORT = 8000

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR - Set to DEBUG for troubleshooting
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

# =============================================================================
# PATHS
# =============================================================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs", exist_ok=True)
