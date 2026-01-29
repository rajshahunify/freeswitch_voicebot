"""
Audio Pipeline Module
Complete audio processing pipeline: Noise Cancellation → VAD → Buffering
"""

from .noise_canceller import (
    NoiseCanceller,
    get_noise_canceller,
    denoise_audio
)

from .vad_detector import (
    VADDetector,
    get_vad_detector,
    detect_speech
)

from .audio_buffer import (
    AudioBuffer,
    CallAudioManager,
    SlidingWindowBuffer
)

__all__ = [
    # Noise Cancellation
    'NoiseCanceller',
    'get_noise_canceller',
    'denoise_audio',
    
    # VAD
    'VADDetector',
    'get_vad_detector',
    'detect_speech',
    
    # Buffering
    'AudioBuffer',
    'CallAudioManager',
    'SlidingWindowBuffer',
]