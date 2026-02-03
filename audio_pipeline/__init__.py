"""
Audio Pipeline Module
Complete audio processing pipeline: Noise Cancellation → VAD → Buffering
"""

from .improved_noise_canceller import (
    ImprovedNoiseCanceller,
    get_improved_noise_canceller,
    denoise_utterance,
    _improved_nc_instance
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
    'ImprovedNoiseCanceller',
    'get_improved_noise_canceller',
    'denoise_utterance',
    '_improved_nc_instance',
    
    # VAD
    'VADDetector',
    'get_vad_detector',
    'detect_speech',
    
    # Buffering
    'AudioBuffer',
    'CallAudioManager',
    'SlidingWindowBuffer',
]