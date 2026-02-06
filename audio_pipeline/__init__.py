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
    PerCallVADManager,
    get_vad_manager,
    VADDetector
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
    'PerCallVADManager',
    'get_vad_manager',
    'detect_speech',
    'VADDetector',
    
    # Buffering
    'AudioBuffer',
    'CallAudioManager',
    'SlidingWindowBuffer',
]