"""
Voice Activity Detection using Silero VAD
Detects speech segments in real-time audio
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple
from collections import deque
import time

logger = logging.getLogger(__name__)


class VADDetector:
    """
    Real-time Voice Activity Detection using Silero VAD
    
    Features:
    - Detects speech start/end in streaming audio
    - Maintains state across chunks
    - Configurable thresholds and padding
    - Very fast inference (~1ms per chunk)
    """
    
    def __init__(self,
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 300,
                 speech_pad_ms: int = 30,
                 sample_rate: int = 16000,
                 window_size: int = 512):
        """
        Initialize VAD detector
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration
            min_silence_duration_ms: Silence before speech end
            speech_pad_ms: Padding before/after speech
            sample_rate: Audio sample rate (8000 or 16000)
            window_size: Samples per window (512 or 1024)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Validate sample rate
        if sample_rate not in [8000, 16000]:
            raise ValueError("Sample rate must be 8000 or 16000")
        
        # Calculate frame counts
        self.min_speech_frames = int(min_speech_duration_ms * sample_rate / 1000 / window_size)
        self.min_silence_frames = int(min_silence_duration_ms * sample_rate / 1000 / window_size)
        self.speech_pad_frames = int(speech_pad_ms * sample_rate / 1000 / window_size)
        
        self.model = None
        self.reset_state()
        
        logger.info(f"Initializing VAD: threshold={threshold}, sr={sample_rate}Hz")
        
    def load_model(self):
        """Load Silero VAD model (called once at startup)"""
        if self.model is not None:
            logger.debug("VAD model already loaded")
            return
        
        start_time = time.time()
        
        try:
            # Load Silero VAD from torch hub
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Get utilities
            (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks) = utils
            
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"✓ Silero VAD loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.error("Ensure PyTorch is installed: pip install torch --break-system-packages")
            raise
    
    def reset_state(self):
        """Reset internal state (call between different calls)"""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_buffer = []
        self.triggered = False
        
        # For Silero internal state
        if hasattr(self, 'model') and self.model is not None:
            self.model.reset_states()
    
    def process_chunk(self, audio_chunk: bytes) -> Tuple[bool, float]:
        """
        Process audio chunk and detect speech
        
        Args:
            audio_chunk: Raw PCM audio (int16, mono)
            
        Returns:
            Tuple of (is_speech: bool, probability: float)
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert bytes to numpy
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Ensure correct chunk size
            if len(audio_np) != self.window_size:
                # Pad or truncate to window size
                if len(audio_np) < self.window_size:
                    audio_np = np.pad(audio_np, (0, self.window_size - len(audio_np)))
                else:
                    audio_np = audio_np[:self.window_size]
            
            # Normalize to float32 [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float)
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # Determine if speech
            is_speech = speech_prob >= self.threshold
            
            return is_speech, speech_prob
            
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return False, 0.0
    
    def process_stream(self, audio_chunk: bytes) -> dict:
        """
        Process audio chunk with state management for speech detection
        
        Args:
            audio_chunk: Raw PCM audio bytes
            
        Returns:
            Dictionary with:
                - is_speech: bool - Current frame contains speech
                - speech_start: bool - Speech just started
                - speech_end: bool - Speech just ended
                - probability: float - Speech probability
                - should_process: bool - Accumulated enough speech to process
        """
        is_speech, probability = self.process_chunk(audio_chunk)
        
        speech_start = False
        speech_end = False
        should_process = False
        
        if is_speech:
            # Speech detected
            self.silence_frames = 0
            self.speech_frames += 1
            
            if not self.is_speaking:
                # Speech just started
                if self.speech_frames >= self.min_speech_frames:
                    self.is_speaking = True
                    speech_start = True
                    self.triggered = True
                    logger.debug(f"🎤 Speech START (prob: {probability:.2f})")
        else:
            # Silence detected
            self.speech_frames = 0
            
            if self.is_speaking:
                self.silence_frames += 1
                
                # Check if speech ended
                if self.silence_frames >= self.min_silence_frames:
                    self.is_speaking = False
                    speech_end = True
                    should_process = self.triggered
                    self.triggered = False
                    logger.debug(f"🎤 Speech END (silence: {self.silence_frames} frames)")
        
        return {
            'is_speech': is_speech,
            'speech_start': speech_start,
            'speech_end': speech_end,
            'probability': probability,
            'should_process': should_process
        }
    
    def __del__(self):
        """Cleanup"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_vad_instance: Optional[VADDetector] = None


def get_vad_detector(
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    sample_rate: int = 16000,
    **kwargs
) -> VADDetector:
    """
    Get or create shared VAD detector instance
    
    Returns:
        Shared VADDetector instance
    """
    global _vad_instance
    
    if _vad_instance is None:
        _vad_instance = VADDetector(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sample_rate=sample_rate,
            **kwargs
        )
        _vad_instance.load_model()
    
    return _vad_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_speech(audio_chunk: bytes) -> Tuple[bool, float]:
    """
    Convenience function to detect speech in audio chunk
    
    Args:
        audio_chunk: Raw PCM audio bytes
        
    Returns:
        Tuple of (is_speech, probability)
    """
    vad = get_vad_detector()
    return vad.process_chunk(audio_chunk)