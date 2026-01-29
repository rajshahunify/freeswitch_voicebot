"""
Audio Buffer Manager
Manages audio buffering with VAD integration
"""

import logging
import time
from typing import Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Manages audio buffering with VAD-based segmentation
    
    Features:
    - Accumulates audio during speech
    - Triggers processing on speech end
    - Prevents buffer overflow
    - Handles timeout scenarios
    """
    
    def __init__(self,
                #  min_length: int = 32000,
                 min_length: int = 4800,
                 max_length: int = 320000,
                 timeout_seconds: float = 10.0,
                 on_speech_end: Optional[Callable] = None):
        """
        Initialize audio buffer
        
        Args:
            min_length: Minimum buffer size before processing (bytes)
            max_length: Maximum buffer size (prevent overflow)
            timeout_seconds: Force processing after this time
            on_speech_end: Callback when speech ends
        """
        self.min_length = min_length
        self.max_length = max_length
        self.timeout_seconds = timeout_seconds
        self.on_speech_end = on_speech_end
        
        self.reset()
        
    def reset(self):
        """Reset buffer state"""
        self.buffer = bytearray()
        self.speech_started = False
        self.speech_start_time = None
        self.last_chunk_time = time.time()
        self.total_bytes = 0
        
    def add_chunk(self, audio_chunk: bytes, vad_result: dict) -> Optional[bytes]:
        """
        Add audio chunk to buffer based on VAD result
        
        Args:
            audio_chunk: Raw PCM audio bytes
            vad_result: VAD detection result dict
            
        Returns:
            Audio bytes if ready to process, None otherwise
        """
        self.last_chunk_time = time.time()
        self.total_bytes += len(audio_chunk)
        
        # Handle speech start
        if vad_result.get('speech_start', False):
            self.speech_started = True
            self.speech_start_time = time.time()
            self.buffer = bytearray()  # Clear any previous data
            logger.debug("🎙️  Buffer: Speech started")
        
        # Accumulate audio during speech
        if vad_result.get('is_speech', False) or self.speech_started:
            self.buffer.extend(audio_chunk)
            
            # Check for buffer overflow
            if len(self.buffer) > self.max_length:
                logger.warning(f"⚠️  Buffer overflow ({len(self.buffer)} bytes), forcing processing")
                return self._extract_buffer()
        
        # Handle speech end
        if vad_result.get('speech_end', False):
            if len(self.buffer) >= self.min_length:
                logger.debug(f"✓ Speech ended, buffer ready ({len(self.buffer)} bytes)")
                return self._extract_buffer()
            else:
                logger.debug(f"⚠️  Speech too short ({len(self.buffer)} bytes), ignoring")
                self.reset()
                return None
        
        # Check timeout
        if self.speech_started and self.speech_start_time:
            elapsed = time.time() - self.speech_start_time
            if elapsed > self.timeout_seconds:
                if len(self.buffer) >= self.min_length:
                    logger.warning(f"⏱️  Buffer timeout ({elapsed:.1f}s), forcing processing")
                    return self._extract_buffer()
                else:
                    logger.warning(f"⏱️  Buffer timeout with insufficient data, resetting")
                    self.reset()
                    return None
        
        return None
    
    def _extract_buffer(self) -> bytes:
        """Extract buffer contents and reset"""
        audio_data = bytes(self.buffer)
        self.reset()
        return audio_data
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'buffer_size': len(self.buffer),
            'speech_started': self.speech_started,
            'total_bytes_received': self.total_bytes,
            'speech_duration': (
                time.time() - self.speech_start_time 
                if self.speech_start_time else 0
            )
        }


class CallAudioManager:
    """
    Manages audio buffers for multiple concurrent calls
    Each call gets its own buffer instance
    """
    
    def __init__(self, **buffer_kwargs):
        """
        Initialize call audio manager
        
        Args:
            **buffer_kwargs: Arguments passed to AudioBuffer constructor
        """
        self.buffer_kwargs = buffer_kwargs
        self.call_buffers = {}
        
    def get_buffer(self, call_uuid: str) -> AudioBuffer:
        """
        Get or create audio buffer for a call
        
        Args:
            call_uuid: Unique call identifier
            
        Returns:
            AudioBuffer instance for this call
        """
        if call_uuid not in self.call_buffers:
            self.call_buffers[call_uuid] = AudioBuffer(**self.buffer_kwargs)
            logger.debug(f"Created buffer for call {call_uuid}")
        
        return self.call_buffers[call_uuid]
    
    def remove_buffer(self, call_uuid: str):
        """
        Remove buffer for ended call
        
        Args:
            call_uuid: Call identifier to remove
        """
        if call_uuid in self.call_buffers:
            # Get final stats
            stats = self.call_buffers[call_uuid].get_stats()
            logger.info(f"Call {call_uuid} stats: {stats}")
            
            del self.call_buffers[call_uuid]
            logger.debug(f"Removed buffer for call {call_uuid}")
    
    def get_all_stats(self) -> dict:
        """Get statistics for all active calls"""
        return {
            uuid: buffer.get_stats() 
            for uuid, buffer in self.call_buffers.items()
        }
    
    def active_calls(self) -> int:
        """Get number of active calls"""
        return len(self.call_buffers)


# =============================================================================
# SLIDING WINDOW BUFFER (Alternative approach)
# =============================================================================

class SlidingWindowBuffer:
    """
    Alternative buffer implementation using sliding window
    Useful for continuous processing scenarios
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 chunk_size: int = 512):
        """
        Initialize sliding window buffer
        
        Args:
            window_size: Number of chunks to keep
            chunk_size: Size of each chunk in samples
        """
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.window = deque(maxlen=window_size)
        
    def add_chunk(self, chunk: bytes):
        """Add chunk to sliding window"""
        self.window.append(chunk)
    
    def get_buffer(self) -> bytes:
        """Get concatenated buffer contents"""
        return b''.join(self.window)
    
    def is_full(self) -> bool:
        """Check if window is full"""
        return len(self.window) == self.window_size
    
    def clear(self):
        """Clear the window"""
        self.window.clear()