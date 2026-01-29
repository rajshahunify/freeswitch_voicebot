"""
Speech-to-Text Handler
Manages STT API requests and response processing
"""

import logging
import requests
import time
from typing import Optional

logger = logging.getLogger(__name__)


class STTHandler:
    """
    Handles speech-to-text transcription
    
    Features:
    - Async-ready STT API calls
    - Error handling and retries
    - Performance logging
    """
    
    def __init__(self,
                 stt_url: str,
                 stt_params: dict,
                 timeout: int = 5):
        """
        Initialize STT handler
        
        Args:
            stt_url: STT API endpoint URL
            stt_params: Default parameters for STT API
            timeout: Request timeout in seconds
        """
        self.stt_url = stt_url
        self.stt_params = stt_params
        self.timeout = timeout
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.total_transcription_time = 0
        
        logger.info(f"STTHandler initialized: {stt_url}")
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Raw PCM audio bytes (int16, mono, 16kHz)
            
        Returns:
            Transcribed text or None on error
        """
        if not audio_data or len(audio_data) == 0:
            logger.warning("Empty audio data, skipping transcription")
            return None
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Make API request
            response = requests.post(
                self.stt_url,
                params=self.stt_params,
                data=audio_data,
                timeout=self.timeout
            )
            
            transcription_time = time.time() - start_time
            self.total_transcription_time += transcription_time
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                
                if text:
                    # Calculate real-time factor
                    audio_duration = len(audio_data) / (16000 * 2)  # 16kHz, int16
                    rtf = transcription_time / audio_duration if audio_duration > 0 else 0
                    
                    logger.info(
                        f"🎯 STT: '{text}' "
                        f"({transcription_time*1000:.0f}ms, RTF: {rtf:.2f}x)"
                    )
                    return text
                else:
                    logger.debug("STT returned empty text")
                    return None
            else:
                self.total_errors += 1
                logger.error(f"❌ STT API error: Status {response.status_code}")
                return None
                
        except requests.Timeout:
            self.total_errors += 1
            logger.error(f"❌ STT timeout after {self.timeout}s")
            return None
        except requests.RequestException as e:
            self.total_errors += 1
            logger.error(f"❌ STT request error: {e}")
            return None
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ STT unexpected error: {e}")
            return None
    
    def transcribe_with_metadata(self, audio_data: bytes) -> dict:
        """
        Transcribe with additional metadata
        
        Args:
            audio_data: Raw PCM audio bytes
            
        Returns:
            Dictionary with text, confidence, and timing info
        """
        start_time = time.time()
        text = self.transcribe(audio_data)
        transcription_time = time.time() - start_time
        
        return {
            'text': text,
            'transcription_time': transcription_time,
            'audio_length': len(audio_data),
            'success': text is not None
        }
    
    def get_stats(self) -> dict:
        """Get STT statistics"""
        avg_time = (
            self.total_transcription_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        error_rate = (
            (self.total_errors / self.total_requests * 100)
            if self.total_requests > 0 else 0
        )
        
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': f"{error_rate:.1f}%",
            'average_time': f"{avg_time*1000:.0f}ms"
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_requests = 0
        self.total_errors = 0
        self.total_transcription_time = 0
        logger.info("STT statistics reset")


# =============================================================================
# ALTERNATIVE STT IMPLEMENTATIONS
# =============================================================================

class WhisperLocalSTT:
    """
    Local Whisper STT (for self-hosted scenarios)
    
    This is a placeholder for local Whisper integration
    Requires: pip install openai-whisper
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize local Whisper
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        logger.info(f"WhisperLocalSTT (model: {model_size})")
    
    def load_model(self):
        """Load Whisper model"""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            logger.info(f"✓ Whisper {self.model_size} loaded")
        except ImportError:
            logger.error("Whisper not installed: pip install openai-whisper")
            raise
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio using local Whisper
        
        Args:
            audio_data: Raw PCM audio bytes
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            self.load_model()
        
        # TODO: Implement local Whisper transcription
        # Need to convert bytes to format Whisper expects
        raise NotImplementedError("Local Whisper not yet implemented")
