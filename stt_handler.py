"""
Speech-to-Text Handler — Multi-Provider
Supports:
  - RemoteSTT: HTTP POST to external Faster-Whisper server (your existing server)
  - LocalWhisperSTT: Local Faster-Whisper model (runs in-process)
  
Provider is selected via STT_PROVIDER env var ("remote" or "local").
Auto-fallback: if remote fails N times consecutively, switch to local.
"""

import io
import logging
import requests
import time
import struct
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASE — all STT providers implement this
# =============================================================================

class BaseSTT(ABC):
    """Base class for all STT providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.total_requests = 0
        self.total_errors = 0
        self.total_transcription_time = 0.0
    
    @abstractmethod
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Raw PCM audio bytes (int16, mono, 16kHz)
            
        Returns:
            Transcribed text or None on error
        """
        pass
    
    def transcribe_with_metadata(self, audio_data: bytes) -> dict:
        """Transcribe with additional metadata"""
        start_time = time.time()
        text = self.transcribe(audio_data)
        transcription_time = time.time() - start_time
        
        return {
            'text': text,
            'transcription_time': transcription_time,
            'audio_length': len(audio_data),
            'success': text is not None,
            'provider': self.name,
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
            'provider': self.name,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': f"{error_rate:.1f}%",
            'average_time': f"{avg_time*1000:.0f}ms",
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_requests = 0
        self.total_errors = 0
        self.total_transcription_time = 0.0
        logger.info(f"[{self.name}] STT statistics reset")


# =============================================================================
# REMOTE STT — HTTP POST to external Faster-Whisper server
# =============================================================================

class RemoteSTT(BaseSTT):
    """
    Remote STT via HTTP POST.
    This is the existing implementation — sends audio to your external server.
    """
    
    def __init__(self, stt_url: str, stt_params: dict, timeout: int = 5):
        super().__init__("remote")
        self.stt_url = stt_url
        self.stt_params = stt_params
        self.timeout = timeout
        logger.info(f"✓ RemoteSTT initialized: {stt_url}")
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        if not audio_data or len(audio_data) == 0:
            logger.warning("Empty audio data, skipping transcription")
            return None
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
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
                    audio_duration = len(audio_data) / (16000 * 2)  # 16kHz, int16
                    rtf = transcription_time / audio_duration if audio_duration > 0 else 0
                    logger.info(
                        f"🎯 STT [remote]: '{text}' "
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
            logger.error(f"❌ STT [remote] timeout after {self.timeout}s")
            return None
        except requests.RequestException as e:
            self.total_errors += 1
            logger.error(f"❌ STT [remote] request error: {e}")
            return None
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ STT [remote] unexpected error: {e}")
            return None


# =============================================================================
# LOCAL WHISPER STT — Faster-Whisper running in-process
# =============================================================================

class LocalWhisperSTT(BaseSTT):
    """
    Local STT using Faster-Whisper.
    Model is loaded once on first transcription and kept in memory.
    
    Requires: pip install faster-whisper
    """
    
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        super().__init__("local")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        logger.info(f"✓ LocalWhisperSTT initialized (model: {model_size}, device: {device}, compute: {compute_type})")
        logger.info(f"  Model will be loaded on first transcription request")
    
    def _load_model(self):
        """Load the Faster-Whisper model (one-time, blocks until loaded)"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"⏳ Loading Faster-Whisper model '{self.model_size}' on {self.device}...")
            load_start = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            load_time = time.time() - load_start
            logger.info(f"✓ Faster-Whisper '{self.model_size}' loaded in {load_time:.1f}s")
            
        except ImportError:
            logger.error("❌ faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load Faster-Whisper model: {e}")
            raise
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        if not audio_data or len(audio_data) == 0:
            logger.warning("Empty audio data, skipping transcription")
            return None
        
        # Lazy-load model on first call
        if self.model is None:
            self._load_model()
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Convert raw PCM int16 bytes to float32 numpy array
            # Audio format: 16kHz, mono, int16 (2 bytes per sample)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run transcription
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language="en",
                vad_filter=True,  # Filter out non-speech segments
                vad_parameters=dict(
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=500,
                ),
            )
            
            # Collect all segment texts
            texts = []
            for segment in segments:
                texts.append(segment.text.strip())
            
            text = " ".join(texts).strip()
            
            transcription_time = time.time() - start_time
            self.total_transcription_time += transcription_time
            
            if text:
                audio_duration = len(audio_data) / (16000 * 2)
                rtf = transcription_time / audio_duration if audio_duration > 0 else 0
                logger.info(
                    f"🎯 STT [local/{self.model_size}]: '{text}' "
                    f"({transcription_time*1000:.0f}ms, RTF: {rtf:.2f}x)"
                )
                return text
            else:
                logger.debug(f"STT [local] returned empty text (detected language: {info.language}, prob: {info.language_probability:.2f})")
                return None
                
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ STT [local] error: {e}", exc_info=True)
            return None


# =============================================================================
# STT WITH AUTO-FALLBACK
# =============================================================================

class FallbackSTT(BaseSTT):
    """
    Wraps a primary STT provider with automatic fallback to a secondary.
    If the primary fails N consecutive times, switches to the fallback.
    Periodically retries the primary to check if it's back.
    """
    
    def __init__(self, primary: BaseSTT, fallback: BaseSTT, threshold: int = 3):
        super().__init__(f"fallback({primary.name}→{fallback.name})")
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.consecutive_failures = 0
        self.using_fallback = False
        self._retry_interval = 30  # seconds before retrying primary
        self._last_primary_attempt = 0.0
        
        logger.info(
            f"✓ FallbackSTT: primary={primary.name}, "
            f"fallback={fallback.name}, threshold={threshold}"
        )
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        # If we're on fallback, periodically retry primary
        if self.using_fallback:
            if time.time() - self._last_primary_attempt > self._retry_interval:
                logger.info(f"🔄 Retrying primary STT ({self.primary.name})...")
                self._last_primary_attempt = time.time()
                result = self.primary.transcribe(audio_data)
                if result is not None:
                    logger.info(f"✅ Primary STT ({self.primary.name}) is back! Switching back.")
                    self.using_fallback = False
                    self.consecutive_failures = 0
                    return result
            
            # Use fallback
            return self.fallback.transcribe(audio_data)
        
        # Try primary
        result = self.primary.transcribe(audio_data)
        
        if result is not None:
            self.consecutive_failures = 0
            return result
        
        # Primary failed
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.threshold:
            logger.warning(
                f"⚠️ Primary STT ({self.primary.name}) failed {self.consecutive_failures} times. "
                f"Switching to fallback ({self.fallback.name})"
            )
            self.using_fallback = True
            self._last_primary_attempt = time.time()
            
            # Try fallback for this request too
            return self.fallback.transcribe(audio_data)
        
        return None
    
    def get_stats(self) -> dict:
        return {
            'provider': self.name,
            'using_fallback': self.using_fallback,
            'consecutive_failures': self.consecutive_failures,
            'primary_stats': self.primary.get_stats(),
            'fallback_stats': self.fallback.get_stats(),
        }


# =============================================================================
# FACTORY — creates the right STT handler based on config
# =============================================================================

def create_stt_handler(
    provider: str = "remote",
    # Remote config
    stt_url: str = "",
    stt_params: dict = None,
    stt_timeout: int = 5,
    # Local config
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    # Fallback config
    fallback_enabled: bool = True,
    fallback_threshold: int = 3,
) -> BaseSTT:
    """
    Factory function to create the appropriate STT handler.
    
    Args:
        provider: "remote" or "local"
        stt_url: URL for remote STT server
        stt_params: Query params for remote STT
        stt_timeout: Timeout for remote STT requests
        model_size: Whisper model size for local STT
        device: "cpu" or "cuda" for local STT
        compute_type: "int8" (CPU) or "float16" (GPU) for local STT
        fallback_enabled: Enable auto-fallback from remote to local
        fallback_threshold: Consecutive failures before switching
    
    Returns:
        A BaseSTT instance (RemoteSTT, LocalWhisperSTT, or FallbackSTT)
    """
    stt_params = stt_params or {}
    
    if provider == "local":
        logger.info(f"📋 STT Provider: local (Faster-Whisper {model_size})")
        return LocalWhisperSTT(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )
    
    elif provider == "remote":
        primary = RemoteSTT(
            stt_url=stt_url,
            stt_params=stt_params,
            timeout=stt_timeout,
        )
        
        if fallback_enabled:
            logger.info(f"📋 STT Provider: remote with local fallback")
            fallback = LocalWhisperSTT(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
            )
            return FallbackSTT(
                primary=primary,
                fallback=fallback,
                threshold=fallback_threshold,
            )
        else:
            logger.info(f"📋 STT Provider: remote (no fallback)")
            return primary
    
    else:
        raise ValueError(f"Unknown STT provider: '{provider}'. Use 'remote' or 'local'.")
