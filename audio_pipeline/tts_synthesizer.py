"""
TTS Synthesizer — Edge-TTS with Streaming + Redis Caching

Features:
- synthesize(text) → wav_file_path (standard mode — full file)
- synthesize_streaming(text, callback) → streams sentences to callback
- Redis cache: repeated phrases skip TTS entirely
- Configurable voice via TTS_VOICE env var
"""

import asyncio
import hashlib
import io
import logging
import os
import struct
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TTSSynthesizer:
    """
    Text-to-Speech synthesizer using Edge-TTS.
    
    Edge-TTS is free, requires no API key, and produces high-quality audio.
    It streams audio chunks over HTTP — we can either wait for the full file
    or stream chunks to FreeSWITCH as they arrive.
    """
    
    def __init__(
        self,
        voice: str = "en-US-GuyNeural",
        cache_dir: str = "/app/tts_cache",
        redis_client=None,
        sample_rate: int = 16000,
    ):
        self.voice = voice
        self.cache_dir = cache_dir
        self.redis_client = redis_client
        self.target_sample_rate = sample_rate
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Stats
        self.total_requests = 0
        self.cache_hits = 0
        self.total_synthesis_time = 0.0
        
        logger.info(f"✓ TTSSynthesizer initialized (voice: {voice}, cache: {cache_dir})")
    
    def _cache_key(self, text: str) -> str:
        """Generate a cache key from text + voice"""
        content = f"{self.voice}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cached_path(self, text: str) -> Optional[str]:
        """Check if a cached WAV file exists for this text"""
        key = self._cache_key(text)
        wav_path = os.path.join(self.cache_dir, f"tts_{key}.wav")
        
        # Check file cache
        if os.path.exists(wav_path):
            return wav_path
        
        # Check Redis cache (stores the file path)
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"tts_cache:{key}")
                if cached and os.path.exists(cached.decode() if isinstance(cached, bytes) else cached):
                    return cached.decode() if isinstance(cached, bytes) else cached
            except Exception:
                pass  # Redis failure is not critical
        
        return None
    
    def _set_cache(self, text: str, wav_path: str):
        """Store the wav path in Redis cache"""
        if self.redis_client:
            try:
                key = self._cache_key(text)
                self.redis_client.setex(f"tts_cache:{key}", 3600, wav_path)  # 1 hour TTL
            except Exception:
                pass  # Redis failure is not critical
    
    def _convert_mp3_to_wav(self, mp3_data: bytes) -> bytes:
        """
        Convert MP3 audio bytes to WAV format (16kHz, mono, int16).
        Edge-TTS outputs MP3 by default — FreeSWITCH needs WAV/PCM.
        
        Uses ffmpeg subprocess (fast, ~50ms) instead of torchaudio (slow, ~1000ms+).
        """
        # Try ffmpeg first — much faster than torchaudio for this simple conversion
        try:
            return self._convert_with_ffmpeg(mp3_data)
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}. Falling back to torchaudio.")
        
        # Fallback to torchaudio
        try:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
            import torch
            import torchaudio
            
            buffer = io.BytesIO(mp3_data)
            waveform, sample_rate = torchaudio.load(buffer, format='mp3')
            
            if sample_rate != self.target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            wav_buffer = io.BytesIO()
            torchaudio.save(wav_buffer, waveform, self.target_sample_rate, format='wav', bits_per_sample=16)
            return wav_buffer.getvalue()
            
        except Exception as e2:
            logger.error(f"All MP3→WAV conversions failed: {e2}")
            raise
    
    def _convert_with_ffmpeg(self, mp3_data: bytes) -> bytes:
        """Fallback: use ffmpeg subprocess for mp3→wav conversion"""
        import subprocess
        
        try:
            process = subprocess.run(
                [
                    "ffmpeg", "-i", "pipe:0",
                    "-ar", str(self.target_sample_rate),
                    "-ac", "1",
                    "-f", "wav",
                    "pipe:1",
                ],
                input=mp3_data,
                capture_output=True,
                timeout=10,
            )
            
            if process.returncode == 0:
                return process.stdout
            else:
                logger.error(f"ffmpeg error: {process.stderr.decode()[:200]}")
                raise RuntimeError("ffmpeg conversion failed")
                
        except FileNotFoundError:
            raise RuntimeError("Neither pydub nor ffmpeg available for MP3→WAV conversion")
    
    def synthesize(self, text: str) -> Optional[str]:
        """
        Synthesize text to a WAV file (blocking/sync mode).
        
        Safe to call from BOTH:
        - Async context (WebSocket handler) — runs edge-tts in a separate thread
        - Sync context (thread pool) — runs edge-tts via asyncio.run()
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to the generated WAV file, or None on error
        """
        if not text or not text.strip():
            logger.warning("Empty text, skipping TTS")
            return None
        
        self.total_requests += 1
        
        # Check cache first
        cached = self._get_cached_path(text)
        if cached:
            self.cache_hits += 1
            logger.debug(f"🔊 TTS cache hit: '{text[:50]}...' → {cached}")
            return cached
        
        start_time = time.time()
        
        # Run the async Edge-TTS directly if no event loop is running on the thread.
        # Otherwise, run it in a single worker thread pool.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self._synthesize_async(text)))
                wav_path = future.result(timeout=30)
        else:
            wav_path = asyncio.run(self._synthesize_async(text))
        
        if wav_path:
            synthesis_time = time.time() - start_time
            self.total_synthesis_time += synthesis_time
            logger.info(
                f"🔊 TTS: '{text[:60]}{'...' if len(text) > 60 else ''}' "
                f"→ {wav_path} ({synthesis_time*1000:.0f}ms)"
            )
        
        return wav_path
    
    async def synthesize_awaitable(self, text: str) -> Optional[str]:
        """
        Async version of synthesize — use this from async handlers.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to the generated WAV file, or None on error
        """
        if not text or not text.strip():
            logger.warning("Empty text, skipping TTS")
            return None
        
        self.total_requests += 1
        
        # Check cache first
        cached = self._get_cached_path(text)
        if cached:
            self.cache_hits += 1
            logger.debug(f"🔊 TTS cache hit: '{text[:50]}...' → {cached}")
            return cached
        
        start_time = time.time()
        
        wav_path = await self._synthesize_async(text)
        
        if wav_path:
            synthesis_time = time.time() - start_time
            self.total_synthesis_time += synthesis_time
            logger.info(
                f"🔊 TTS: '{text[:60]}{'...' if len(text) > 60 else ''}' "
                f"→ {wav_path} ({synthesis_time*1000:.0f}ms)"
            )
        
        return wav_path
    
    async def _synthesize_async(self, text: str) -> Optional[str]:
        """Internal async implementation of TTS synthesis"""
        try:
            import edge_tts
            
            communicate = edge_tts.Communicate(text, self.voice)
            t_start = time.time()
            
            # Collect all audio chunks
            mp3_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_chunks.append(chunk["data"])
            
            if not mp3_chunks:
                logger.error("Edge-TTS returned no audio chunks")
                return None
            
            # Combine MP3 chunks and convert to WAV
            mp3_data = b"".join(mp3_chunks)
            edge_time = time.time()
            logger.debug(f"🔊 TTS edge-tts streaming: {(edge_time - t_start)*1000:.0f}ms, {len(mp3_data)} bytes MP3")
            
            wav_data = self._convert_mp3_to_wav(mp3_data)
            convert_time = time.time()
            logger.debug(f"🔊 TTS MP3→WAV conversion: {(convert_time - edge_time)*1000:.0f}ms")
            
            # Save to cache
            key = self._cache_key(text)
            wav_path = os.path.join(self.cache_dir, f"tts_{key}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_data)
            
            # Update Redis cache
            self._set_cache(text, wav_path)
            
            return wav_path
            
        except ImportError:
            logger.error("❌ edge-tts not installed. Run: pip install edge-tts")
            return None
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}", exc_info=True)
            return None
    
    async def synthesize_streaming(
        self,
        text: str,
        on_sentence_ready: Callable[[str], None],
    ):
        """
        Streaming TTS: synthesize sentence-by-sentence and call back as each is ready.
        
        This is used when LLM streams sentences. Each sentence arrives,
        gets synthesized, and the callback plays it immediately.
        
        Args:
            text: A single sentence to synthesize
            on_sentence_ready: Callback called with wav_path when audio is ready
        """
        # Check cache first
        cached = self._get_cached_path(text)
        if cached:
            self.cache_hits += 1
            on_sentence_ready(cached)
            return
        
        wav_path = await self._synthesize_async(text)
        if wav_path:
            on_sentence_ready(wav_path)
    
    def get_stats(self) -> dict:
        """Get TTS statistics"""
        avg_time = (
            self.total_synthesis_time / (self.total_requests - self.cache_hits)
            if (self.total_requests - self.cache_hits) > 0 else 0
        )
        cache_rate = (
            (self.cache_hits / self.total_requests * 100)
            if self.total_requests > 0 else 0
        )
        return {
            'voice': self.voice,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_rate': f"{cache_rate:.1f}%",
            'average_synthesis_time': f"{avg_time*1000:.0f}ms",
        }
