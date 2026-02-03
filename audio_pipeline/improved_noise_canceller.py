"""
Improved Noise Cancellation using DeepFilterNet2
Based on successful reference implementation
Processes complete utterances instead of tiny chunks
"""

import os
import wave
from datetime import datetime
import numpy as np
import torch
import logging
from typing import Optional, Tuple
import time
import resampy

logger = logging.getLogger(__name__)


class ImprovedNoiseCanceller:
    """
    Improved real-time noise cancellation using DeepFilterNet2
    
    Key improvements over original:
    1. Processes complete utterances (not 32ms chunks)
    2. Proper resampling to/from DeepFilter SR
    3. Better normalization strategy
    4. No crude gain multiplication
    """
    
    def __init__(self,
                 model_name: str = 'DeepFilterNet2',
                 use_gpu: bool = False,
                 post_filter: bool = False,
                 attenuation_limit: float = 100.0,  # Use default, not limited
                 normalization_gain: float = 1.5,  # Smarter than raw gain
                 sample_rate: int = 16000,
                 debug_rms: bool = False,
                 debug_save_dir: Optional[str] = None):
        """
        Initialize improved noise canceller
        
        Args:
            model_name: 'DeepFilterNet2' or 'DeepFilterNet3'
            use_gpu: Use CUDA if available
            post_filter: Apply perceptual post-filtering (keep False for speed)
            attenuation_limit: Max noise reduction in dB (100 = use model default)
            normalization_gain: Gain for normalization (1.5 = reference implementation)
            sample_rate: Target sample rate for output (16kHz for Whisper)
            debug_rms: Log RMS values
            debug_save_dir: Directory to save debug audio files
        """
        self.model = None
        self.df_state = None
        self.device = None
        
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.post_filter = post_filter
        self.attenuation_limit = attenuation_limit
        self.normalization_gain = normalization_gain
        self.target_sr = sample_rate
        self.debug_rms = debug_rms
        self.debug_save_dir = debug_save_dir
        
        if self.debug_save_dir:
            try:
                os.makedirs(self.debug_save_dir, exist_ok=True)
            except Exception:
                logger.exception("Could not create debug_save_dir: %s", self.debug_save_dir)
            self._nc_segment_counter = 0
        
        logger.info(f"Initializing ImprovedNoiseCanceller: {model_name}")
        logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}")
        logger.info(f"Normalization: {normalization_gain}x, Target SR: {sample_rate}Hz")
    
    def load_model(self):
        """Load DeepFilterNet model (called once at startup)"""
        if getattr(self, "model", None) is not None:
            logger.debug("Model already loaded")
            return
        
        start_time = time.time()
        
        try:
            from df.enhance import enhance, init_df
            
            # Load model and state
            self.model, self.df_state, _ = init_df(
                model_base_dir=None,
                post_filter=self.post_filter,
                log_level="WARNING"
            )
            
            # Set device
            if self.use_gpu:
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
                logger.info("✓ Model loaded on GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("✓ Model loaded on CPU")
            
            load_time = time.time() - start_time
            logger.info(f"✓ DeepFilterNet loaded in {load_time:.2f}s")
            logger.info(f"  Model SR: {self.df_state.sr()}Hz")
            
        except Exception as e:
            logger.error(f"Failed to load DeepFilterNet model: {e}")
            logger.error("Install with: pip install deepfilternet --break-system-packages")
            raise
    
    def _write_wav_bytes(self, path: str, pcm_bytes: bytes, sample_rate: int):
        """Write int16 PCM bytes to WAV file"""
        if not pcm_bytes or len(pcm_bytes) == 0:
            return
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)
            logger.debug(f"Saved: {path} ({len(pcm_bytes)} bytes @ {sample_rate}Hz)")
        except Exception:
            logger.exception(f"Failed to write WAV: {path}")
    
    def process_utterance(self, audio_data: bytes, input_sr: int = 16000) -> bytes:
        """
        Process complete utterance (not tiny chunks!)
        
        This is the KEY difference from original implementation:
        - Processes 1-10 second utterances
        - Resamples to DeepFilter SR
        - Processes full segment
        - Normalizes intelligently
        - Resamples back to target SR
        
        Args:
            audio_data: Complete utterance (int16 PCM bytes, mono)
            input_sr: Input sample rate
            
        Returns:
            Denoised audio (int16 PCM bytes at target_sr)
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert bytes → float32 array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            if audio_int16.size == 0:
                return b''
            
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Save original for debugging
            should_save_debug = (
                self.debug_save_dir and 
                len(audio_data) > (input_sr * 2 * 0.3)  # > 0.3 seconds
            )
            
            if should_save_debug:
                cnt = getattr(self, "_nc_segment_counter", 0)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
                orig_path = os.path.join(
                    self.debug_save_dir, 
                    f"nc_{ts}_{cnt}_orig_{input_sr}hz.wav"
                )
                self._write_wav_bytes(orig_path, audio_data, input_sr)
            
            # Log input stats
            if self.debug_rms:
                rms_in = float(np.sqrt(np.mean(audio_float ** 2)))
                duration = len(audio_float) / input_sr
                logger.info(f"[NC Input] RMS={rms_in:.4f}, Duration={duration:.2f}s, SR={input_sr}Hz")
            
            # STEP 1: Resample to DeepFilter SR if needed
            df_sr = self.df_state.sr()
            if input_sr != df_sr:
                logger.debug(f"Resampling: {input_sr}Hz → {df_sr}Hz")
                audio_float = resampy.resample(audio_float, input_sr, df_sr)
            
            # STEP 2: Convert to tensor
            audio_tensor = torch.tensor(audio_float).float().unsqueeze(0)  # (1, samples)
            if self.use_gpu:
                audio_tensor = audio_tensor.to(self.device)
            
            # STEP 3: Denoise using DeepFilterNet
            with torch.no_grad():
                from df.enhance import enhance
                enhanced = enhance(
                    self.model,
                    self.df_state,
                    audio_tensor,
                    atten_lim_db=float(self.attenuation_limit)
                )
            
            if self.use_gpu:
                enhanced = enhanced.cpu()
            
            enhanced_float = enhanced.squeeze().numpy()
            
            # Handle multi-channel output (shouldn't happen, but just in case)
            if enhanced_float.ndim > 1:
                logger.warning(f"Multi-channel output detected, averaging")
                enhanced_float = enhanced_float.mean(axis=0)
            
            # STEP 4: Normalize intelligently (like reference implementation)
            max_val = np.max(np.abs(enhanced_float))
            if max_val > 0:
                # This is MUCH better than crude gain multiplication
                enhanced_float = np.clip(
                    enhanced_float / max_val * self.normalization_gain, 
                    -1.0, 
                    1.0
                )
            
            # Log after NC stats
            if self.debug_rms:
                rms_out = float(np.sqrt(np.mean(enhanced_float ** 2)))
                logger.info(f"[NC Output] RMS={rms_out:.4f} (gain: {rms_out/rms_in:.2f}x)")
            
            # STEP 5: Resample to target SR (16kHz for Whisper)
            if df_sr != self.target_sr:
                logger.debug(f"Resampling: {df_sr}Hz → {self.target_sr}Hz")
                enhanced_float = resampy.resample(enhanced_float, df_sr, self.target_sr)
            
            # STEP 6: Convert back to int16
            enhanced_int16 = np.clip(
                enhanced_float * 32768.0, 
                -32768, 
                32767
            ).astype(np.int16)
            
            # Log final stats
            if self.debug_rms:
                zeros_pct = 100.0 * float(np.mean(enhanced_int16 == 0))
                logger.info(f"[NC Final] Zeros={zeros_pct:.1f}%, Samples={len(enhanced_int16)}")
            
            # Save processed audio for debugging
            if should_save_debug:
                cnt = getattr(self, "_nc_segment_counter", 0)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
                proc_path = os.path.join(
                    self.debug_save_dir,
                    f"nc_{ts}_{cnt}_proc_{self.target_sr}hz.wav"
                )
                self._write_wav_bytes(proc_path, enhanced_int16.tobytes(), self.target_sr)
                self._nc_segment_counter = cnt + 1
            
            return enhanced_int16.tobytes()
            
        except Exception as e:
            logger.exception(f"Error in noise cancellation: {e}")
            # Return original on error
            return audio_data
    
    def reset_state(self):
        """Reset internal state between calls"""
        # DeepFilterNet is mostly stateless, but good practice
        pass
    
    def __del__(self):
        """Cleanup"""
        try:
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
            if getattr(self, "use_gpu", False):
                torch.cuda.empty_cache()
        except Exception:
            pass


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_improved_nc_instance: Optional[ImprovedNoiseCanceller] = None


def get_improved_noise_canceller(**kwargs) -> ImprovedNoiseCanceller:
    """Get or create shared ImprovedNoiseCanceller instance"""
    global _improved_nc_instance
    
    if _improved_nc_instance is None:
        _improved_nc_instance = ImprovedNoiseCanceller(**kwargs)
        _improved_nc_instance.load_model()
    
    return _improved_nc_instance


def denoise_utterance(audio_data: bytes, input_sr: int = 16000) -> bytes:
    """
    Convenience function to denoise complete utterance
    
    Args:
        audio_data: Complete utterance (int16 PCM bytes, mono)
        input_sr: Input sample rate
        
    Returns:
        Denoised audio bytes
    """
    nc = get_improved_noise_canceller()
    return nc.process_utterance(audio_data, input_sr)
