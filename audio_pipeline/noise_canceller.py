# """
# Noise Cancellation using DeepFilterNet2
# Removes background noise from audio in real-time
# """

# import numpy as np
# import torch
# import logging
# from typing import Optional
# import time

# logger = logging.getLogger(__name__)


# class NoiseCanceller:
#     """
#     Real-time noise cancellation using DeepFilterNet2
    
#     Features:
#     - Loads model once, reuses for all calls
#     - Processes audio in chunks (32ms frames)
#     - Maintains state for streaming audio
#     - GPU support with CPU fallback
#     """
    
#     def __init__(self, 
#                  model_name: str = 'DeepFilterNet2',
#                  use_gpu: bool = False,
#                  post_filter: bool = True,
#                  compensate_delay: bool = True,
#                  attenuation_limit: float = 100.0):
#         """
#         Initialize noise canceller
        
#         Args:
#             model_name: 'DeepFilterNet2' or 'DeepFilterNet3'
#             use_gpu: Use CUDA if available
#             post_filter: Apply perceptual post-filtering
#             compensate_delay: Compensate for processing delay
#             attenuation_limit: Max noise reduction in dB
#         """
#         self.model_name = model_name
#         self.use_gpu = use_gpu and torch.cuda.is_available()
#         self.post_filter = post_filter
#         self.compensate_delay = compensate_delay
#         self.attenuation_limit = attenuation_limit
        
#         self.model = None
#         self.df_state = None
#         self.device = None
        
#         logger.info(f"Initializing NoiseCanceller: {model_name}")
#         logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}")
        
#     def load_model(self):
#         """Load DeepFilterNet model (called once at startup)"""
#         if self.model is not None:
#             logger.debug("Model already loaded")
#             return
        
#         start_time = time.time()
        
#         try:
#             from df.enhance import enhance, init_df, load_audio, save_audio
#             from df.io import resample
            
#             # Load model and state
#             self.model, self.df_state, _ = init_df(
#                 model_base_dir=None,  # Auto-download if needed
#                 post_filter=self.post_filter,
#                 log_level="WARNING"
#             )
            
#             # Set device
#             if self.use_gpu:
#                 self.device = torch.device("cuda")
#                 self.model = self.model.to(self.device)
#                 logger.info("✓ Model loaded on GPU")
#             else:
#                 self.device = torch.device("cpu")
#                 logger.info("✓ Model loaded on CPU")
            
#             load_time = time.time() - start_time
#             logger.info(f"Model loaded in {load_time:.2f}s")
            
#         except Exception as e:
#             logger.error(f"Failed to load DeepFilterNet model: {e}")
#             logger.error("Install with: pip install deepfilternet --break-system-packages")
#             raise
    
#     def process_chunk(self, audio_chunk: bytes) -> bytes:
#         """
#         Process a single audio chunk (32ms frame)
        
#         Args:
#             audio_chunk: Raw PCM audio bytes (int16, mono, 16kHz)
            
#         Returns:
#             Enhanced audio bytes (same format)
#         """
#         if self.model is None:
#             self.load_model()
        
#         try:
#             # Convert bytes to numpy array
#             audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
#             # Normalize to float32 [-1, 1]
#             audio_float = audio_np.astype(np.float32) / 32768.0
            
#             # Convert to torch tensor [1, samples]
#             audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
            
#             if self.use_gpu:
#                 audio_tensor = audio_tensor.to(self.device)
            
#             # Enhance audio
#             with torch.no_grad():
#                 from df.enhance import enhance
#                 enhanced = enhance(
#                     self.model,
#                     self.df_state,
#                     audio_tensor,
#                     atten_lim_db=self.attenuation_limit
#                 )
            
#             # Convert back to CPU if needed
#             if self.use_gpu:
#                 enhanced = enhanced.cpu()
            
#             # # Convert back to int16
#             # enhanced_np = enhanced.squeeze().numpy()
#             # # enhanced_np = np.clip(enhanced_np * 32768.0, -32768, 32767)
#             # # In noise_canceller.py - Added 2.5x volume boost:
#             # enhanced_np = np.clip(enhanced_np * 2.5, -1.0, 1.0)
#             # enhanced_int16 = enhanced_np.astype(np.int16)
            
#             # return enhanced_int16.tobytes()
#             # In noise_canceller.py, in the enhance_audio method:

#             enhanced_np = enhanced.squeeze().numpy()
#             enhanced_np = enhanced_np * 2.5  # Apply 2.5x gain
#             enhanced_int16 = np.clip(enhanced_np * 32768.0, -32768, 32767).astype(np.int16)
#             # # Apply 2.5x gain AFTER converting to int16
#             # enhanced_int16 = (enhanced_np * 32768.0).astype(np.int16)
#             # enhanced_int16 = np.clip(enhanced_int16.astype(np.float32) * 2.5, -32768, 32767).astype(np.int16)

#             return enhanced_int16.tobytes()
        
#         except Exception as e:
#             logger.error(f"Error processing audio chunk: {e}")
#             # Return original audio on error
#             return audio_chunk
    
#     def process_audio(self, audio_data: bytes) -> bytes:
#         """
#         Process longer audio buffer
        
#         Args:
#             audio_data: Raw PCM audio bytes
            
#         Returns:
#             Enhanced audio bytes
#         """
#         # For longer audio, we can process it all at once
#         # or in chunks - for now, process all at once
#         return self.process_chunk(audio_data)
    
#     def reset_state(self):
#         """Reset internal state (call between different calls)"""
#         # DeepFilterNet is mostly stateless for our use case
#         # State is maintained per-chunk automatically
#         pass
    
#     def __del__(self):
#         """Cleanup"""
#         if self.model is not None:
#             del self.model
#             self.model = None
#             if self.use_gpu:
#                 torch.cuda.empty_cache()


# # =============================================================================
# # SINGLETON INSTANCE
# # =============================================================================
# # Create a single shared instance for all calls
# _noise_canceller_instance: Optional[NoiseCanceller] = None


# def get_noise_canceller(
#     model_name: str = 'DeepFilterNet2',
#     use_gpu: bool = False,
#     post_filter: bool = True,
#     **kwargs
# ) -> NoiseCanceller:
#     """
#     Get or create the shared NoiseCanceller instance
    
#     Returns:
#         Shared NoiseCanceller instance
#     """
#     global _noise_canceller_instance
    
#     if _noise_canceller_instance is None:
#         _noise_canceller_instance = NoiseCanceller(
#             model_name=model_name,
#             use_gpu=use_gpu,
#             post_filter=post_filter,
#             **kwargs
#         )
#         _noise_canceller_instance.load_model()
    
#     return _noise_canceller_instance


# # =============================================================================
# # CONVENIENCE FUNCTIONS
# # =============================================================================

# def denoise_audio(audio_data: bytes) -> bytes:
#     """
#     Convenience function to denoise audio
    
#     Args:
#         audio_data: Raw PCM audio bytes (int16, mono, 16kHz)
        
#     Returns:
#         Denoised audio bytes
#     """
#     nc = get_noise_canceller()
#     return nc.process_audio(audio_data)




"""
Noise Cancellation using DeepFilterNet2
Removes background noise from audio in real-time
"""

import numpy as np
import torch
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)


class NoiseCanceller:
    """
    Real-time noise cancellation using DeepFilterNet2
    
    Features:
    - Loads model once, reuses for all calls
    - Processes audio in chunks (32ms frames)
    - Maintains state for streaming audio
    - GPU support with CPU fallback
    """
    
    def __init__(self, 
                 model_name: str = 'DeepFilterNet2',
                 use_gpu: bool = False,
                 post_filter: bool = True,
                 compensate_delay: bool = True,
                 attenuation_limit: float = 100.0):
        """
        Initialize noise canceller
        
        Args:
            model_name: 'DeepFilterNet2' or 'DeepFilterNet3'
            use_gpu: Use CUDA if available
            post_filter: Apply perceptual post-filtering
            compensate_delay: Compensate for processing delay
            attenuation_limit: Max noise reduction in dB
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.post_filter = post_filter
        self.compensate_delay = compensate_delay
        self.attenuation_limit = attenuation_limit
        
        self.model = None
        self.df_state = None
        self.device = None
        
        logger.info(f"Initializing NoiseCanceller: {model_name}")
        logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}")
        
    def load_model(self):
        """Load DeepFilterNet model (called once at startup)"""
        if self.model is not None:
            logger.debug("Model already loaded")
            return
        
        start_time = time.time()
        
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
            from df.io import resample
            
            # Load model and state
            self.model, self.df_state, _ = init_df(
                model_base_dir=None,  # Auto-download if needed
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
            logger.info(f"Model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load DeepFilterNet model: {e}")
            logger.error("Install with: pip install deepfilternet --break-system-packages")
            raise
    
    def process_chunk(self, audio_chunk: bytes) -> bytes:
        """
        Process a single audio chunk (32ms frame)
        
        Args:
            audio_chunk: Raw PCM audio bytes (int16, mono, 16kHz)
            
        Returns:
            Enhanced audio bytes (same format)
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Normalize to float32 [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to torch tensor [1, samples]
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
            
            if self.use_gpu:
                audio_tensor = audio_tensor.to(self.device)
            
            # Enhance audio with REDUCED attenuation to preserve voice
            with torch.no_grad():
                from df.enhance import enhance
                enhanced = enhance(
                    self.model,
                    self.df_state,
                    audio_tensor,
                    atten_lim_db=6.0  # CRITICAL: Low attenuation preserves voice
                )
            
            # Convert back to CPU if needed
            if self.use_gpu:
                enhanced = enhanced.cpu()
            
            # Convert back to int16 with volume boost
            # (boost compensates for NC reducing overall amplitude)
            enhanced_np = enhanced.squeeze().numpy()
            enhanced_np = enhanced_np * 10.0  # 3x volume boost
            enhanced_int16 = np.clip(enhanced_np * 32768.0, -32768, 32767).astype(np.int16)
            
            return enhanced_int16.tobytes()
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Return original audio on error
            return audio_chunk
    
    def process_audio(self, audio_data: bytes) -> bytes:
        """
        Process longer audio buffer
        
        Args:
            audio_data: Raw PCM audio bytes
            
        Returns:
            Enhanced audio bytes
        """
        # For longer audio, we can process it all at once
        # or in chunks - for now, process all at once
        return self.process_chunk(audio_data)
    
    def reset_state(self):
        """Reset internal state (call between different calls)"""
        # DeepFilterNet is mostly stateless for our use case
        # State is maintained per-chunk automatically
        pass
    
    def __del__(self):
        """Cleanup"""
        if self.model is not None:
            del self.model
            self.model = None
            if self.use_gpu:
                torch.cuda.empty_cache()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
# Create a single shared instance for all calls
_noise_canceller_instance: Optional[NoiseCanceller] = None


def get_noise_canceller(
    model_name: str = 'DeepFilterNet2',
    use_gpu: bool = False,
    post_filter: bool = True,
    **kwargs
) -> NoiseCanceller:
    """
    Get or create the shared NoiseCanceller instance
    
    Returns:
        Shared NoiseCanceller instance
    """
    global _noise_canceller_instance
    
    if _noise_canceller_instance is None:
        _noise_canceller_instance = NoiseCanceller(
            model_name=model_name,
            use_gpu=use_gpu,
            post_filter=post_filter,
            **kwargs
        )
        _noise_canceller_instance.load_model()
    
    return _noise_canceller_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def denoise_audio(audio_data: bytes) -> bytes:
    """
    Convenience function to denoise audio
    
    Args:
        audio_data: Raw PCM audio bytes (int16, mono, 16kHz)
        
    Returns:
        Denoised audio bytes
    """
    nc = get_noise_canceller()
    return nc.process_audio(audio_data)