# # # """
# # # Noise Cancellation using DeepFilterNet2
# # # Removes background noise from audio in real-time
# # # """

# # # import numpy as np
# # # import torch
# # # import logging
# # # from typing import Optional
# # # import time

# # # logger = logging.getLogger(__name__)


# # # class NoiseCanceller:
# # #     """
# # #     Real-time noise cancellation using DeepFilterNet2

# # #     Features:
# # #     - Loads model once, reuses for all calls
# # #     - Processes audio in chunks (32ms frames)
# # #     - Maintains state for streaming audio
# # #     - GPU support with CPU fallback
# # #     """

# # #     def __init__(self,
# # #                  model_name: str = 'DeepFilterNet2',
# # #                  use_gpu: bool = False,
# # #                  post_filter: bool = True,
# # #                  compensate_delay: bool = True,
# # #                  attenuation_limit: float = 6.0,
# # #                  sample_rate: int = 16000,
# # #                  frame_ms: int = 32,
# # #                  gain: float = 1.0,
# # #                  debug_rms: bool = False):
# # #         """
# # #         Initialize noise canceller

# # #         Args:
# # #             model_name: 'DeepFilterNet2' or 'DeepFilterNet3'
# # #             use_gpu: Use CUDA if available
# # #             post_filter: Apply perceptual post-filtering
# # #             compensate_delay: Compensate for processing delay
# # #             attenuation_limit: Max noise reduction in dB (used for atten_lim_db)
# # #             sample_rate: Expected sample rate of audio (must be 16000 for DFN2 typical models)
# # #             frame_ms: Frame length in milliseconds (32 ms recommended)
# # #             gain: Linear gain applied to output (1.0 = no boost)
# # #             debug_rms: If True, log input/output RMS for debugging
# # #         """
# # #         self.model_name = model_name
# # #         self.use_gpu = use_gpu and torch.cuda.is_available()
# # #         self.post_filter = post_filter
# # #         self.compensate_delay = compensate_delay
# # #         self.attenuation_limit = attenuation_limit

# # #         self.sample_rate = sample_rate
# # #         self.frame_ms = frame_ms
# # #         self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))  # samples per frame
# # #         self.gain = gain
# # #         self.debug_rms = debug_rms

# # #         self.model = None
# # #         self.df_state = None
# # #         self.device = None

# # #         logger.info(f"Initializing NoiseCanceller: {model_name}")
# # #         logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}, frame={self.frame_size} samples")

# # #     def load_model(self):
# # #         """Load DeepFilterNet model (called once at startup)"""
# # #         if self.model is not None:
# # #             logger.debug("Model already loaded")
# # #             return

# # #         start_time = time.time()

# # #         try:
# # #             from df.enhance import enhance, init_df, load_audio, save_audio
# # #             from df.io import resample

# # #             # Load model and state
# # #             self.model, self.df_state, _ = init_df(
# # #                 model_base_dir=None,  # Auto-download if needed
# # #                 post_filter=self.post_filter,
# # #                 log_level="WARNING"
# # #             )

# # #             # Set device
# # #             if self.use_gpu:
# # #                 self.device = torch.device("cuda")
# # #                 self.model = self.model.to(self.device)
# # #                 logger.info("✓ Model loaded on GPU")
# # #             else:
# # #                 self.device = torch.device("cpu")
# # #                 logger.info("✓ Model loaded on CPU")

# # #             load_time = time.time() - start_time
# # #             logger.info(f"Model loaded in {load_time:.2f}s")

# # #         except Exception as e:
# # #             logger.error(f"Failed to load DeepFilterNet model: {e}")
# # #             logger.error("Install with: pip install deepfilternet --break-system-packages")
# # #             raise

# # #     def _process_frame_float(self, audio_float: np.ndarray) -> np.ndarray:
# # #         """
# # #         Process a single frame of float audio in range [-1, 1].
# # #         Expects audio_float.shape == (frame_size,)
# # #         Returns processed float frame with same length.
# # #         """
# # #         # prepare tensor: shape [1, samples]
# # #         audio_tensor = torch.from_numpy(audio_float.astype(np.float32)).unsqueeze(0)
# # #         if self.use_gpu:
# # #             audio_tensor = audio_tensor.to(self.device)

# # #         with torch.no_grad():
# # #             from df.enhance import enhance
# # #             enhanced = enhance(
# # #                 self.model,
# # #                 self.df_state,
# # #                 audio_tensor,
# # #                 atten_lim_db=float(self.attenuation_limit)
# # #             )

# # #         if self.use_gpu:
# # #             enhanced = enhanced.cpu()

# # #         enhanced_np = enhanced.squeeze().numpy()
# # #         return enhanced_np

# # #     def process_audio(self, audio_data: bytes) -> bytes:
# # #         """
# # #         Process arbitrary-length audio buffer (int16 PCM bytes, mono, expected sample rate)
# # #         This does streaming-friendly splitting into frame-size pieces and preserves model state.
# # #         """
# # #         if self.model is None:
# # #             self.load_model()

# # #         try:
# # #             # Convert bytes -> int16 samples
# # #             audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

# # #             if audio_int16.size == 0:
# # #                 return b''

# # #             # Convert to float32 in [-1, 1]
# # #             audio_float = audio_int16.astype(np.float32) / 32768.0

# # #             # Debug input RMS
# # #             if self.debug_rms:
# # #                 rms_in = np.sqrt(np.mean(audio_float ** 2))
# # #                 logger.debug(f"[NC] input RMS={rms_in:.6f} len_samples={audio_float.size}")

# # #             # split into frames of self.frame_size samples
# # #             frame = self.frame_size
# # #             total_samples = audio_float.shape[0]
# # #             n_full = total_samples // frame
# # #             tail = total_samples % frame

# # #             outputs = []
# # #             idx = 0
# # #             for _ in range(n_full):
# # #                 in_frame = audio_float[idx:idx + frame]
# # #                 out_frame = self._process_frame_float(in_frame)
# # #                 outputs.append(out_frame)
# # #                 idx += frame

# # #             # tail: pad with zeros to frame_size
# # #             if tail > 0:
# # #                 last = np.zeros(frame, dtype=np.float32)
# # #                 last[:tail] = audio_float[idx:idx + tail]
# # #                 out_last = self._process_frame_float(last)
# # #                 # only keep original length (avoid adding padded zeros beyond original length)
# # #                 outputs.append(out_last[:tail])

# # #             # concatenate output frames
# # #             if len(outputs) == 0:
# # #                 processed_float = np.array([], dtype=np.float32)
# # #             else:
# # #                 processed_float = np.concatenate(outputs, axis=0)

# # #             # Apply modest gain if configured (avoid huge boosts)
# # #             if self.gain != 1.0:
# # #                 processed_float = processed_float * float(self.gain)

# # #             # Clip and convert back to int16
# # #             processed_int16 = np.clip(processed_float * 32768.0, -32768, 32767).astype(np.int16)

# # #             if self.debug_rms:
# # #                 rms_out = np.sqrt(np.mean(processed_float ** 2)) if processed_float.size else 0.0
# # #                 zeros_pct = 100.0 * np.mean(processed_int16 == 0) if processed_int16.size else 0.0
# # #                 logger.debug(f"[NC] output RMS={rms_out:.6f} zeros%={zeros_pct:.1f}")

# # #             return processed_int16.tobytes()

# # #         except Exception as e:
# # #             logger.error(f"Error processing audio buffer: {e}", exc_info=True)
# # #             # Return original audio on error
# # #             return audio_data

# # #     # keep a compatibility wrapper if other callers call process_chunk
# # #     def process_chunk(self, audio_chunk: bytes) -> bytes:
# # #         """
# # #         Backwards-compatible: process a single chunk of audio (int16 PCM bytes).
# # #         This will route to process_audio (which handles arbitrary length).
# # #         """
# # #         return self.process_audio(audio_chunk)

# # #     def reset_state(self):
# # #         """Reset internal state (call between different calls)"""
# # #         # DeepFilterNet is mostly stateless for our use case
# # #         # State is maintained per-chunk automatically, but you may want to re-init df_state
# # #         pass

# # #     def __del__(self):
# # #         """Cleanup"""
# # #         if self.model is not None:
# # #             del self.model
# # #             self.model = None
# # #             if self.use_gpu:
# # #                 torch.cuda.empty_cache()


# # # # =============================================================================
# # # # SINGLETON INSTANCE
# # # # =============================================================================
# # # # Create a single shared instance for all calls
# # # _noise_canceller_instance: Optional[NoiseCanceller] = None


# # # def get_noise_canceller(
# # #     model_name: str = 'DeepFilterNet2',
# # #     use_gpu: bool = False,
# # #     post_filter: bool = True,
# # #     **kwargs
# # # ) -> NoiseCanceller:
# # #     """
# # #     Get or create the shared NoiseCanceller instance

# # #     Returns:
# # #         Shared NoiseCanceller instance
# # #     """
# # #     global _noise_canceller_instance

# # #     if _noise_canceller_instance is None:
# # #         _noise_canceller_instance = NoiseCanceller(
# # #             model_name=model_name,
# # #             use_gpu=use_gpu,
# # #             post_filter=post_filter,
# # #             **kwargs
# # #         )
# # #         _noise_canceller_instance.load_model()

# # #     return _noise_canceller_instance


# # # # =============================================================================
# # # # CONVENIENCE FUNCTIONS
# # # # =============================================================================

# # # def denoise_audio(audio_data: bytes) -> bytes:
# # #     """
# # #     Convenience function to denoise audio

# # #     Args:
# # #         audio_data: Raw PCM audio bytes (int16, mono, 16kHz)

# # #     Returns:
# # #         Denoised audio bytes
# # #     """
# # #     nc = get_noise_canceller()
# # #     return nc.process_audio(audio_data)



# # """
# # Noise Cancellation using DeepFilterNet2
# # Removes background noise from audio in real-time
# # """

# # import os
# # import wave
# # from datetime import datetime
# # import numpy as np
# # import torch
# # import logging
# # from typing import Optional
# # import time

# # logger = logging.getLogger(__name__)


# # class NoiseCanceller:
# #     """
# #     Real-time noise cancellation using DeepFilterNet2

# #     Features:
# #     - Loads model once, reuses for all calls
# #     - Processes audio in frames (configurable frame_ms)
# #     - Maintains state for streaming audio
# #     - GPU support with CPU fallback
# #     - Optional debug: RMS logging and saving before/after WAVs
# #     """

# #     def __init__(self,
# #                  model_name: str = 'DeepFilterNet2',
# #                  use_gpu: bool = False,
# #                  post_filter: bool = True,
# #                  compensate_delay: bool = True,
# #                  attenuation_limit: float = 6.0,
# #                  sample_rate: int = 16000,
# #                  frame_ms: int = 32,
# #                  gain: float = 1.0,
# #                  debug_rms: bool = False,
# #                  debug_save_dir: Optional[str] = None,
# #                  debug_segment_prefix: str = "nc"):
# #         """
# #         Initialize noise canceller

# #         Args:
# #             model_name: 'DeepFilterNet2' or 'DeepFilterNet3'
# #             use_gpu: Use CUDA if available
# #             post_filter: Apply perceptual post-filtering
# #             compensate_delay: Compensate for processing delay
# #             attenuation_limit: Max noise reduction in dB (used for atten_lim_db)
# #             sample_rate: Expected sample rate of audio (must be 16000 for DFN2 typical models)
# #             frame_ms: Frame length in milliseconds (32 ms recommended)
# #             gain: Linear gain applied to output (1.0 = no boost)
# #             debug_rms: If True, log input/output RMS for debugging
# #             debug_save_dir: If provided, write before/after WAVs for each processed segment
# #             debug_segment_prefix: Prefix for debug files
# #         """
# #         self.model_name = model_name
# #         self.use_gpu = use_gpu and torch.cuda.is_available()
# #         self.post_filter = post_filter
# #         self.compensate_delay = compensate_delay
# #         self.attenuation_limit = attenuation_limit

# #         self.sample_rate = sample_rate
# #         self.frame_ms = frame_ms
# #         self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))  # samples per frame
# #         self.gain = gain
# #         self.debug_rms = debug_rms
# #         self.debug_save_dir = debug_save_dir
# #         self.debug_segment_prefix = debug_segment_prefix

# #         self.model = None
# #         self.df_state = None
# #         self.device = None

# #         if self.debug_save_dir:
# #             os.makedirs(self.debug_save_dir, exist_ok=True)
# #             self._nc_segment_counter = 0

# #         logger.info(f"Initializing NoiseCanceller: {model_name}")
# #         logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}, frame={self.frame_size} samples")

# #     def load_model(self):
# #         """Load DeepFilterNet model (called once at startup)"""
# #         if self.model is not None:
# #             logger.debug("Model already loaded")
# #             return

# #         start_time = time.time()

# #         try:
# #             from df.enhance import enhance, init_df, load_audio, save_audio
# #             from df.io import resample

# #             # Load model and state
# #             self.model, self.df_state, _ = init_df(
# #                 model_base_dir=None,  # Auto-download if needed
# #                 post_filter=self.post_filter,
# #                 log_level="WARNING"
# #             )

# #             # Set device
# #             if self.use_gpu:
# #                 self.device = torch.device("cuda")
# #                 self.model = self.model.to(self.device)
# #                 logger.info("✓ Model loaded on GPU")
# #             else:
# #                 self.device = torch.device("cpu")
# #                 logger.info("✓ Model loaded on CPU")

# #             load_time = time.time() - start_time
# #             logger.info(f"Model loaded in {load_time:.2f}s")

# #         except Exception as e:
# #             logger.error(f"Failed to load DeepFilterNet model: {e}")
# #             logger.error("Install with: pip install deepfilternet --break-system-packages")
# #             raise

# #     def _write_wav_bytes(self, path: str, pcm_bytes: bytes, sample_rate: int = 16000):
# #         """Write int16 PCM bytes (mono) to a WAV file."""
# #         if not pcm_bytes:
# #             return
# #         try:
# #             with wave.open(path, "wb") as wf:
# #                 wf.setnchannels(1)
# #                 wf.setsampwidth(2)  # int16
# #                 wf.setframerate(sample_rate)
# #                 wf.writeframes(pcm_bytes)
# #         except Exception:
# #             logger.exception("Failed to write WAV debug file: %s", path)

# #     def _process_frame_float(self, audio_float: np.ndarray) -> np.ndarray:
# #         """
# #         Process a single frame of float audio in range [-1, 1].
# #         Expects audio_float.shape == (frame_size,) or shorter (padded by caller).
# #         Returns processed float frame with same length.
# #         """
# #         # prepare tensor: shape [1, samples]
# #         audio_tensor = torch.from_numpy(audio_float.astype(np.float32)).unsqueeze(0)
# #         if self.use_gpu:
# #             audio_tensor = audio_tensor.to(self.device)

# #         with torch.no_grad():
# #             from df.enhance import enhance
# #             enhanced = enhance(
# #                 self.model,
# #                 self.df_state,
# #                 audio_tensor,
# #                 atten_lim_db=float(self.attenuation_limit)
# #             )

# #         if self.use_gpu:
# #             enhanced = enhanced.cpu()

# #         enhanced_np = enhanced.squeeze().numpy()
# #         return enhanced_np

# #     def process_audio(self, audio_data: bytes) -> bytes:
# #         """
# #         Process arbitrary-length audio buffer (int16 PCM bytes, mono, expected sample rate)
# #         This does streaming-friendly splitting into frames and preserves model state.
# #         """
# #         if self.model is None:
# #             self.load_model()

# #         try:
# #             # optional: save original bytes for debugging
# #             if self.debug_save_dir:
# #                 cnt = getattr(self, "_nc_segment_counter", 0)
# #                 ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
# #                 orig_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_orig.wav")
# #                 # write original bytes immediately
# #                 self._write_wav_bytes(orig_path, audio_data, sample_rate=self.sample_rate)

# #             # Convert bytes -> int16 samples
# #             audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

# #             if audio_int16.size == 0:
# #                 return b''

# #             # Convert to float32 in [-1, 1]
# #             audio_float = audio_int16.astype(np.float32) / 32768.0

# #             # Debug input RMS
# #             if self.debug_rms:
# #                 rms_in = float(np.sqrt(np.mean(audio_float ** 2)))
# #                 logger.debug(f"[NC] input RMS={rms_in:.6f} len_samples={audio_float.size}")

# #             # split into frames of self.frame_size samples
# #             frame = self.frame_size
# #             total_samples = audio_float.shape[0]
# #             n_full = total_samples // frame
# #             tail = total_samples % frame

# #             outputs = []
# #             idx = 0
# #             for _ in range(n_full):
# #                 in_frame = audio_float[idx:idx + frame]
# #                 out_frame = self._process_frame_float(in_frame)
# #                 outputs.append(out_frame)
# #                 idx += frame

# #             # tail: pad with zeros to frame_size and then crop back
# #             if tail > 0:
# #                 last = np.zeros(frame, dtype=np.float32)
# #                 last[:tail] = audio_float[idx:idx + tail]
# #                 out_last = self._process_frame_float(last)
# #                 outputs.append(out_last[:tail])

# #             # concatenate output frames
# #             if len(outputs) == 0:
# #                 processed_float = np.array([], dtype=np.float32)
# #             else:
# #                 processed_float = np.concatenate(outputs, axis=0)

# #             # Apply modest gain if configured
# #             if self.gain != 1.0:
# #                 processed_float = processed_float * float(self.gain)

# #             # Clip and convert back to int16
# #             processed_int16 = np.clip(processed_float * 32768.0, -32768, 32767).astype(np.int16)

# #             if self.debug_rms:
# #                 rms_out = float(np.sqrt(np.mean(processed_float ** 2))) if processed_float.size else 0.0
# #                 zeros_pct = 100.0 * float(np.mean(processed_int16 == 0)) if processed_int16.size else 0.0
# #                 logger.debug(f"[NC] output RMS={rms_out:.6f} zeros%={zeros_pct:.1f}")

# #             # optional: write processed wav for debugging
# #             if self.debug_save_dir:
# #                 cnt = getattr(self, "_nc_segment_counter", 0)
# #                 ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
# #                 proc_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_proc.wav")
# #                 self._write_wav_bytes(proc_path, processed_int16.tobytes(), sample_rate=self.sample_rate)
# #                 self._nc_segment_counter = cnt + 1

# #             return processed_int16.tobytes()

# #         except Exception as e:
# #             logger.exception(f"Error processing audio buffer: {e}")
# #             # Return original audio on error
# #             return audio_data

# #     # backwards-compatible wrapper that treats chunk same as buffer
# #     def process_chunk(self, audio_chunk: bytes) -> bytes:
# #         """
# #         Backwards-compatible: process a single chunk of audio (int16 PCM bytes).
# #         This will route to process_audio (which handles arbitrary length).
# #         """
# #         return self.process_audio(audio_chunk)

# #     def reset_state(self):
# #         """Reset internal state (call between different calls)"""
# #         # DeepFilterNet is mostly stateless for our use case.
# #         # If you want to fully reset, you could re-init df_state via init_df again.
# #         pass

# #     def __del__(self):
# #         """Cleanup"""
# #         if self.model is not None:
# #             del self.model
# #             self.model = None
# #             if self.use_gpu:
# #                 torch.cuda.empty_cache()


# # # =============================================================================
# # # SINGLETON INSTANCE
# # # =============================================================================
# # # Create a single shared instance for all calls
# # _noise_canceller_instance: Optional[NoiseCanceller] = None


# # def get_noise_canceller(
# #     model_name: str = 'DeepFilterNet2',
# #     use_gpu: bool = False,
# #     post_filter: bool = True,
# #     **kwargs
# # ) -> NoiseCanceller:
# #     """
# #     Get or create the shared NoiseCanceller instance

# #     Returns:
# #         Shared NoiseCanceller instance
# #     """
# #     global _noise_canceller_instance

# #     if _noise_canceller_instance is None:
# #         _noise_canceller_instance = NoiseCanceller(
# #             model_name=model_name,
# #             use_gpu=use_gpu,
# #             post_filter=post_filter,
# #             **kwargs
# #         )
# #         _noise_canceller_instance.load_model()

# #     return _noise_canceller_instance


# # # =============================================================================
# # # CONVENIENCE FUNCTIONS
# # # =============================================================================

# # def denoise_audio(audio_data: bytes) -> bytes:
# #     """
# #     Convenience function to denoise audio

# #     Args:
# #         audio_data: Raw PCM audio bytes (int16, mono, 16kHz)

# #     Returns:
# #         Denoised audio bytes
# #     """
# #     nc = get_noise_canceller()
# #     return nc.process_audio(audio_data)



# """
# Noise Cancellation using DeepFilterNet2
# Removes background noise from audio in real-time
# """

# import os
# import wave
# from datetime import datetime
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
#     - Processes audio in frames (configurable frame_ms)
#     - Maintains state for streaming audio
#     - GPU support with CPU fallback
#     - Optional debug: RMS logging and saving before/after WAVs
#     """

#     def __init__(self,
#                  model_name: str = 'DeepFilterNet2',
#                  use_gpu: bool = False,
#                  post_filter: bool = True,
#                  compensate_delay: bool = True,
#                  attenuation_limit: float = 100.0,
#                  sample_rate: int = 16000,
#                  frame_ms: int = 32,
#                  gain: float = 1.0,
#                  debug_rms: bool = False,
#                  debug_save_dir: Optional[str] = None,
#                  debug_save_path: Optional[str] = None,
#                  debug_segment_prefix: str = "nc"):
#         """
#         Initialize noise canceller

#         Notes:
#             - Accepts both debug_save_dir and debug_save_path (alias) to avoid TypeError
#         """
#         # assign early so __del__ is safe even if __init__ fails later
#         self.model = None
#         self.df_state = None
#         self.device = None

#         self.model_name = model_name
#         self.use_gpu = use_gpu and torch.cuda.is_available()
#         self.post_filter = post_filter
#         self.compensate_delay = compensate_delay
#         self.attenuation_limit = attenuation_limit

#         self.sample_rate = sample_rate
#         self.frame_ms = frame_ms
#         self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))  # samples per frame
#         self.gain = gain
#         self.debug_rms = debug_rms

#         # allow either param name (debug_save_dir OR debug_save_path)
#         self.debug_save_dir = debug_save_dir or debug_save_path
#         self.debug_segment_prefix = debug_segment_prefix

#         if self.debug_save_dir:
#             try:
#                 os.makedirs(self.debug_save_dir, exist_ok=True)
#             except Exception:
#                 logger.exception("Could not create debug_save_dir: %s", self.debug_save_dir)
#             self._nc_segment_counter = 0

#         logger.info(f"Initializing NoiseCanceller: {model_name}")
#         logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}, frame={self.frame_size} samples")

#     def load_model(self):
#         """Load DeepFilterNet model (called once at startup)"""
#         if getattr(self, "model", None) is not None:
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

#     def _write_wav_bytes(self, path: str, pcm_bytes: bytes, sample_rate: int = 16000):
#         """Write int16 PCM bytes (mono) to a WAV file."""
#         if not pcm_bytes:
#             return
#         try:
#             with wave.open(path, "wb") as wf:
#                 wf.setnchannels(1)
#                 wf.setsampwidth(2)  # int16
#                 wf.setframerate(sample_rate)
#                 wf.writeframes(pcm_bytes)
#         except Exception:
#             logger.exception("Failed to write WAV debug file: %s", path)

#     def _process_frame_float(self, audio_float: np.ndarray) -> np.ndarray:
#         """
#         Process a single frame of float audio in range [-1, 1].
#         Expects audio_float.shape == (frame_size,) or shorter (padded by caller).
#         Returns processed float frame with same length.
#         """
#         # prepare tensor: shape [1, samples]
#         audio_tensor = torch.from_numpy(audio_float.astype(np.float32)).unsqueeze(0)
#         if self.use_gpu:
#             audio_tensor = audio_tensor.to(self.device)

#         with torch.no_grad():
#             from df.enhance import enhance
#             enhanced = enhance(
#                 self.model,
#                 self.df_state,
#                 audio_tensor,
#                 atten_lim_db=float(self.attenuation_limit)
#             )

#         if self.use_gpu:
#             enhanced = enhanced.cpu()

#         enhanced_np = enhanced.squeeze().numpy()
#         return enhanced_np

#     def process_audio(self, audio_data: bytes) -> bytes:
#         """
#         Process arbitrary-length audio buffer (int16 PCM bytes, mono, expected sample rate)
#         This does streaming-friendly splitting into frames and preserves model state.
#         """
#         if self.model is None:
#             self.load_model()

#         try:
#             # optional: save original bytes for debugging
#             if self.debug_save_dir:
#                 cnt = getattr(self, "_nc_segment_counter", 0)
#                 ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
#                 orig_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_orig.wav")
#                 # write original bytes immediately
#                 self._write_wav_bytes(orig_path, audio_data, sample_rate=self.sample_rate)

#             # Convert bytes -> int16 samples
#             audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

#             if audio_int16.size == 0:
#                 return b''

#             # Convert to float32 in [-1, 1]
#             audio_float = audio_int16.astype(np.float32) / 32768.0

#             # Debug input RMS
#             if self.debug_rms:
#                 rms_in = float(np.sqrt(np.mean(audio_float ** 2)))
#                 logger.debug(f"[NC] input RMS={rms_in:.6f} len_samples={audio_float.size}")

#             # split into frames of self.frame_size samples
#             frame = self.frame_size
#             total_samples = audio_float.shape[0]
#             n_full = total_samples // frame
#             tail = total_samples % frame

#             outputs = []
#             idx = 0
#             for _ in range(n_full):
#                 in_frame = audio_float[idx:idx + frame]
#                 out_frame = self._process_frame_float(in_frame)
#                 outputs.append(out_frame)
#                 idx += frame

#             # tail: pad with zeros to frame_size and then crop back
#             if tail > 0:
#                 last = np.zeros(frame, dtype=np.float32)
#                 last[:tail] = audio_float[idx:idx + tail]
#                 out_last = self._process_frame_float(last)
#                 outputs.append(out_last[:tail])

#             # concatenate output frames
#             if len(outputs) == 0:
#                 processed_float = np.array([], dtype=np.float32)
#             else:
#                 processed_float = np.concatenate(outputs, axis=0)

#             # Apply modest gain if configured
#             if self.gain != 1.0:
#                 processed_float = processed_float * float(self.gain)

#             # Clip and convert back to int16
#             processed_int16 = np.clip(processed_float * 32768.0, -32768, 32767).astype(np.int16)

#             if self.debug_rms:
#                 rms_out = float(np.sqrt(np.mean(processed_float ** 2))) if processed_float.size else 0.0
#                 zeros_pct = 100.0 * float(np.mean(processed_int16 == 0)) if processed_int16.size else 0.0
#                 logger.debug(f"[NC] output RMS={rms_out:.6f} zeros%={zeros_pct:.1f}")

#             # optional: write processed wav for debugging
#             if self.debug_save_dir:
#                 cnt = getattr(self, "_nc_segment_counter", 0)
#                 ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
#                 proc_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_proc.wav")
#                 self._write_wav_bytes(proc_path, processed_int16.tobytes(), sample_rate=self.sample_rate)
#                 self._nc_segment_counter = cnt + 1

#             return processed_int16.tobytes()

#         except Exception as e:
#             logger.exception(f"Error processing audio buffer: {e}")
#             # Return original audio on error
#             return audio_data

#     # backwards-compatible wrapper that treats chunk same as buffer
#     def process_chunk(self, audio_chunk: bytes) -> bytes:
#         """
#         Backwards-compatible: process a single chunk of audio (int16 PCM bytes).
#         This will route to process_audio (which handles arbitrary length).
#         """
#         return self.process_audio(audio_chunk)

#     def reset_state(self):
#         """Reset internal state (call between different calls)"""
#         # DeepFilterNet is mostly stateless for our use case.
#         # If you want to fully reset, you could re-init df_state via init_df again.
#         pass

#     def __del__(self):
#         """Cleanup safely even if object partially constructed"""
#         try:
#             model = getattr(self, "model", None)
#             if model is not None:
#                 try:
#                     del self.model
#                 except Exception:
#                     pass
#                 self.model = None
#             # free cuda if available
#             if getattr(self, "use_gpu", False):
#                 try:
#                     torch.cuda.empty_cache()
#                 except Exception:
#                     pass
#         except Exception:
#             # never raise in __del__
#             pass


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

import os
import wave
from datetime import datetime
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
    - Processes audio in frames (configurable frame_ms)
    - Maintains state for streaming audio
    - GPU support with CPU fallback
    - Optional debug: RMS logging and saving before/after WAVs
    """

    def __init__(self,
                 model_name: str = 'DeepFilterNet2',
                 use_gpu: bool = False,
                 post_filter: bool = True,
                 compensate_delay: bool = True,
                 attenuation_limit: float = 100.0,
                 sample_rate: int = 16000,
                 frame_ms: int = 32,
                 gain: float = 1.0,
                 debug_rms: bool = False,
                 debug_save_dir: Optional[str] = None,
                 debug_save_path: Optional[str] = None,
                 debug_segment_prefix: str = "nc"):
        """
        Initialize noise canceller

        Notes:
            - Accepts both debug_save_dir and debug_save_path (alias) to avoid TypeError
        """
        # assign early so __del__ is safe even if __init__ fails later
        self.model = None
        self.df_state = None
        self.device = None

        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.post_filter = post_filter
        self.compensate_delay = compensate_delay
        self.attenuation_limit = attenuation_limit

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))  # samples per frame
        self.gain = gain
        self.debug_rms = debug_rms

        # allow either param name (debug_save_dir OR debug_save_path)
        self.debug_save_dir = debug_save_dir or debug_save_path
        self.debug_segment_prefix = debug_segment_prefix

        if self.debug_save_dir:
            try:
                os.makedirs(self.debug_save_dir, exist_ok=True)
            except Exception:
                logger.exception("Could not create debug_save_dir: %s", self.debug_save_dir)
            self._nc_segment_counter = 0

        logger.info(f"Initializing NoiseCanceller: {model_name}")
        logger.info(f"GPU: {self.use_gpu}, Post-filter: {post_filter}, frame={self.frame_size} samples")

    def load_model(self):
        """Load DeepFilterNet model (called once at startup)"""
        if getattr(self, "model", None) is not None:
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

    def _write_wav_bytes(self, path: str, pcm_bytes: bytes, sample_rate: int = 16000):
        """Write int16 PCM bytes (mono) to a WAV file."""
        if not pcm_bytes or len(pcm_bytes) == 0:
            logger.warning(f"Skipping empty WAV file: {path}")
            return
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)
            logger.debug(f"Saved debug audio: {path} ({len(pcm_bytes)} bytes)")
        except Exception:
            logger.exception("Failed to write WAV debug file: %s", path)

    def _process_frame_float(self, audio_float: np.ndarray) -> np.ndarray:
        """
        Process a single frame of float audio in range [-1, 1].
        Expects audio_float.shape == (frame_size,) or shorter (padded by caller).
        Returns processed float frame with same length.
        """
        # prepare tensor: shape [1, samples]
        audio_tensor = torch.from_numpy(audio_float.astype(np.float32)).unsqueeze(0)
        if self.use_gpu:
            audio_tensor = audio_tensor.to(self.device)

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

        enhanced_np = enhanced.squeeze().numpy()
        return enhanced_np

    def process_audio(self, audio_data: bytes) -> bytes:
        """
        Process arbitrary-length audio buffer (int16 PCM bytes, mono, expected sample rate)
        This does streaming-friendly splitting into frames and preserves model state.
        """
        if self.model is None:
            self.load_model()

        try:
            # Convert bytes -> int16 samples
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            if audio_int16.size == 0:
                return b''
            
            # Only save debug files if we have substantial audio (> 0.5 seconds)
            # This prevents saving empty chunks and reduces file spam
            should_save_debug = (
                self.debug_save_dir and 
                len(audio_data) > (self.sample_rate * 2 * 0.5)  # 0.5 seconds of int16 mono
            )
            
            if should_save_debug:
                cnt = getattr(self, "_nc_segment_counter", 0)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
                orig_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_orig.wav")
                # write original bytes immediately
                self._write_wav_bytes(orig_path, audio_data, sample_rate=self.sample_rate)

            # Convert to float32 in [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Debug input RMS
            if self.debug_rms:
                rms_in = float(np.sqrt(np.mean(audio_float ** 2)))
                logger.debug(f"[NC] input RMS={rms_in:.6f} len_samples={audio_float.size}")

            # split into frames of self.frame_size samples
            frame = self.frame_size
            total_samples = audio_float.shape[0]
            n_full = total_samples // frame
            tail = total_samples % frame

            outputs = []
            idx = 0
            for _ in range(n_full):
                in_frame = audio_float[idx:idx + frame]
                out_frame = self._process_frame_float(in_frame)
                outputs.append(out_frame)
                idx += frame

            # tail: pad with zeros to frame_size and then crop back
            if tail > 0:
                last = np.zeros(frame, dtype=np.float32)
                last[:tail] = audio_float[idx:idx + tail]
                out_last = self._process_frame_float(last)
                outputs.append(out_last[:tail])

            # concatenate output frames
            if len(outputs) == 0:
                processed_float = np.array([], dtype=np.float32)
            else:
                processed_float = np.concatenate(outputs, axis=0)

            # Apply modest gain if configured
            if self.gain != 1.0:
                processed_float = processed_float * float(self.gain)

            # Clip and convert back to int16
            processed_int16 = np.clip(processed_float * 32768.0, -32768, 32767).astype(np.int16)

            if self.debug_rms:
                rms_out = float(np.sqrt(np.mean(processed_float ** 2))) if processed_float.size else 0.0
                zeros_pct = 100.0 * float(np.mean(processed_int16 == 0)) if processed_int16.size else 0.0
                logger.debug(f"[NC] output RMS={rms_out:.6f} zeros%={zeros_pct:.1f}")

            # optional: write processed wav for debugging
            if should_save_debug:
                cnt = getattr(self, "_nc_segment_counter", 0)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
                proc_path = os.path.join(self.debug_save_dir, f"{self.debug_segment_prefix}_{ts}_{cnt}_proc.wav")
                self._write_wav_bytes(proc_path, processed_int16.tobytes(), sample_rate=self.sample_rate)
                self._nc_segment_counter = cnt + 1

            return processed_int16.tobytes()

        except Exception as e:
            logger.exception(f"Error processing audio buffer: {e}")
            # Return original audio on error
            return audio_data

    # backwards-compatible wrapper that treats chunk same as buffer
    def process_chunk(self, audio_chunk: bytes) -> bytes:
        """
        Backwards-compatible: process a single chunk of audio (int16 PCM bytes).
        This will route to process_audio (which handles arbitrary length).
        """
        return self.process_audio(audio_chunk)

    def reset_state(self):
        """Reset internal state (call between different calls)"""
        # DeepFilterNet is mostly stateless for our use case.
        # If you want to fully reset, you could re-init df_state via init_df again.
        pass

    def __del__(self):
        """Cleanup safely even if object partially constructed"""
        try:
            model = getattr(self, "model", None)
            if model is not None:
                try:
                    del self.model
                except Exception:
                    pass
                self.model = None
            # free cuda if available
            if getattr(self, "use_gpu", False):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            # never raise in __del__
            pass


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