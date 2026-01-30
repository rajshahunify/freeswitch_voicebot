# # # """
# # # FreeSWITCH VoiceBot - Main WebSocket Server
# # # Complete audio processing pipeline with noise cancellation and VAD
# # # """

# # # import uvicorn
# # # from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# # # import asyncio
# # # import logging
# # # import subprocess
# # # import json
# # # import time
# # # from concurrent.futures import ThreadPoolExecutor

# # # # Import our modules
# # # import config
# # # from audio_pipeline import (
# # #     get_noise_canceller,
# # #     get_vad_detector,
# # #     CallAudioManager
# # # )
# # # from ivr import IntentMatcher, ResponseHandler
# # # from stt_handler import STTHandler

# # # # =============================================================================
# # # # LOGGING SETUP
# # # # =============================================================================
# # # logging.basicConfig(
# # #     level=getattr(logging, config.LOG_LEVEL),
# # #     format=config.LOG_FORMAT,
# # #     handlers=[
# # #         logging.FileHandler(config.LOG_FILE),
# # #         logging.StreamHandler()
# # #     ]
# # # )

# # # logger = logging.getLogger(__name__)

# # # # Filter out spam logs
# # # class SpamFilter(logging.Filter):
# # #     def filter(self, record):
# # #         spam_phrases = ["[End of Speech]", "Speech probability"]
# # #         return not any(phrase in record.getMessage() for phrase in spam_phrases)

# # # logging.getLogger().addFilter(SpamFilter())

# # # # =============================================================================
# # # # INITIALIZE COMPONENTS
# # # # =============================================================================
# # # app = FastAPI(title="FreeSWITCH VoiceBot")

# # # # Thread pool for CPU-intensive tasks
# # # executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# # # # Initialize components (loaded once, shared across all calls)
# # # logger.info("=" * 60)
# # # logger.info("🚀 Initializing VoiceBot Components")
# # # logger.info("=" * 60)

# # # # Audio processing pipeline
# # # noise_canceller = get_noise_canceller(
# # #     model_name=config.DF_MODEL,
# # #     use_gpu=config.DF_USE_GPU,
# # #     post_filter=config.DF_POST_FILTER
# # # )

# # # vad_detector = get_vad_detector(
# # #     threshold=config.VAD_THRESHOLD,
# # #     min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
# # #     min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
# # #     sample_rate=config.VAD_SAMPLE_RATE,
# # #     window_size=config.VAD_WINDOW_SIZE
# # # )

# # # # Buffer manager (separate buffer per call)
# # # buffer_manager = CallAudioManager(
# # #     min_length=config.MIN_AUDIO_LENGTH_BYTES,
# # #     max_length=config.MAX_AUDIO_LENGTH_BYTES,
# # #     timeout_seconds=config.BUFFER_TIMEOUT_SECONDS
# # # )

# # # # IVR components
# # # intent_matcher = IntentMatcher(
# # #     intent_keywords=config.INTENT_KEYWORDS,
# # #     fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
# # # )

# # # response_handler = ResponseHandler(
# # #     audio_base_path=config.AUDIO_BASE_PATH,
# # #     allow_interruptions=config.ALLOW_INTERRUPTIONS,
# # #     speaking_timeout=config.BOT_SPEAKING_TIMEOUT
# # # )

# # # # STT handler
# # # stt_handler = STTHandler(
# # #     stt_url=config.STT_URL,
# # #     stt_params=config.STT_PARAMS,
# # #     timeout=config.STT_TIMEOUT
# # # )

# # # logger.info("✓ All components initialized")
# # # logger.info("=" * 60)

# # # # =============================================================================
# # # # HELPER FUNCTIONS
# # # # =============================================================================

# # # async def get_active_call_uuid(retries: int = 10, delay: float = 0.2) -> str:
# # #     """
# # #     Get the most recent active call UUID from FreeSWITCH
    
# # #     Args:
# # #         retries: Number of retry attempts
# # #         delay: Delay between retries
        
# # #     Returns:
# # #         Call UUID or None
# # #     """
# # #     for i in range(retries):
# # #         try:
# # #             cmd = ["fs_cli", "-x", "show channels as json"]
# # #             process = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
# # #             output = process.stdout.strip()
            
# # #             if output:
# # #                 data = json.loads(output)
# # #                 if data and "rows" in data and len(data["rows"]) > 0:
# # #                     # Get most recent call
# # #                     sorted_calls = sorted(
# # #                         data["rows"],
# # #                         key=lambda x: x.get('created_epoch', 0),
# # #                         reverse=True
# # #                     )
# # #                     uuid = sorted_calls[0]["uuid"]
# # #                     return uuid
# # #         except Exception as e:
# # #             logger.debug(f"Attempt {i+1}/{retries} to get UUID failed: {e}")
        
# # #         await asyncio.sleep(delay)
    
# # #     return None


# # # def process_audio_segment(audio_data: bytes, call_uuid: str):
# # #     """
# # #     Process complete audio segment through full pipeline
# # #     This runs in a thread pool to avoid blocking
    
# # #     Args:
# # #         audio_data: Raw PCM audio bytes
# # #         call_uuid: Call identifier
# # #     """
# # #     # Check if bot is speaking (and interruptions not allowed)
# # #     if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
# # #         logger.debug("🔇 Ignoring audio - bot is speaking")
# # #         return
    
# # #     pipeline_start = time.time()
    
# # #     try:
# # #         # Step 1: Noise Cancellation
# # #         nc_start = time.time()
# # #         enhanced_audio = noise_canceller.process_audio(audio_data)
# # #         nc_time = (time.time() - nc_start) * 1000
        
# # #         # Step 2: STT Transcription
# # #         stt_start = time.time()
# # #         text = stt_handler.transcribe(enhanced_audio)
# # #         stt_time = (time.time() - stt_start) * 1000
        
# # #         if text:
# # #             # Step 3: Intent Matching
# # #             intent_start = time.time()
# # #             audio_file = intent_matcher.match_intent(text)
# # #             intent_time = (time.time() - intent_start) * 1000
            
# # #             # Step 4: Play Response
# # #             response_start = time.time()
# # #             response_handler.play_audio(call_uuid, audio_file, text)
# # #             response_time = (time.time() - response_start) * 1000
            
# # #             # Log performance
# # #             total_time = (time.time() - pipeline_start) * 1000
# # #             if config.ENABLE_TIMING_LOGS:
# # #                 logger.info(
# # #                     f"⏱️  Pipeline: NC={nc_time:.0f}ms, "
# # #                     f"STT={stt_time:.0f}ms, "
# # #                     f"Intent={intent_time:.0f}ms, "
# # #                     f"Response={response_time:.0f}ms, "
# # #                     f"Total={total_time:.0f}ms"
# # #                 )
        
# # #     except Exception as e:
# # #         logger.error(f"❌ Error processing audio segment: {e}", exc_info=True)


# # # # =============================================================================
# # # # WEBSOCKET ENDPOINT
# # # # =============================================================================

# # # @app.websocket("/media")
# # # async def websocket_endpoint(websocket: WebSocket):
# # #     """
# # #     Main WebSocket endpoint for audio streaming
# # #     Handles: Audio reception → NC → VAD → Buffering → STT → Response
# # #     """
# # #     await websocket.accept()
    
# # #     connection_start = time.time()
# # #     call_uuid = None
    
# # #     logger.info("=" * 60)
# # #     logger.info("📞 NEW CALL STARTING")
# # #     logger.info(f"⚙️  NC: {config.DF_MODEL}, VAD: Silero, STT: Whisper")
# # #     logger.info(f"⚙️  Interruptions: {'ON' if config.ALLOW_INTERRUPTIONS else 'OFF'}")
# # #     logger.info("=" * 60)
    
# # #     try:
# # #         # Get call UUID
# # #         call_uuid = await get_active_call_uuid()
        
# # #         if call_uuid:
# # #             connection_time = (time.time() - connection_start) * 1000
# # #             logger.info(f"✓ Connected to call {call_uuid} ({connection_time:.0f}ms)")
            
# # #             # Initialize VAD state for this call
# # #             vad_detector.reset_state()
            
# # #             # Get buffer for this call
# # #             audio_buffer = buffer_manager.get_buffer(call_uuid)
            
# # #             # Stop any existing audio and play welcome
# # #             subprocess.run(
# # #                 ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
# # #                 capture_output=True
# # #             )
# # #             await asyncio.sleep(0.5)
# # #             response_handler.play_audio(call_uuid, "english_menu.wav")
# # #         else:
# # #             logger.error("⚠️  Could not find active call UUID")
# # #             await websocket.close()
# # #             return
        
# # #         # Main audio processing loop
# # #         chunk_count = 0
        
# # #         while True:
# # #             message = await websocket.receive()
            
# # #             # Handle disconnection
# # #             if message["type"] == "websocket.disconnect":
# # #                 logger.info("🚫 Call ended by client")
# # #                 break
            
# # #             # Process audio data
# # #             if "bytes" in message:
# # #                 chunk_count += 1
# # #                 raw_chunk = message["bytes"]
                
# # #                 # Ensure we still have the UUID
# # #                 if not call_uuid:
# # #                     call_uuid = await get_active_call_uuid(retries=1)
# # #                     if not call_uuid:
# # #                         continue
                
# # #                 # PIPELINE STEP 1: Noise Cancellation
# # #                 # Run in executor to avoid blocking event loop
# # #                 enhanced_chunk = await asyncio.get_event_loop().run_in_executor(
# # #                     executor,
# # #                     noise_canceller.process_chunk,
# # #                     raw_chunk
# # #                 )
                
# # #                 # PIPELINE STEP 2: VAD Detection
# # #                 vad_result = await asyncio.get_event_loop().run_in_executor(
# # #                     executor,
# # #                     vad_detector.process_stream,
# # #                     enhanced_chunk
# # #                 )
                
# # #                 # PIPELINE STEP 3: Buffer Management
# # #                 ready_audio = audio_buffer.add_chunk(enhanced_chunk, vad_result)
                
# # #                 # PIPELINE STEP 4: Process complete speech segment
# # #                 if ready_audio:
# # #                     logger.info(
# # #                         f"🎤 Speech segment complete "
# # #                         f"({len(ready_audio)} bytes, {chunk_count} chunks)"
# # #                     )
                    
# # #                     # Process in background thread
# # #                     asyncio.get_event_loop().run_in_executor(
# # #                         executor,
# # #                         process_audio_segment,
# # #                         ready_audio,
# # #                         call_uuid
# # #                     )
                    
# # #                     chunk_count = 0
    
# # #     except WebSocketDisconnect:
# # #         logger.info("🚫 WebSocket disconnected")
# # #     except Exception as e:
# # #         logger.error(f"❌ WebSocket error: {e}", exc_info=True)
# # #     finally:
# # #         # Cleanup
# # #         if call_uuid:
# # #             buffer_manager.remove_buffer(call_uuid)
# # #             response_handler.cleanup_call(call_uuid)
# # #             vad_detector.reset_state()
            
# # #             logger.info("=" * 60)
# # #             logger.info("📊 CALL STATISTICS")
# # #             logger.info(f"   STT: {stt_handler.get_stats()}")
# # #             logger.info(f"   Response: {response_handler.get_stats()}")
# # #             logger.info("=" * 60)


# # # # =============================================================================
# # # # HEALTH CHECK ENDPOINT
# # # # =============================================================================

# # # @app.get("/health")
# # # async def health_check():
# # #     """Health check endpoint"""
# # #     return {
# # #         "status": "healthy",
# # #         "components": {
# # #             "noise_canceller": "loaded",
# # #             "vad_detector": "loaded",
# # #             "stt_handler": "ready",
# # #             "intent_matcher": "ready",
# # #             "response_handler": "ready"
# # #         },
# # #         "active_calls": buffer_manager.active_calls(),
# # #         "stt_stats": stt_handler.get_stats()
# # #     }


# # # @app.get("/stats")
# # # async def get_stats():
# # #     """Get detailed statistics"""
# # #     return {
# # #         "stt": stt_handler.get_stats(),
# # #         "response_handler": response_handler.get_stats(),
# # #         "buffer_manager": buffer_manager.get_all_stats(),
# # #         "config": {
# # #             "noise_cancellation": config.DF_MODEL,
# # #             "vad_threshold": config.VAD_THRESHOLD,
# # #             "allow_interruptions": config.ALLOW_INTERRUPTIONS
# # #         }
# # #     }


# # # # =============================================================================
# # # # MAIN
# # # # =============================================================================

# # # if __name__ == "__main__":
# # #     logger.info("🚀 Starting FreeSWITCH VoiceBot Server")
# # #     logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
# # #     logger.info(f"   STT: {config.STT_URL}")
# # #     logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
# # #     logger.info(f"   NC Model: {config.DF_MODEL}")
# # #     logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
# # #     logger.info("=" * 60)
    
# # #     uvicorn.run(
# # #         app,
# # #         host=config.WS_HOST,
# # #         port=config.WS_PORT,
# # #         log_level=config.LOG_LEVEL.lower()
# # #     )


# # """
# # FreeSWITCH VoiceBot - Main WebSocket Server
# # Complete audio processing pipeline with noise cancellation and VAD
# # """

# # import uvicorn
# # from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# # import asyncio
# # import logging
# # import subprocess
# # import json
# # import time
# # from concurrent.futures import ThreadPoolExecutor

# # # Import our modules
# # import config
# # from audio_pipeline import (
# #     get_noise_canceller,
# #     get_vad_detector,
# #     CallAudioManager
# # )
# # from ivr import IntentMatcher, ResponseHandler
# # from stt_handler import STTHandler

# # # =============================================================================
# # # LOGGING SETUP
# # # =============================================================================
# # logging.basicConfig(
# #     level=getattr(logging, config.LOG_LEVEL),
# #     format=config.LOG_FORMAT,
# #     handlers=[
# #         logging.FileHandler(config.LOG_FILE),
# #         logging.StreamHandler()
# #     ]
# # )

# # logger = logging.getLogger(__name__)

# # # Filter out spam logs
# # class SpamFilter(logging.Filter):
# #     def filter(self, record):
# #         spam_phrases = ["[End of Speech]", "Speech probability"]
# #         return not any(phrase in record.getMessage() for phrase in spam_phrases)

# # logging.getLogger().addFilter(SpamFilter())

# # # =============================================================================
# # # INITIALIZE COMPONENTS
# # # =============================================================================
# # app = FastAPI(title="FreeSWITCH VoiceBot")

# # # Thread pool for CPU-intensive tasks
# # executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# # # Initialize components (loaded once, shared across all calls)
# # logger.info("=" * 60)
# # logger.info("🚀 Initializing VoiceBot Components")
# # logger.info("=" * 60)

# # # Audio processing pipeline
# # noise_canceller = get_noise_canceller(
# #     model_name=config.DF_MODEL,
# #     use_gpu=config.DF_USE_GPU,
# #     post_filter=config.DF_POST_FILTER
# # )

# # vad_detector = get_vad_detector(
# #     threshold=config.VAD_THRESHOLD,
# #     min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
# #     min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
# #     sample_rate=config.VAD_SAMPLE_RATE,
# #     window_size=config.VAD_WINDOW_SIZE
# # )

# # # Buffer manager (separate buffer per call)
# # buffer_manager = CallAudioManager(
# #     min_length=config.MIN_AUDIO_LENGTH_BYTES,
# #     max_length=config.MAX_AUDIO_LENGTH_BYTES,
# #     timeout_seconds=config.BUFFER_TIMEOUT_SECONDS
# # )

# # # IVR components
# # intent_matcher = IntentMatcher(
# #     intent_keywords=config.INTENT_KEYWORDS,
# #     fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
# # )

# # response_handler = ResponseHandler(
# #     audio_base_path=config.AUDIO_BASE_PATH,
# #     allow_interruptions=config.ALLOW_INTERRUPTIONS,
# #     speaking_timeout=config.BOT_SPEAKING_TIMEOUT
# # )

# # # STT handler
# # stt_handler = STTHandler(
# #     stt_url=config.STT_URL,
# #     stt_params=config.STT_PARAMS,
# #     timeout=config.STT_TIMEOUT
# # )

# # logger.info("✓ All components initialized")
# # logger.info("=" * 60)

# # # =============================================================================
# # # HELPER FUNCTIONS
# # # =============================================================================

# # async def get_active_call_uuid(retries: int = 10, delay: float = 0.2) -> str:
# #     """
# #     Get the most recent active call UUID from FreeSWITCH
    
# #     Args:
# #         retries: Number of retry attempts
# #         delay: Delay between retries
        
# #     Returns:
# #         Call UUID or None
# #     """
# #     for i in range(retries):
# #         try:
# #             cmd = ["fs_cli", "-x", "show channels as json"]
# #             process = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
# #             output = process.stdout.strip()
            
# #             if output:
# #                 data = json.loads(output)
# #                 if data and "rows" in data and len(data["rows"]) > 0:
# #                     # Get most recent call
# #                     sorted_calls = sorted(
# #                         data["rows"],
# #                         key=lambda x: x.get('created_epoch', 0),
# #                         reverse=True
# #                     )
# #                     uuid = sorted_calls[0]["uuid"]
# #                     return uuid
# #         except Exception as e:
# #             logger.debug(f"Attempt {i+1}/{retries} to get UUID failed: {e}")
        
# #         await asyncio.sleep(delay)
    
# #     return None


# # def process_audio_segment(audio_data: bytes, call_uuid: str):
# #     """
# #     Process complete audio segment through full pipeline
# #     This runs in a thread pool to avoid blocking
    
# #     Args:
# #         audio_data: Raw PCM audio bytes
# #         call_uuid: Call identifier
# #     """
# #     # Check if bot is speaking (and interruptions not allowed)
# #     if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
# #         logger.debug("🔇 Ignoring audio - bot is speaking")
# #         return
    
# #     pipeline_start = time.time()
    
# #     try:
# #         # Step 1: Noise Cancellation - DISABLED (reduces volume)
# #         # nc_start = time.time()
# #         # enhanced_audio = noise_canceller.process_audio(audio_data)
# #         # nc_time = (time.time() - nc_start) * 1000
# #         enhanced_audio = audio_data  # Skip NC
# #         nc_time = 0
        
# #         # Step 2: STT Transcription
# #         stt_start = time.time()
# #         text = stt_handler.transcribe(enhanced_audio)
# #         stt_time = (time.time() - stt_start) * 1000
        
# #         if text:
# #             # Step 3: Intent Matching
# #             intent_start = time.time()
# #             audio_file = intent_matcher.match_intent(text)
# #             intent_time = (time.time() - intent_start) * 1000
            
# #             # Step 4: Play Response
# #             response_start = time.time()
# #             response_handler.play_audio(call_uuid, audio_file, text)
# #             response_time = (time.time() - response_start) * 1000
            
# #             # Log performance
# #             total_time = (time.time() - pipeline_start) * 1000
# #             if config.ENABLE_TIMING_LOGS:
# #                 logger.info(
# #                     f"⏱️  Pipeline: NC={nc_time:.0f}ms, "
# #                     f"STT={stt_time:.0f}ms, "
# #                     f"Intent={intent_time:.0f}ms, "
# #                     f"Response={response_time:.0f}ms, "
# #                     f"Total={total_time:.0f}ms"
# #                 )
        
# #     except Exception as e:
# #         logger.error(f"❌ Error processing audio segment: {e}", exc_info=True)


# # # =============================================================================
# # # WEBSOCKET ENDPOINT
# # # =============================================================================

# # @app.websocket("/media")
# # async def websocket_endpoint(websocket: WebSocket):
# #     """
# #     Main WebSocket endpoint for audio streaming
# #     Handles: Audio reception → NC → VAD → Buffering → STT → Response
# #     """
# #     await websocket.accept()
    
# #     connection_start = time.time()
# #     call_uuid = None
    
# #     logger.info("=" * 60)
# #     logger.info("📞 NEW CALL STARTING")
# #     logger.info(f"⚙️  NC: {config.DF_MODEL}, VAD: Silero, STT: Whisper")
# #     logger.info(f"⚙️  Interruptions: {'ON' if config.ALLOW_INTERRUPTIONS else 'OFF'}")
# #     logger.info("=" * 60)
    
# #     try:
# #         # Get call UUID
# #         call_uuid = await get_active_call_uuid()
        
# #         if call_uuid:
# #             connection_time = (time.time() - connection_start) * 1000
# #             logger.info(f"✓ Connected to call {call_uuid} ({connection_time:.0f}ms)")
            
# #             # Initialize VAD state for this call
# #             vad_detector.reset_state()
            
# #             # Get buffer for this call
# #             audio_buffer = buffer_manager.get_buffer(call_uuid)
            
# #             # Stop any existing audio and play welcome
# #             subprocess.run(
# #                 ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
# #                 capture_output=True
# #             )
# #             await asyncio.sleep(0.5)
# #             response_handler.play_audio(call_uuid, "english_menu.wav")
# #         else:
# #             logger.error("⚠️  Could not find active call UUID")
# #             await websocket.close()
# #             return
        
# #         # Main audio processing loop
# #         chunk_count = 0
# #         vad_speech_count = 0
# #         vad_silence_count = 0
        
# #         while True:
# #             message = await websocket.receive()
            
# #             # Handle disconnection
# #             if message["type"] == "websocket.disconnect":
# #                 logger.info("🚫 Call ended by client")
# #                 break
            
# #             # Process audio data
# #             if "bytes" in message:
# #                 chunk_count += 1
# #                 raw_chunk = message["bytes"]
                
# #                 # Log every 50 chunks to show activity
# #                 if chunk_count % 50 == 0:
# #                     logger.debug(f"📊 Received {chunk_count} chunks ({len(raw_chunk)} bytes each)")
                
# #                 # Ensure we still have the UUID
# #                 if not call_uuid:
# #                     call_uuid = await get_active_call_uuid(retries=1)
# #                     if not call_uuid:
# #                         continue
                
# #                 # PIPELINE STEP 1: Noise Cancellation
# #                 # TEMPORARILY DISABLED - NC reduces volume too much
# #                 # enhanced_chunk = await asyncio.get_event_loop().run_in_executor(
# #                 #     executor,
# #                 #     noise_canceller.process_chunk,
# #                 #     raw_chunk
# #                 # )
# #                 enhanced_chunk = raw_chunk  # Skip NC for now
                
# #                 # PIPELINE STEP 2: VAD Detection
# #                 vad_result = await asyncio.get_event_loop().run_in_executor(
# #                     executor,
# #                     vad_detector.process_stream,
# #                     enhanced_chunk
# #                 )
                
# #                 # Track VAD results
# #                 if vad_result['is_speech']:
# #                     vad_speech_count += 1
# #                 else:
# #                     vad_silence_count += 1
                
# #                 # Log VAD events
# #                 if vad_result.get('speech_start'):
# #                     logger.info(f"🎤 SPEECH START detected (prob: {vad_result['probability']:.2f})")
# #                 if vad_result.get('speech_end'):
# #                     logger.info(f"🎤 SPEECH END detected (speech chunks: {vad_speech_count}, silence chunks: {vad_silence_count})")
# #                     vad_speech_count = 0
# #                     vad_silence_count = 0
                
# #                 # PIPELINE STEP 3: Buffer Management
# #                 ready_audio = audio_buffer.add_chunk(enhanced_chunk, vad_result)
                
# #                 # PIPELINE STEP 4: Process complete speech segment
# #                 if ready_audio:
# #                     logger.info(
# #                         f"🎤 Speech segment complete "
# #                         f"({len(ready_audio)} bytes, {chunk_count} total chunks received)"
# #                     )
                    
# #                     # Process in background thread
# #                     asyncio.get_event_loop().run_in_executor(
# #                         executor,
# #                         process_audio_segment,
# #                         ready_audio,
# #                         call_uuid
# #                     )
    
# #     except WebSocketDisconnect:
# #         logger.info("🚫 WebSocket disconnected")
# #     except Exception as e:
# #         logger.error(f"❌ WebSocket error: {e}", exc_info=True)
# #     finally:
# #         # Cleanup
# #         if call_uuid:
# #             buffer_manager.remove_buffer(call_uuid)
# #             response_handler.cleanup_call(call_uuid)
# #             vad_detector.reset_state()
            
# #             logger.info("=" * 60)
# #             logger.info("📊 CALL STATISTICS")
# #             logger.info(f"   STT: {stt_handler.get_stats()}")
# #             logger.info(f"   Response: {response_handler.get_stats()}")
# #             logger.info("=" * 60)


# # # =============================================================================
# # # HEALTH CHECK ENDPOINT
# # # =============================================================================

# # @app.get("/health")
# # async def health_check():
# #     """Health check endpoint"""
# #     return {
# #         "status": "healthy",
# #         "components": {
# #             "noise_canceller": "loaded",
# #             "vad_detector": "loaded",
# #             "stt_handler": "ready",
# #             "intent_matcher": "ready",
# #             "response_handler": "ready"
# #         },
# #         "active_calls": buffer_manager.active_calls(),
# #         "stt_stats": stt_handler.get_stats()
# #     }


# # @app.get("/stats")
# # async def get_stats():
# #     """Get detailed statistics"""
# #     return {
# #         "stt": stt_handler.get_stats(),
# #         "response_handler": response_handler.get_stats(),
# #         "buffer_manager": buffer_manager.get_all_stats(),
# #         "config": {
# #             "noise_cancellation": config.DF_MODEL,
# #             "vad_threshold": config.VAD_THRESHOLD,
# #             "allow_interruptions": config.ALLOW_INTERRUPTIONS
# #         }
# #     }


# # # =============================================================================
# # # MAIN
# # # =============================================================================

# # if __name__ == "__main__":
# #     logger.info("🚀 Starting FreeSWITCH VoiceBot Server")
# #     logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
# #     logger.info(f"   STT: {config.STT_URL}")
# #     logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
# #     logger.info(f"   NC Model: {config.DF_MODEL}")
# #     logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
# #     logger.info("=" * 60)
    
# #     uvicorn.run(
# #         app,
# #         host=config.WS_HOST,
# #         port=config.WS_PORT,
# #         log_level=config.LOG_LEVEL.lower()
# #     )




# """
# FreeSWITCH VoiceBot - Main WebSocket Server
# Complete audio processing pipeline with noise cancellation and VAD
# """

# import uvicorn
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# import asyncio
# import logging
# import subprocess
# import json
# import time
# from concurrent.futures import ThreadPoolExecutor

# # Import our modules
# import config
# from audio_pipeline import (
#     get_noise_canceller,
#     get_vad_detector,
#     CallAudioManager
# )
# from ivr import IntentMatcher, ResponseHandler
# from stt_handler import STTHandler

# # =============================================================================
# # LOGGING SETUP
# # =============================================================================
# logging.basicConfig(
#     level=getattr(logging, config.LOG_LEVEL),
#     format=config.LOG_FORMAT,
#     handlers=[
#         logging.FileHandler(config.LOG_FILE),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# # Filter out spam logs
# class SpamFilter(logging.Filter):
#     def filter(self, record):
#         spam_phrases = ["[End of Speech]", "Speech probability"]
#         return not any(phrase in record.getMessage() for phrase in spam_phrases)

# logging.getLogger().addFilter(SpamFilter())

# # =============================================================================
# # INITIALIZE COMPONENTS
# # =============================================================================
# app = FastAPI(title="FreeSWITCH VoiceBot")

# # Thread pool for CPU-intensive tasks
# executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# # Initialize components (loaded once, shared across all calls)
# logger.info("=" * 60)
# logger.info("🚀 Initializing VoiceBot Components")
# logger.info("=" * 60)

# # Audio processing pipeline
# # noise_canceller = get_noise_canceller(
# #     model_name=config.DF_MODEL,
# #     use_gpu=config.DF_USE_GPU,
# #     post_filter=config.DF_POST_FILTER,
# #     attenuation_limit=config.DF_ATTENUATION_LIMIT,
# #     gain=config.DF_GAIN,
# #     debug_rms=config.DF_DEBUG_RMS,
# #     debug_save_path=config.DF_DEBUG_SAVE_PATH
# # )
# noise_canceller = get_noise_canceller(
#     model_name=config.DF_MODEL,
#     use_gpu=config.DF_USE_GPU,
#     post_filter=config.DF_POST_FILTER,
#     attenuation_limit=getattr(config, "DF_ATTENUATION_LIMIT", 6.0),
#     gain=1.05,
#     debug_rms=True,
#     debug_save_dir="/mnt/c/Users/unify/freeswitch_voicebot/nc_debug/"   # or another writable path on your machine
# )

# vad_detector = get_vad_detector(
#     threshold=config.VAD_THRESHOLD,
#     min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
#     min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
#     sample_rate=config.VAD_SAMPLE_RATE,
#     window_size=config.VAD_WINDOW_SIZE
# )

# # Buffer manager (separate buffer per call)
# buffer_manager = CallAudioManager(
#     min_length=config.MIN_AUDIO_LENGTH_BYTES,
#     max_length=config.MAX_AUDIO_LENGTH_BYTES,
#     timeout_seconds=config.BUFFER_TIMEOUT_SECONDS
# )

# # IVR components
# intent_matcher = IntentMatcher(
#     intent_keywords=config.INTENT_KEYWORDS,
#     fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
# )

# response_handler = ResponseHandler(
#     audio_base_path=config.AUDIO_BASE_PATH,
#     allow_interruptions=config.ALLOW_INTERRUPTIONS,
#     speaking_timeout=config.BOT_SPEAKING_TIMEOUT
# )

# # STT handler
# stt_handler = STTHandler(
#     stt_url=config.STT_URL,
#     stt_params=config.STT_PARAMS,
#     timeout=config.STT_TIMEOUT
# )

# logger.info("✓ All components initialized")
# logger.info("=" * 60)

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# async def get_active_call_uuid(retries: int = 10, delay: float = 0.2) -> str:
#     """
#     Get the most recent active call UUID from FreeSWITCH
    
#     Args:
#         retries: Number of retry attempts
#         delay: Delay between retries
        
#     Returns:
#         Call UUID or None
#     """
#     for i in range(retries):
#         try:
#             cmd = ["fs_cli", "-x", "show channels as json"]
#             process = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
#             output = process.stdout.strip()
            
#             if output:
#                 data = json.loads(output)
#                 if data and "rows" in data and len(data["rows"]) > 0:
#                     # Get most recent call
#                     sorted_calls = sorted(
#                         data["rows"],
#                         key=lambda x: x.get('created_epoch', 0),
#                         reverse=True
#                     )
#                     uuid = sorted_calls[0]["uuid"]
#                     return uuid
#         except Exception as e:
#             logger.debug(f"Attempt {i+1}/{retries} to get UUID failed: {e}")
        
#         await asyncio.sleep(delay)
    
#     return None


# def process_audio_segment(audio_data: bytes, call_uuid: str):
#     """
#     Process complete audio segment through full pipeline
#     This runs in a thread pool to avoid blocking
    
#     Args:
#         audio_data: Raw PCM audio bytes
#         call_uuid: Call identifier
#     """
#     # Check if bot is speaking (and interruptions not allowed)
#     if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
#         logger.debug("🔇 Ignoring audio - bot is speaking")
#         return
    
#     pipeline_start = time.time()
    
#     try:
#         # Step 1: Noise Cancellation (with volume boost)
#         nc_start = time.time()
#         enhanced_audio = noise_canceller.process_audio(audio_data)
#         nc_time = (time.time() - nc_start) * 1000
        
#         # Step 2: STT Transcription
#         stt_start = time.time()
#         text = stt_handler.transcribe(enhanced_audio)
#         stt_time = (time.time() - stt_start) * 1000
        
#         if text:
#             # Step 3: Intent Matching
#             intent_start = time.time()
#             audio_file = intent_matcher.match_intent(text)
#             intent_time = (time.time() - intent_start) * 1000
            
#             # Step 4: Play Response
#             response_start = time.time()
#             response_handler.play_audio(call_uuid, audio_file, text)
#             response_time = (time.time() - response_start) * 1000
            
#             # Log performance
#             total_time = (time.time() - pipeline_start) * 1000
#             if config.ENABLE_TIMING_LOGS:
#                 logger.info(
#                     f"⏱️  Pipeline: NC={nc_time:.0f}ms, "
#                     f"STT={stt_time:.0f}ms, "
#                     f"Intent={intent_time:.0f}ms, "
#                     f"Response={response_time:.0f}ms, "
#                     f"Total={total_time:.0f}ms"
#                 )
        
#     except Exception as e:
#         logger.error(f"❌ Error processing audio segment: {e}", exc_info=True)


# # =============================================================================
# # WEBSOCKET ENDPOINT
# # =============================================================================

# @app.websocket("/media")
# async def websocket_endpoint(websocket: WebSocket):
#     """
#     Main WebSocket endpoint for audio streaming
#     Handles: Audio reception → NC → VAD → Buffering → STT → Response
#     """
#     await websocket.accept()
    
#     connection_start = time.time()
#     call_uuid = None
    
#     logger.info("=" * 60)
#     logger.info("📞 NEW CALL STARTING")
#     logger.info(f"⚙️  NC: {config.DF_MODEL}, VAD: Silero, STT: Whisper")
#     logger.info(f"⚙️  Interruptions: {'ON' if config.ALLOW_INTERRUPTIONS else 'OFF'}")
#     logger.info("=" * 60)
    
#     try:
#         # Get call UUID
#         call_uuid = await get_active_call_uuid()
        
#         if call_uuid:
#             connection_time = (time.time() - connection_start) * 1000
#             logger.info(f"✓ Connected to call {call_uuid} ({connection_time:.0f}ms)")
            
#             # Initialize VAD state for this call
#             vad_detector.reset_state()
            
#             # Get buffer for this call
#             audio_buffer = buffer_manager.get_buffer(call_uuid)
            
#             # Stop any existing audio and play welcome
#             subprocess.run(
#                 ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
#                 capture_output=True
#             )
#             await asyncio.sleep(0.5)
#             response_handler.play_audio(call_uuid, "english_menu.wav")
#         else:
#             logger.error("⚠️  Could not find active call UUID")
#             await websocket.close()
#             return
        
#         # Main audio processing loop
#         chunk_count = 0
#         vad_speech_count = 0
#         vad_silence_count = 0
        
#         while True:
#             message = await websocket.receive()
            
#             # Handle disconnection
#             if message["type"] == "websocket.disconnect":
#                 logger.info("🚫 Call ended by client")
#                 break
            
#             # Process audio data
#             if "bytes" in message:
#                 chunk_count += 1
#                 raw_chunk = message["bytes"]
                
#                 # Log every 50 chunks to show activity
#                 if chunk_count % 50 == 0:
#                     logger.debug(f"📊 Received {chunk_count} chunks ({len(raw_chunk)} bytes each)")
                
#                 # Ensure we still have the UUID
#                 if not call_uuid:
#                     call_uuid = await get_active_call_uuid(retries=1)
#                     if not call_uuid:
#                         continue
                
#                 # PIPELINE STEP 1: Noise Cancellation (WITH VOLUME BOOST)
#                 enhanced_chunk = await asyncio.get_event_loop().run_in_executor(
#                     executor,
#                     noise_canceller.process_chunk,
#                     raw_chunk
#                 )
                
#                 # PIPELINE STEP 2: VAD Detection
#                 vad_result = await asyncio.get_event_loop().run_in_executor(
#                     executor,
#                     vad_detector.process_stream,
#                     enhanced_chunk
#                 )
                
#                 # Track VAD results
#                 if vad_result['is_speech']:
#                     vad_speech_count += 1
#                 else:
#                     vad_silence_count += 1
                
#                 # Log VAD events
#                 if vad_result.get('speech_start'):
#                     logger.info(f"🎤 SPEECH START detected (prob: {vad_result['probability']:.2f})")
#                 if vad_result.get('speech_end'):
#                     logger.info(f"🎤 SPEECH END detected (speech chunks: {vad_speech_count}, silence chunks: {vad_silence_count})")
#                     vad_speech_count = 0
#                     vad_silence_count = 0
                
#                 # PIPELINE STEP 3: Buffer Management
#                 ready_audio = audio_buffer.add_chunk(enhanced_chunk, vad_result)
                
#                 # PIPELINE STEP 4: Process complete speech segment
#                 if ready_audio:
#                     logger.info(
#                         f"🎤 Speech segment complete "
#                         f"({len(ready_audio)} bytes, {chunk_count} total chunks received)"
#                     )
                    
#                     # Process in background thread
#                     asyncio.get_event_loop().run_in_executor(
#                         executor,
#                         process_audio_segment,
#                         ready_audio,
#                         call_uuid
#                     )
    
#     except WebSocketDisconnect:
#         logger.info("🚫 WebSocket disconnected")
#     except Exception as e:
#         logger.error(f"❌ WebSocket error: {e}", exc_info=True)
#     finally:
#         # Cleanup
#         if call_uuid:
#             buffer_manager.remove_buffer(call_uuid)
#             response_handler.cleanup_call(call_uuid)
#             vad_detector.reset_state()
            
#             logger.info("=" * 60)
#             logger.info("📊 CALL STATISTICS")
#             logger.info(f"   STT: {stt_handler.get_stats()}")
#             logger.info(f"   Response: {response_handler.get_stats()}")
#             logger.info("=" * 60)


# # =============================================================================
# # HEALTH CHECK ENDPOINT
# # =============================================================================

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "components": {
#             "noise_canceller": "loaded",
#             "vad_detector": "loaded",
#             "stt_handler": "ready",
#             "intent_matcher": "ready",
#             "response_handler": "ready"
#         },
#         "active_calls": buffer_manager.active_calls(),
#         "stt_stats": stt_handler.get_stats()
#     }


# @app.get("/stats")
# async def get_stats():
#     """Get detailed statistics"""
#     return {
#         "stt": stt_handler.get_stats(),
#         "response_handler": response_handler.get_stats(),
#         "buffer_manager": buffer_manager.get_all_stats(),
#         "config": {
#             "noise_cancellation": config.DF_MODEL,
#             "vad_threshold": config.VAD_THRESHOLD,
#             "allow_interruptions": config.ALLOW_INTERRUPTIONS
#         }
#     }


# # =============================================================================
# # MAIN
# # =============================================================================

# if __name__ == "__main__":
#     logger.info("🚀 Starting FreeSWITCH VoiceBot Server")
#     logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
#     logger.info(f"   STT: {config.STT_URL}")
#     logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
#     logger.info(f"   NC Model: {config.DF_MODEL}")
#     logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
#     logger.info("=" * 60)
    
#     uvicorn.run(
#         app,
#         host=config.WS_HOST,
#         port=config.WS_PORT,
#         log_level=config.LOG_LEVEL.lower()
#     )


"""
FreeSWITCH VoiceBot - Main WebSocket Server
Complete audio processing pipeline with noise cancellation and VAD
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import logging
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Import our modules
import config
from audio_pipeline import (
    get_noise_canceller,
    get_vad_detector,
    CallAudioManager
)
from ivr import IntentMatcher, ResponseHandler
from stt_handler import STTHandler

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Filter out spam logs
class SpamFilter(logging.Filter):
    def filter(self, record):
        spam_phrases = ["[End of Speech]", "Speech probability"]
        return not any(phrase in record.getMessage() for phrase in spam_phrases)

logging.getLogger().addFilter(SpamFilter())

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================
app = FastAPI(title="FreeSWITCH VoiceBot")

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Initialize components (loaded once, shared across all calls)
logger.info("=" * 60)
logger.info("🚀 Initializing VoiceBot Components")
logger.info("=" * 60)

# Audio processing pipeline with debug enabled
noise_canceller = get_noise_canceller(
    model_name=config.DF_MODEL,
    use_gpu=config.DF_USE_GPU,
    post_filter=config.DF_POST_FILTER,
    attenuation_limit=config.DF_ATTENUATION_LIMIT,
    gain=config.DF_GAIN,
    debug_rms=config.DF_DEBUG_RMS,
    debug_save_dir=config.DF_DEBUG_SAVE_DIR  # Enable debug audio saving
)

vad_detector = get_vad_detector(
    threshold=config.VAD_THRESHOLD,
    min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
    sample_rate=config.VAD_SAMPLE_RATE,
    window_size=config.VAD_WINDOW_SIZE
)

# Buffer manager (separate buffer per call)
buffer_manager = CallAudioManager(
    min_length=config.MIN_AUDIO_LENGTH_BYTES,
    max_length=config.MAX_AUDIO_LENGTH_BYTES,
    timeout_seconds=config.BUFFER_TIMEOUT_SECONDS
)

# IVR components
intent_matcher = IntentMatcher(
    intent_keywords=config.INTENT_KEYWORDS,
    fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
)

response_handler = ResponseHandler(
    audio_base_path=config.AUDIO_BASE_PATH,
    allow_interruptions=config.ALLOW_INTERRUPTIONS,
    speaking_timeout=config.BOT_SPEAKING_TIMEOUT
)

# STT handler
stt_handler = STTHandler(
    stt_url=config.STT_URL,
    stt_params=config.STT_PARAMS,
    timeout=config.STT_TIMEOUT
)

logger.info("✓ All components initialized")
if config.DF_DEBUG_SAVE_DIR:
    logger.info(f"📁 Debug audio files will be saved to: {config.DF_DEBUG_SAVE_DIR}")
logger.info("=" * 60)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_active_call_uuid(retries: int = 10, delay: float = 0.2) -> str:
    """
    Get the most recent active call UUID from FreeSWITCH
    
    Args:
        retries: Number of retry attempts
        delay: Delay between retries
        
    Returns:
        Call UUID or None
    """
    for i in range(retries):
        try:
            cmd = ["fs_cli", "-x", "show channels as json"]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            output = process.stdout.strip()
            
            if output:
                data = json.loads(output)
                if data and "rows" in data and len(data["rows"]) > 0:
                    # Get most recent call
                    sorted_calls = sorted(
                        data["rows"],
                        key=lambda x: x.get('created_epoch', 0),
                        reverse=True
                    )
                    uuid = sorted_calls[0]["uuid"]
                    return uuid
        except Exception as e:
            logger.debug(f"Attempt {i+1}/{retries} to get UUID failed: {e}")
        
        await asyncio.sleep(delay)
    
    return None


def process_audio_segment(audio_data: bytes, call_uuid: str):
    """
    Process complete audio segment through full pipeline
    This runs in a thread pool to avoid blocking
    
    Args:
        audio_data: Raw PCM audio bytes
        call_uuid: Call identifier
    """
    # Check if bot is speaking (and interruptions not allowed)
    if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
        logger.debug("🔇 Ignoring audio - bot is speaking")
        return
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Noise Cancellation (with volume boost and debug saving)
        nc_start = time.time()
        enhanced_audio = noise_canceller.process_audio(audio_data)
        nc_time = (time.time() - nc_start) * 1000
        
        # Step 2: STT Transcription
        stt_start = time.time()
        text = stt_handler.transcribe(enhanced_audio)
        stt_time = (time.time() - stt_start) * 1000
        
        if text:
            # Step 3: Intent Matching
            intent_start = time.time()
            audio_file = intent_matcher.match_intent(text)
            intent_time = (time.time() - intent_start) * 1000
            
            # Step 4: Play Response
            response_start = time.time()
            response_handler.play_audio(call_uuid, audio_file, text)
            response_time = (time.time() - response_start) * 1000
            
            # Log performance
            total_time = (time.time() - pipeline_start) * 1000
            if config.ENABLE_TIMING_LOGS:
                logger.info(
                    f"⏱️  Pipeline: NC={nc_time:.0f}ms, "
                    f"STT={stt_time:.0f}ms, "
                    f"Intent={intent_time:.0f}ms, "
                    f"Response={response_time:.0f}ms, "
                    f"Total={total_time:.0f}ms"
                )
        
    except Exception as e:
        logger.error(f"❌ Error processing audio segment: {e}", exc_info=True)


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for audio streaming
    Handles: Audio reception → NC → VAD → Buffering → STT → Response
    """
    await websocket.accept()
    
    connection_start = time.time()
    call_uuid = None
    
    logger.info("=" * 60)
    logger.info("📞 NEW CALL STARTING")
    logger.info(f"⚙️  NC: {config.DF_MODEL} (atten={config.DF_ATTENUATION_LIMIT}dB, gain={config.DF_GAIN}x)")
    logger.info(f"⚙️  VAD: Silero (threshold={config.VAD_THRESHOLD}, silence={config.VAD_MIN_SILENCE_DURATION_MS}ms)")
    logger.info(f"⚙️  STT: Whisper")
    logger.info(f"⚙️  Interruptions: {'ON' if config.ALLOW_INTERRUPTIONS else 'OFF'}")
    if config.DF_DEBUG_SAVE_DIR:
        logger.info(f"⚙️  Debug: Saving audio to {config.DF_DEBUG_SAVE_DIR}")
    logger.info("=" * 60)
    
    try:
        # Get call UUID
        call_uuid = await get_active_call_uuid()
        
        if call_uuid:
            connection_time = (time.time() - connection_start) * 1000
            logger.info(f"✓ Connected to call {call_uuid} ({connection_time:.0f}ms)")
            
            # Initialize VAD state for this call
            vad_detector.reset_state()
            
            # Get buffer for this call
            audio_buffer = buffer_manager.get_buffer(call_uuid)
            
            # Stop any existing audio and play welcome
            subprocess.run(
                ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
                capture_output=True
            )
            await asyncio.sleep(0.5)
            response_handler.play_audio(call_uuid, "english_menu.wav")
        else:
            logger.error("⚠️  Could not find active call UUID")
            await websocket.close()
            return
        
        # Main audio processing loop
        chunk_count = 0
        vad_speech_count = 0
        vad_silence_count = 0
        last_activity_time = time.time()
        
        while True:
            try:
                # Add timeout to receive to prevent hanging
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                last_activity_time = time.time()
                
            except asyncio.TimeoutError:
                # Check if call is still active
                current_uuid = await get_active_call_uuid(retries=1)
                if current_uuid != call_uuid:
                    logger.info("⚠️  Call UUID changed or ended, closing WebSocket")
                    break
                # If call is still active, continue waiting
                logger.debug("WebSocket timeout but call still active, continuing...")
                continue
                
            # Handle disconnection
            if message["type"] == "websocket.disconnect":
                logger.info("🚫 Call ended by client")
                break
            
            # Process audio data
            if "bytes" in message:
                chunk_count += 1
                raw_chunk = message["bytes"]
                
                # Log every 100 chunks to show activity (reduced noise)
                if chunk_count % 100 == 0:
                    logger.debug(f"📊 Received {chunk_count} chunks ({len(raw_chunk)} bytes each)")
                
                # Ensure we still have the UUID
                if not call_uuid:
                    call_uuid = await get_active_call_uuid(retries=1)
                    if not call_uuid:
                        continue
                
                # PIPELINE STEP 1: Noise Cancellation (WITH VOLUME BOOST & DEBUG SAVING)
                try:
                    enhanced_chunk = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        noise_canceller.process_chunk,
                        raw_chunk
                    )
                except Exception as e:
                    logger.error(f"NC processing error: {e}")
                    enhanced_chunk = raw_chunk  # Fallback to raw audio
                
                # PIPELINE STEP 2: VAD Detection
                try:
                    vad_result = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        vad_detector.process_stream,
                        enhanced_chunk
                    )
                except Exception as e:
                    logger.error(f"VAD processing error: {e}")
                    vad_result = {'is_speech': False, 'speech_start': False, 'speech_end': False, 'probability': 0.0}
                
                # Track VAD results
                if vad_result['is_speech']:
                    vad_speech_count += 1
                else:
                    vad_silence_count += 1
                
                # Log VAD events
                if vad_result.get('speech_start'):
                    logger.info(f"🎤 SPEECH START detected (prob: {vad_result['probability']:.2f})")
                if vad_result.get('speech_end'):
                    logger.info(f"🎤 SPEECH END detected (speech chunks: {vad_speech_count}, silence chunks: {vad_silence_count})")
                    vad_speech_count = 0
                    vad_silence_count = 0
                
                # PIPELINE STEP 3: Buffer Management
                ready_audio = audio_buffer.add_chunk(enhanced_chunk, vad_result)
                
                # PIPELINE STEP 4: Process complete speech segment
                if ready_audio:
                    logger.info(
                        f"🎤 Speech segment complete "
                        f"({len(ready_audio)} bytes, {chunk_count} total chunks received)"
                    )
                    
                    # Process in background thread
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_audio_segment,
                        ready_audio,
                        call_uuid
                    )
    
    except WebSocketDisconnect:
        logger.info("🚫 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        if call_uuid:
            buffer_manager.remove_buffer(call_uuid)
            response_handler.cleanup_call(call_uuid)
            vad_detector.reset_state()
            
            logger.info("=" * 60)
            logger.info("📊 CALL STATISTICS")
            logger.info(f"   STT: {stt_handler.get_stats()}")
            logger.info(f"   Response: {response_handler.get_stats()}")
            if config.DF_DEBUG_SAVE_DIR:
                logger.info(f"   Debug audio saved to: {config.DF_DEBUG_SAVE_DIR}")
            logger.info("=" * 60)


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "noise_canceller": "loaded",
            "vad_detector": "loaded",
            "stt_handler": "ready",
            "intent_matcher": "ready",
            "response_handler": "ready"
        },
        "active_calls": buffer_manager.active_calls(),
        "stt_stats": stt_handler.get_stats(),
        "debug_enabled": config.DF_DEBUG_SAVE_DIR is not None
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    return {
        "stt": stt_handler.get_stats(),
        "response_handler": response_handler.get_stats(),
        "buffer_manager": buffer_manager.get_all_stats(),
        "config": {
            "noise_cancellation": config.DF_MODEL,
            "nc_attenuation": config.DF_ATTENUATION_LIMIT,
            "nc_gain": config.DF_GAIN,
            "vad_threshold": config.VAD_THRESHOLD,
            "vad_silence_ms": config.VAD_MIN_SILENCE_DURATION_MS,
            "allow_interruptions": config.ALLOW_INTERRUPTIONS,
            "debug_enabled": config.DF_DEBUG_SAVE_DIR is not None
        }
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting FreeSWITCH VoiceBot Server")
    logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
    logger.info(f"   STT: {config.STT_URL}")
    logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
    logger.info(f"   NC Model: {config.DF_MODEL} (atten={config.DF_ATTENUATION_LIMIT}dB, gain={config.DF_GAIN}x)")
    logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
    logger.info(f"   VAD Silence: {config.VAD_MIN_SILENCE_DURATION_MS}ms")
    if config.DF_DEBUG_SAVE_DIR:
        logger.info(f"   Debug Audio: {config.DF_DEBUG_SAVE_DIR}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=config.WS_HOST,
        port=config.WS_PORT,
        log_level=config.LOG_LEVEL.lower()
    )