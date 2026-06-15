"""
FreeSWITCH VoiceBot - Multi-Call WebSocket Server
Handles multiple concurrent calls with Redis-based session management

Pipeline: Audio → NC → VAD → STT → LLM → TTS → Playback
Supports: barge-in, streaming LLM→TTS, switchable providers
"""

# Suppress torchaudio deprecation warnings (floods logs with 6+ lines per TTS call)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torio.io.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import logging
import subprocess
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

# Import our modules
import config
from audio_pipeline import (
    get_improved_noise_canceller,
    CallAudioManager
)
from audio_pipeline.vad_detector import PerCallVADManager
from audio_pipeline.tts_synthesizer import TTSSynthesizer
from ivr import ResponseHandler
from ivr.llm_agent import create_llm_agent
from stt_handler import create_stt_handler
from session_manager import get_session_manager
from latency_tracker import LatencyTracker
from event_emitter import (
    emit_event, emit_transcription, emit_bot_response,
    emit_barge_in, emit_turn_metrics, emit_speech_started, emit_speech_ended
)

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler"""
    # Startup
    asyncio.create_task(cleanup_stale_sessions())
    logger.info("✓ Background cleanup task started")
    
    # Background LLM health check — proactively test Gemini every 60s
    # so we don't waste call time discovering it's down
    asyncio.create_task(llm_health_check_loop())
    logger.info("✓ Background LLM health check started")
    
    yield
    # Shutdown (nothing to do currently)


async def llm_health_check_loop():
    """
    Background health check for Gemini availability.
    
    Only runs when we're on fallback (Ollama). Checks if Gemini has recovered.
    When quota is exhausted, backs off to 30-minute intervals.
    When Gemini is healthy, does NOT ping it (saves API quota).
    
    Also disables FallbackLLM's own internal retry so we don't get
    duplicate Gemini attempts during live calls.
    """
    await asyncio.sleep(2)  # Brief wait for server init
    
    # Disable FallbackLLM's own internal retry — this health check manages it
    if hasattr(llm_agent, '_retry_interval'):
        llm_agent._retry_interval = 999999  # Effectively infinite — health check handles retries
    
    # Initial startup probe: test Gemini ONCE to know the state
    if hasattr(llm_agent, 'primary') and hasattr(llm_agent, 'using_fallback'):
        try:
            test_response = llm_agent.primary.get_response("Say OK", [])
            if test_response and "I'm sorry, I'm having a moment" not in test_response:
                logger.info("✅ Startup check: Gemini is available")
            else:
                logger.warning("⚠️ Startup check: Gemini unavailable — starting with Ollama")
                llm_agent.using_fallback = True
        except Exception:
            logger.warning("⚠️ Startup check: Gemini unreachable — starting with Ollama")
            llm_agent.using_fallback = True
    
    check_interval = 60  # Normal check every 60s
    quota_interval = 1800  # When quota exhausted, check every 30 min
    
    while True:
        try:
            if hasattr(llm_agent, 'primary') and hasattr(llm_agent, 'using_fallback'):
                if not llm_agent.using_fallback:
                    # Gemini is healthy — don't ping, don't waste quota
                    await asyncio.sleep(check_interval)
                    continue
                
                # Currently on Ollama — check if Gemini recovered
                # If daily quota exhausted, use longer interval
                if hasattr(llm_agent.primary, 'daily_quota_exhausted') and llm_agent.primary.daily_quota_exhausted:
                    logger.debug(f"⏳ Gemini daily quota exhausted — next check in {quota_interval}s")
                    await asyncio.sleep(quota_interval)
                    # Reset flag to allow a test
                    llm_agent.primary.daily_quota_exhausted = False
                
                # Try a lightweight Gemini call
                try:
                    test_response = llm_agent.primary.get_response("Say OK", [])
                    if test_response and "I'm sorry, I'm having a moment" not in test_response:
                        logger.info("✅ Gemini recovered! Switching back from Ollama.")
                        llm_agent.using_fallback = False
                        llm_agent.consecutive_failures = 0
                    else:
                        logger.info("⏳ Gemini still failing — staying on Ollama")
                except Exception as e:
                    error_str = str(e).lower()
                    if 'limit: 0' in error_str or 'quota' in error_str:
                        if hasattr(llm_agent.primary, 'daily_quota_exhausted'):
                            llm_agent.primary.daily_quota_exhausted = True
                        logger.info("⏳ Gemini daily quota still exhausted — staying on Ollama")
                    else:
                        logger.info(f"⏳ Gemini error ({type(e).__name__}) — staying on Ollama")
        except Exception as e:
            logger.error(f"LLM health check error: {e}")
        
        await asyncio.sleep(check_interval)

app = FastAPI(title="FreeSWITCH VoiceBot - Multi-Call", lifespan=lifespan)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Session manager for multi-call coordination
session_manager = get_session_manager(
    redis_host=config.REDIS_HOST,
    redis_port=config.REDIS_PORT,
    redis_db=config.REDIS_DB,
    session_ttl=config.SESSION_TTL
)

# Track active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

logger.info("=" * 60)
logger.info("🚀 Initializing VoiceBot Components (Multi-Call)")
logger.info("=" * 60)

# Audio processing pipeline (shared across all calls)
if config.NC_ENABLED:
    noise_canceller = get_improved_noise_canceller(
        model_name=config.DF_MODEL,
        use_gpu=config.DF_USE_GPU,
        post_filter=config.DF_POST_FILTER,
        attenuation_limit=config.DF_ATTENUATION_LIMIT,
        normalization_gain=config.DF_GAIN,
        debug_rms=config.DF_DEBUG_RMS,
        debug_save_dir=config.DF_DEBUG_SAVE_DIR
    )
else:
    noise_canceller = None
    logger.info("⚠️  Noise Cancellation DISABLED (NC_ENABLED=false) — audio passes straight to STT")

vad_manager = PerCallVADManager(
    threshold=config.VAD_THRESHOLD,
    min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
    sample_rate=config.VAD_SAMPLE_RATE,
    window_size=config.VAD_WINDOW_SIZE
)

# Buffer manager (manages buffers for ALL calls)
buffer_manager = CallAudioManager(
    min_length=config.MIN_AUDIO_LENGTH_BYTES,
    max_length=config.MAX_AUDIO_LENGTH_BYTES,
    timeout_seconds=config.BUFFER_TIMEOUT_SECONDS
)

# Response handler (plays audio to caller via FreeSWITCH)
response_handler = ResponseHandler(
    audio_base_path=config.AUDIO_BASE_PATH,
    allow_interruptions=config.ALLOW_INTERRUPTIONS,
    speaking_timeout=config.BOT_SPEAKING_TIMEOUT
)

# STT handler (switchable: remote HTTP API or local Faster-Whisper)
stt_handler = create_stt_handler(
    provider=config.STT_PROVIDER,
    stt_url=config.STT_URL,
    stt_params=config.STT_PARAMS,
    stt_timeout=config.STT_TIMEOUT,
    model_size=config.STT_LOCAL_MODEL,
    device=config.STT_LOCAL_DEVICE,
    compute_type=config.STT_LOCAL_COMPUTE_TYPE,
    fallback_enabled=config.STT_FALLBACK_ENABLED,
    fallback_threshold=config.STT_FALLBACK_THRESHOLD,
)

# LLM agent (switchable: Gemini, Groq, or Ollama with fallback)
llm_agent = create_llm_agent(
    provider=config.LLM_PROVIDER,
    gemini_api_key=config.GEMINI_API_KEY,
    gemini_model=config.GEMINI_MODEL,
    ollama_url=config.OLLAMA_URL,
    ollama_model=config.OLLAMA_MODEL,
    groq_api_key=config.GROQ_API_KEY,
    groq_model=config.GROQ_MODEL,
    system_prompt_file=config.LLM_SYSTEM_PROMPT_FILE,
    fallback_enabled=config.LLM_FALLBACK_ENABLED,
    fallback_threshold=config.LLM_FALLBACK_THRESHOLD,
)

# TTS synthesizer (Edge-TTS with Redis caching)
try:
    redis_client = session_manager.redis_client
except Exception:
    redis_client = None

tts_synthesizer = TTSSynthesizer(
    voice=config.TTS_VOICE,
    cache_dir=config.TTS_CACHE_DIR,
    redis_client=redis_client,
)

logger.info("✓ All components initialized")
logger.info(f"✓ Worker ID: {config.WORKER_ID}")
logger.info(f"✓ Max concurrent calls: {config.MAX_CONCURRENT_CALLS}")
logger.info(f"✓ STT: {config.STT_PROVIDER} | LLM: {config.LLM_PROVIDER} | TTS: {config.TTS_VOICE}")
logger.info(f"✓ Barge-in: {'enabled' if config.ALLOW_INTERRUPTIONS else 'disabled'}")
if config.DF_DEBUG_SAVE_DIR:
    logger.info(f"📁 Debug audio files will be saved to: {config.DF_DEBUG_SAVE_DIR}")
logger.info("=" * 60)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_active_call_uuid(retries: int = 10, delay: float = 0.2) -> Optional[str]:
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
                    # Get most recent call not already handled
                    sorted_calls = sorted(
                        data["rows"],
                        key=lambda x: x.get('created_epoch', 0),
                        reverse=True
                    )
                    
                    # Find first call not in active_connections
                    for call in sorted_calls:
                        uuid = call["uuid"]
                        if uuid not in active_connections:
                            return uuid
                    
                    # If all calls are already handled, return most recent
                    return sorted_calls[0]["uuid"]
        except Exception as e:
            logger.debug(f"Attempt {i+1}/{retries} to get UUID failed: {e}")
        
        await asyncio.sleep(delay)
    
    return None


def _schedule_call_hangup(call_uuid: str, audio_file: str = None):
    """
    Schedule a call hangup after the final audio finishes playing + delay.
    Runs in a daemon thread.
    """
    try:
        # Wait for audio playback to finish
        wait_time = 0.0
        if audio_file:
            full_path = os.path.join(config.AUDIO_BASE_PATH, audio_file)
            wait_time = response_handler._get_audio_duration(full_path)
        
        total_wait = wait_time + config.CALL_END_DISCONNECT_DELAY
        logger.info(f"[{call_uuid}] 📲 Scheduling disconnect in {total_wait:.1f}s (audio={wait_time:.1f}s + delay={config.CALL_END_DISCONNECT_DELAY}s)")
        time.sleep(total_wait)
        
        # Hang up the call via FreeSWITCH
        subprocess.run(
            ["fs_cli", "-x", f"uuid_kill {call_uuid}"],
            capture_output=True,
            timeout=3
        )
        logger.info(f"[{call_uuid}] 📲 Call disconnected after flow end")
    except Exception as e:
        logger.error(f"[{call_uuid}] ❌ Failed to disconnect call: {e}")



def process_audio_segment(audio_data: bytes, call_uuid: str):
    """
    Process complete audio segment through the full AI pipeline.
    
    Pipeline: NC → STT → LLM (streaming) → TTS (per-sentence) → Playback
    
    This runs in a thread pool to avoid blocking the WebSocket loop.
    """
    # Barge-in check: if bot is speaking, either interrupt or ignore
    if response_handler.is_speaking(call_uuid):
        if config.ALLOW_INTERRUPTIONS:
            # BARGE-IN: Stop the bot and process the new speech
            logger.info(f"[{call_uuid}] 🛑 BARGE-IN: User interrupted bot")
            try:
                subprocess.run(
                    ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
                    capture_output=True, timeout=2
                )
            except Exception:
                pass
            response_handler.mark_stopped(call_uuid)
            emit_barge_in(call_uuid)
        else:
            logger.debug(f"[{call_uuid}] 🔇 Ignoring audio - bot is speaking")
            return
    
    # Clear any stale barge-in flag from previous interaction
    response_handler.clear_barge_in(call_uuid)
    
    tracker = LatencyTracker(call_uuid)
    
    try:
        # Step 1: Noise Cancellation (skipped if NC_ENABLED=false)
        tracker.start("nc")
        if noise_canceller is not None:
            enhanced_audio = noise_canceller.process_utterance(audio_data)
        else:
            enhanced_audio = audio_data  # passthrough
        tracker.stop("nc")
        
        # Step 2: STT Transcription (switchable provider)
        tracker.start("stt")
        text = stt_handler.transcribe(enhanced_audio)
        tracker.stop("stt")
        
        # If STT returned nothing, skip LLM — don't waste a call
        if not text:
            logger.info(f"[{call_uuid}] 🔇 STT returned empty — ignoring")
            return
        
        # Emit transcription event
        emit_transcription(call_uuid, text, tracker.get_component_ms("stt"), config.STT_PROVIDER)
        
        # Step 3: Get conversation history from session
        session_meta = session_manager.get_session(call_uuid) or {}
        history = session_meta.get("conversation_history", [])
        
        # Step 4: LLM Response (streaming) → concurrent TTS → Playback
        tracker.start("llm")
        
        full_response = ""
        first_sentence = True
        
        import queue
        sentence_queue = queue.Queue()
        producer_done = False
        
        # Producer thread to stream sentences from the LLM
        def llm_producer():
            nonlocal producer_done
            try:
                for sentence in llm_agent.get_response_streaming(text, history):
                    sentence_queue.put(sentence)
            except Exception as e:
                logger.error(f"[{call_uuid}] LLM producer error: {e}", exc_info=True)
            finally:
                producer_done = True
                sentence_queue.put(None)  # EOF Sentinel
        
        producer_thread = threading.Thread(target=llm_producer, daemon=True)
        producer_thread.start()
        
        from concurrent.futures import ThreadPoolExecutor as TtsPool
        pending_futures: list = []  # List of Future objects
        first_audio_played = False
        last_wav_path = None
        
        # 2 threads: one synthesizing, one ready
        try:
            with TtsPool(max_workers=2) as tts_pool:
                while not producer_done or pending_futures or not sentence_queue.empty():
                    # 1. Fetch new sentences from queue without blocking
                    try:
                        sentence = sentence_queue.get_nowait()
                        if sentence is not None:
                            if first_sentence:
                                tracker.stop("llm")
                                tracker.start("tts")
                                first_sentence = False
                            
                            full_response += sentence + " "
                            
                            # Strip out action tags before sending to TTS
                            import re as _re
                            clean_sentence = _re.sub(r'\[ACTION:[^\]]+\]', '', sentence).strip()
                            if clean_sentence:
                                # Submit TTS to thread pool immediately (non-blocking)
                                future = tts_pool.submit(tts_synthesizer.synthesize, clean_sentence)
                                pending_futures.append(future)
                    except queue.Empty:
                        pass
                    
                    # 2. Check if the next pending TTS is complete
                    if pending_futures and pending_futures[0].done():
                        future = pending_futures.pop(0)
                        try:
                            wav_path = future.result()
                            if wav_path:
                                tracker.mark_first_audio_sent()
                                response_handler.play_audio_queued(call_uuid, wav_path)
                                first_audio_played = True
                                last_wav_path = wav_path
                        except Exception as e:
                            logger.warning(f"[{call_uuid}] TTS future error: {e}")
                    
                    # 3. Check for barge-in (user actually spoke during playback)
                    if config.ALLOW_INTERRUPTIONS and response_handler.was_barge_in(call_uuid):
                        logger.info(f"[{call_uuid}] 🛑 Barge-in detected, stopping TTS pipeline")
                        response_handler.clear_barge_in(call_uuid)
                        # Cancel remaining futures
                        for f in pending_futures:
                            f.cancel()
                        pending_futures.clear()
                        break
                    
                    # 4. Sleep briefly to avoid CPU spinning
                    time.sleep(0.02)
        finally:
            pass
        
        if first_sentence:
            # LLM returned nothing or errored before yielding
            tracker.stop("llm")
            tracker.start("tts")
        
        tracker.stop("tts")
        
        full_response = full_response.strip()
        
        # Detect and handle action triggers from LLM response
        trigger_transfer = False
        trigger_hangup = False
        
        if "[ACTION:TRANSFER_AGENT]" in full_response:
            full_response = full_response.replace("[ACTION:TRANSFER_AGENT]", "").strip()
            trigger_transfer = True
            logger.info(f"[{call_uuid}] 🎯 Action detected: TRANSFER_AGENT")
            
        if "[ACTION:HANGUP]" in full_response:
            full_response = full_response.replace("[ACTION:HANGUP]", "").strip()
            trigger_hangup = True
            logger.info(f"[{call_uuid}] 🎯 Action detected: HANGUP")
        
        # Execute triggered actions
        audio_file = os.path.basename(last_wav_path) if last_wav_path else None
        
        if trigger_transfer or trigger_hangup:
            threading.Thread(
                target=_schedule_call_hangup,
                args=(call_uuid, audio_file),
                daemon=True
            ).start()
        
        # Emit bot response event (use actual provider, not config default)
        actual_provider = config.LLM_PROVIDER
        if hasattr(llm_agent, 'using_fallback') and llm_agent.using_fallback:
            actual_provider = llm_agent.fallback.name if hasattr(llm_agent, 'fallback') else "ollama"
        emit_bot_response(call_uuid, full_response, tracker.get_component_ms("llm"), actual_provider)
        
        # Step 6: Update conversation history in session
        history.append({"role": "user", "text": text})
        history.append({"role": "assistant", "text": full_response})
        
        # Trim history to max length
        if len(history) > config.LLM_MAX_HISTORY * 2:
            history = history[-(config.LLM_MAX_HISTORY * 2):]
        
        session_meta["conversation_history"] = history
        session_meta["last_transcription"] = text
        session_meta["last_response"] = full_response
        session_manager.update_session(call_uuid, session_meta)
        
        # Step 7: Log full latency breakdown + RTT
        if config.ENABLE_TIMING_LOGS:
            tracker.log_summary(providers={
                "stt": config.STT_PROVIDER,
                "llm": actual_provider,
            })
        
        # Emit turn metrics event
        emit_turn_metrics(call_uuid, tracker.summary())
        
    except Exception as e:
        logger.error(f"[{call_uuid}] ❌ Error processing audio segment: {e}", exc_info=True)


# =============================================================================
# WEBSOCKET ENDPOINT (Multi-Call Capable)
# =============================================================================

@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for audio streaming
    NOW HANDLES MULTIPLE CONCURRENT CALLS
    """
    await websocket.accept()
    
    connection_start = time.time()
    call_uuid = None
    acquired_lock = False  # Track lock state for safe cleanup
    
    logger.info("=" * 60)
    logger.info("📞 NEW CALL STARTING")
    
    try:
        # Check concurrent call limit
        current_count = len(active_connections)
        if current_count >= config.MAX_CONCURRENT_CALLS:
            logger.error(f"⚠️  Max concurrent calls reached ({current_count}/{config.MAX_CONCURRENT_CALLS})")
            await websocket.close(code=1008, reason="Server at capacity")
            return
        
        # Get call UUID
        call_uuid = await get_active_call_uuid()
        
        if not call_uuid:
            logger.error("⚠️  Could not find active call UUID")
            await websocket.close()
            return
        
        # Check if session already exists (reconnection scenario)
        existing_session = session_manager.get_session(call_uuid)
        if existing_session:
            logger.info(f"Reconnecting to existing session {call_uuid}")
        else:
            # Create new session
            session_manager.create_session(call_uuid, metadata={
                'worker_id': config.WORKER_ID,
                'connection_time': connection_start
            })
        
        # Acquire session lock
        acquired_lock = session_manager.acquire_session_lock(
            call_uuid, config.WORKER_ID, timeout=600  # 10 min max call
        )
        if not acquired_lock:
            logger.error(f"⚠️  Could not acquire lock for {call_uuid}")
            await websocket.close()
            return
        
        # Register connection
        active_connections[call_uuid] = websocket
        
        connection_time = (time.time() - connection_start) * 1000
        logger.info(f"✓ Connected to call {call_uuid} ({connection_time:.0f}ms)")
        logger.info(f"📊 Active calls: {len(active_connections)}/{config.MAX_CONCURRENT_CALLS}")
        if config.NC_ENABLED:
            logger.info(f"⚙️  NC: {config.DF_MODEL} (atten={config.DF_ATTENUATION_LIMIT}dB, gain={config.DF_GAIN}x)")
        else:
            logger.info("⚙️  NC: DISABLED")
        logger.info(f"⚙️  VAD: threshold={config.VAD_THRESHOLD}, silence={config.VAD_MIN_SILENCE_DURATION_MS}ms")
        
        # Initialize VAD state for this specific call
        vad_detector = vad_manager.get_vad(call_uuid)
        
        # Get buffer for this call
        audio_buffer = buffer_manager.get_buffer(call_uuid)

        # Initialize conversation session
        session_meta = session_manager.get_session(call_uuid) or {}
        session_meta['conversation_history'] = []
        session_manager.update_session(call_uuid, session_meta)
        
        # Generate and play welcome message via TTS
        welcome_text = "Hello! Welcome to Unified Reach Fiber. How can I help you today?"
        
        # Stop any existing audio
        subprocess.run(
            ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
            capture_output=True
        )
        await asyncio.sleep(0.5)
        
        # Synthesize welcome and play (use await since we're in an async handler)
        welcome_wav = await tts_synthesizer.synthesize_awaitable(welcome_text)
        if welcome_wav:
            response_handler.play_audio(call_uuid, welcome_wav, welcome_text)
        
        # Main audio processing loop
        chunk_count = 0
        vad_speech_count = 0
        vad_silence_count = 0
        last_activity_time = time.time()
        
        while True:
            try:
                # Add timeout to prevent hanging (120s — generous to avoid false disconnects)
                message = await asyncio.wait_for(websocket.receive(), timeout=120.0)
                last_activity_time = time.time()
                
            except asyncio.TimeoutError:
                # 120s with no audio data is very likely a dead call
                elapsed_idle = time.time() - last_activity_time
                logger.warning(f"[{call_uuid}] ⚠️  No audio data for {elapsed_idle:.0f}s — checking if call is still alive")
                
                # Check if call is still active via FreeSWITCH
                try:
                    result = subprocess.run(
                        ["fs_cli", "-x", f"uuid_exists {call_uuid}"],
                        capture_output=True, text=True, timeout=3
                    )
                    call_alive = "true" in result.stdout.strip().lower()
                except Exception:
                    call_alive = True  # Assume alive if we can't check (fs_cli timeout)
                
                if not call_alive:
                    logger.info(f"[{call_uuid}] ⚠️  Call no longer exists in FreeSWITCH — closing WebSocket")
                    break
                logger.debug(f"[{call_uuid}] WebSocket timeout but call still active — continuing")
                continue
                
            # Handle disconnection
            if message["type"] == "websocket.disconnect":
                logger.info(f"[{call_uuid}] 🚫 Call ended by client")
                break
            
            # Process audio data
            if "bytes" in message:
                chunk_count += 1
                raw_chunk = message["bytes"]
                
                # Log activity periodically and refresh lock TTL
                if chunk_count % 100 == 0:
                    logger.debug(f"[{call_uuid}] 📊 Received {chunk_count} chunks")
                    # Refresh lock every ~3s (100 chunks × 32ms) to prevent expiry on long calls
                    session_manager.acquire_session_lock(call_uuid, config.WORKER_ID, timeout=600)
                
                # PIPELINE STEP 1: Noise Cancellation (SKIP per-chunk NC)
                # We skip NC here and do it on the full utterance before STT
                enhanced_chunk = raw_chunk

                
                # PIPELINE STEP 2: VAD Detection
                try:
                    vad_result = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        vad_detector.process_stream,
                        enhanced_chunk
                    )
                except Exception as e:
                    logger.error(f"[{call_uuid}] VAD processing error: {e}")
                    vad_result = {'is_speech': False, 'speech_start': False, 
                                 'speech_end': False, 'probability': 0.0}
                
                # Track VAD results
                if vad_result['is_speech']:
                    vad_speech_count += 1
                else:
                    vad_silence_count += 1
                
                # Log VAD events + emit WebSocket events
                if vad_result.get('speech_start'):
                    logger.info(f"[{call_uuid}] 🎤 SPEECH START (prob: {vad_result['probability']:.2f})")
                    emit_speech_started(call_uuid)
                    
                    # Interruption logic (barge-in): stop playback instantly when user starts speaking
                    if response_handler.is_speaking(call_uuid) and config.ALLOW_INTERRUPTIONS:
                        logger.info(f"[{call_uuid}] 🛑 BARGE-IN: User started speaking, stopping bot playback immediately")
                        try:
                            subprocess.run(
                                ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
                                capture_output=True, timeout=2
                            )
                        except Exception as e:
                            logger.error(f"Error running uuid_break: {e}")
                        response_handler.mark_stopped(call_uuid)
                        emit_barge_in(call_uuid)
                if vad_result.get('speech_end'):
                    duration_ms = vad_speech_count * config.CHUNK_DURATION_MS
                    logger.info(f"[{call_uuid}] 🎤 SPEECH END (speech={vad_speech_count}, silence={vad_silence_count})")
                    emit_speech_ended(call_uuid, duration_ms)
                    vad_speech_count = 0
                    vad_silence_count = 0
                
                # PIPELINE STEP 3: Buffer Management (per-call buffers)
                ready_audio = audio_buffer.add_chunk(enhanced_chunk, vad_result)
                
                # PIPELINE STEP 4: Process complete speech segment
                if ready_audio:
                    logger.info(
                        f"[{call_uuid}] 🎤 Speech segment complete "
                        f"({len(ready_audio)} bytes, {chunk_count} chunks)"
                    )
                    
                    # Process in background thread
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_audio_segment,
                        ready_audio,
                        call_uuid
                    )
    
    except WebSocketDisconnect:
        logger.info(f"[{call_uuid}] 🚫 WebSocket disconnected")
    except Exception as e:
        logger.error(f"[{call_uuid}] ❌ WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        if call_uuid:
            # Release session lock only if we acquired it
            if acquired_lock:
                session_manager.release_session_lock(call_uuid, config.WORKER_ID)
            
            # End session
            session_manager.end_session(call_uuid)
            
            # Remove from active connections
            if call_uuid in active_connections:
                del active_connections[call_uuid]
            
            # Cleanup buffers and handlers
            buffer_manager.remove_buffer(call_uuid)
            vad_manager.remove_vad(call_uuid)
            response_handler.cleanup_call(call_uuid)
            
            logger.info("=" * 60)
            logger.info(f"📊 CALL {call_uuid} STATISTICS")
            logger.info(f"   Active calls remaining: {len(active_connections)}")
            logger.info(f"   STT [{config.STT_PROVIDER}]: {stt_handler.get_stats()}")
            logger.info(f"   LLM [{config.LLM_PROVIDER}]: {llm_agent.get_stats()}")
            logger.info(f"   TTS: {tts_synthesizer.get_stats()}")
            logger.info(f"   Response: {response_handler.get_stats()}")
            if config.DF_DEBUG_SAVE_DIR:
                logger.info(f"   Debug audio saved to: {config.DF_DEBUG_SAVE_DIR}")
            logger.info("=" * 60)


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def cleanup_stale_sessions():
    """Background task to clean up stale sessions"""
    while True:
        try:
            await asyncio.sleep(config.SESSION_CLEANUP_INTERVAL)
            cleaned = session_manager.cleanup_stale_sessions()
            if cleaned > 0:
                logger.info(f"🧹 Cleaned up {cleaned} stale sessions")
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")


# startup_event removed — handled by lifespan context manager above


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint with multi-call stats"""
    redis_ok = False
    try:
        redis_ok = session_manager.redis_client.ping()
    except Exception:
        pass
    
    return {
        "status": "healthy",
        "worker_id": config.WORKER_ID,
        "components": {
            "noise_canceller": "loaded" if noise_canceller else "disabled",
            "vad_detector": "loaded",
            "stt_handler": f"ready ({config.STT_PROVIDER})",
            "llm_agent": f"ready ({config.LLM_PROVIDER})",
            "tts_synthesizer": f"ready ({config.TTS_VOICE})",
            "response_handler": "ready",
            "session_manager": "ready",
            "redis": redis_ok,
        },
        "providers": {
            "stt": config.STT_PROVIDER,
            "llm": config.LLM_PROVIDER,
            "tts_voice": config.TTS_VOICE,
            "barge_in": config.ALLOW_INTERRUPTIONS,
        },
        "capacity": {
            "active_calls": len(active_connections),
            "max_concurrent": config.MAX_CONCURRENT_CALLS,
            "utilization": f"{len(active_connections)/config.MAX_CONCURRENT_CALLS*100:.1f}%"
        },
        "sessions": session_manager.get_stats(),
        "stt_stats": stt_handler.get_stats(),
        "llm_stats": llm_agent.get_stats(),
        "tts_stats": tts_synthesizer.get_stats(),
        "debug_enabled": config.DF_DEBUG_SAVE_DIR is not None
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    return {
        "worker": {
            "id": config.WORKER_ID,
            "active_connections": len(active_connections),
            "max_concurrent": config.MAX_CONCURRENT_CALLS
        },
        "providers": {
            "stt": config.STT_PROVIDER,
            "llm": config.LLM_PROVIDER,
            "tts_voice": config.TTS_VOICE,
        },
        "sessions": session_manager.get_stats(),
        "stt": stt_handler.get_stats(),
        "llm": llm_agent.get_stats(),
        "tts": tts_synthesizer.get_stats(),
        "response_handler": response_handler.get_stats(),
        "buffer_manager": buffer_manager.get_all_stats(),
        "config": {
            "noise_cancellation": config.DF_MODEL if config.NC_ENABLED else "disabled",
            "nc_attenuation": config.DF_ATTENUATION_LIMIT,
            "nc_gain": config.DF_GAIN,
            "vad_threshold": config.VAD_THRESHOLD,
            "vad_silence_ms": config.VAD_MIN_SILENCE_DURATION_MS,
            "allow_interruptions": config.ALLOW_INTERRUPTIONS,
            "debug_enabled": config.DF_DEBUG_SAVE_DIR is not None
        }
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    active_uuids = session_manager.get_active_sessions()
    sessions = []
    
    for uuid in active_uuids:
        session = session_manager.get_session(uuid)
        if session:
            sessions.append(session)
    
    return {
        "active_count": len(sessions),
        "sessions": sessions
    }


# =============================================================================
# VOICE TESTING ENDPOINTS — change TTS voice without restart
# =============================================================================

@app.get("/voices")
async def list_voices():
    """
    List popular Edge-TTS voices for testing.
    Full list: run `edge-tts --list-voices` inside the container.
    """
    return {
        "current_voice": tts_synthesizer.voice,
        "popular_voices": {
            "en-US": [
                "en-US-AvaMultilingualNeural",     # Female, natural, multilingual
                "en-US-AndrewMultilingualNeural",  # Male, natural, multilingual
                "en-US-JennyNeural",               # Female, customer service
                "en-US-GuyNeural",                 # Male, customer service
                "en-US-AriaNeural",                # Female, expressive
                "en-US-DavisNeural",               # Male, casual
            ],
            "en-GB": [
                "en-GB-SoniaNeural",               # Female, British
                "en-GB-RyanNeural",                # Male, British
                "en-GB-LibbyNeural",               # Female, British
            ],
            "en-KE": [
                "en-KE-AsiliaNeural",              # Female, Kenyan English
                "en-KE-ChilembaNeural",            # Male, Kenyan English
            ],
        },
        "tip": "POST /voice with {\"voice\": \"<name>\", \"preview\": true} to test a voice"
    }


@app.post("/voice")
async def set_voice(request: dict):
    """
    Live-swap the TTS voice without restarting the server.

    Body: { "voice": "en-US-JennyNeural", "preview": true }
    - voice: Edge-TTS voice name (required)
    - preview: if true, synthesizes a sample sentence and returns the cache path (optional)

    Does NOT affect calls currently in progress — takes effect on next call.
    """
    voice = request.get("voice", "").strip()
    if not voice:
        return {"error": "voice field is required"}

    old_voice = tts_synthesizer.voice
    tts_synthesizer.voice = voice
    logger.info(f"🔊 TTS voice changed: {old_voice} → {voice}")

    result = {
        "status": "ok",
        "previous_voice": old_voice,
        "current_voice": voice,
        "note": "Voice change takes effect on next call. Current calls are unaffected.",
    }

    # Optional: synthesize a preview sentence to hear the voice immediately
    if request.get("preview", False):
        preview_text = request.get(
            "preview_text",
            "Hello! Thank you for calling Unified Reach Fiber. How can I help you today?"
        )
        try:
            wav_path = await tts_synthesizer.synthesize_awaitable(preview_text)
            result["preview_wav"] = wav_path
            result["preview_text"] = preview_text
        except Exception as e:
            result["preview_error"] = str(e)

    return result


# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting FreeSWITCH VoiceBot Server (AI-Powered, Multi-Call)")
    logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
    logger.info(f"   Worker ID: {config.WORKER_ID}")
    logger.info(f"   Max Concurrent: {config.MAX_CONCURRENT_CALLS}")
    logger.info(f"   Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
    logger.info(f"   STT: {config.STT_PROVIDER} ({'→ ' + config.STT_URL if config.STT_PROVIDER == 'remote' else config.STT_LOCAL_MODEL})")
    logger.info(f"   LLM: {config.LLM_PROVIDER} ({config.GEMINI_MODEL if config.LLM_PROVIDER == 'gemini' else config.OLLAMA_MODEL})")
    logger.info(f"   TTS: Edge-TTS ({config.TTS_VOICE})")
    logger.info(f"   Barge-in: {'enabled' if config.ALLOW_INTERRUPTIONS else 'disabled'}")
    logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
    if config.NC_ENABLED:
        logger.info(f"   NC Model: {config.DF_MODEL} (atten={config.DF_ATTENUATION_LIMIT}dB, gain={config.DF_GAIN}x)")
    else:
        logger.info(f"   NC: DISABLED")
    logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
    if config.DF_DEBUG_SAVE_DIR:
        logger.info(f"   Debug Audio: {config.DF_DEBUG_SAVE_DIR}")
    logger.info("=" * 60)
    
    # Suppress noisy /health access logs (Docker pings every ~10s)
    class HealthCheckFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "/health" not in msg
    
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
    
    uvicorn.run(
        app,
        host=config.WS_HOST,
        port=config.WS_PORT,
        log_level=config.LOG_LEVEL.lower()
    )

