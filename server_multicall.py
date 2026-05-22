"""
FreeSWITCH VoiceBot - Multi-Call WebSocket Server
Handles multiple concurrent calls with Redis-based session management
"""

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
from ivr import ResponseHandler, FlowEngine
from stt_handler import STTHandler
from session_manager import get_session_manager

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
    yield
    # Shutdown (nothing to do currently)

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

# IVR components
flow_engine = FlowEngine()

response_handler = ResponseHandler(
    audio_base_path=config.AUDIO_BASE_PATH,
    allow_interruptions=config.ALLOW_INTERRUPTIONS,
    speaking_timeout=config.BOT_SPEAKING_TIMEOUT
)

# STT handler (shared, thread-safe)
stt_handler = STTHandler(
    stt_url=config.STT_URL,
    stt_params=config.STT_PARAMS,
    timeout=config.STT_TIMEOUT
)

logger.info("✓ All components initialized")
logger.info(f"✓ Worker ID: {config.WORKER_ID}")
logger.info(f"✓ Max concurrent calls: {config.MAX_CONCURRENT_CALLS}")
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
    Process complete audio segment through full pipeline
    This runs in a thread pool to avoid blocking
    """
    # Check if bot is speaking (and interruptions not allowed)
    if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
        logger.debug(f"[{call_uuid}] 🔇 Ignoring audio - bot is speaking")
        return
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Noise Cancellation (skipped if NC_ENABLED=false)
        nc_start = time.time()
        if noise_canceller is not None:
            enhanced_audio = noise_canceller.process_utterance(audio_data)
        else:
            enhanced_audio = audio_data  # passthrough
        nc_time = (time.time() - nc_start) * 1000
        
        # Step 2: STT Transcription
        stt_start = time.time()
        text = stt_handler.transcribe(enhanced_audio)
        stt_time = (time.time() - stt_start) * 1000
        
        # If STT returned nothing, treat as unrecognized input
        # This prevents the bot from hanging when input was too quiet/noisy
        if not text:
            logger.info(f"[{call_uuid}] 🔇 STT returned empty — treating as unrecognized speech")
            text = ""  # Pass empty text so flow engine triggers retry/sorry prompt
        
        # Step 3: IVR Flow Process
        intent_start = time.time()
        
        # Fetch session state to pass to the engine
        session_meta = session_manager.get_session(call_uuid) or {}
        flow_state = session_meta.get("flow_state", {})
        
        answer_text, audio_file, should_end, is_fallback = flow_engine.process_input(flow_state, text)
        
        intent_time = (time.time() - intent_start) * 1000
        
        # Step 4: Play Response
        response_start = time.time()
        if audio_file:
            response_handler.play_audio(call_uuid, audio_file, answer_text)
        response_time = (time.time() - response_start) * 1000
        
        # Update session activity
        session_meta['flow_state'] = flow_state
        session_meta['last_transcription'] = text
        session_meta['last_audio'] = audio_file
        session_manager.update_session(call_uuid, session_meta)
        
        if should_end:
            logger.info(f"[{call_uuid}] Call flow ended.")
            # Schedule call disconnect after audio finishes + delay
            threading.Thread(
                target=_schedule_call_hangup,
                args=(call_uuid, audio_file),
                daemon=True
            ).start()
        
        # Log performance
        total_time = (time.time() - pipeline_start) * 1000
        if config.ENABLE_TIMING_LOGS:
            logger.info(
                f"[{call_uuid}] ⏱️  Pipeline: NC={nc_time:.0f}ms, "
                f"STT={stt_time:.0f}ms, "
                f"Intent={intent_time:.0f}ms, "
                f"Response={response_time:.0f}ms, "
                f"Total={total_time:.0f}ms"
            )
        
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

        # Set up IVR Flow state
        session_meta = session_manager.get_session(call_uuid) or {}
        session_meta['flow_state'] = {"lang": "en"}
        session_manager.update_session(call_uuid, session_meta)
        ans_text, ans_audio = flow_engine.get_initial_step()
        
        # Stop any existing audio and play welcome
        subprocess.run(
            ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
            capture_output=True
        )
        await asyncio.sleep(0.5)
        if ans_audio:
            response_handler.play_audio(call_uuid, ans_audio, ans_text)
        
        # Main audio processing loop
        chunk_count = 0
        vad_speech_count = 0
        vad_silence_count = 0
        last_activity_time = time.time()
        
        while True:
            try:
                # Add timeout to prevent hanging
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                last_activity_time = time.time()
                
            except asyncio.TimeoutError:
                # Check if call is still active
                current_uuid = await get_active_call_uuid(retries=1)
                if current_uuid != call_uuid:
                    logger.info(f"[{call_uuid}] ⚠️  Call ended, closing WebSocket")
                    break
                logger.debug(f"[{call_uuid}] WebSocket timeout but call still active")
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
                
                # Log VAD events
                if vad_result.get('speech_start'):
                    logger.info(f"[{call_uuid}] 🎤 SPEECH START (prob: {vad_result['probability']:.2f})")
                if vad_result.get('speech_end'):
                    logger.info(f"[{call_uuid}] 🎤 SPEECH END (speech={vad_speech_count}, silence={vad_silence_count})")
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
            logger.info(f"   STT: {stt_handler.get_stats()}")
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
    return {
        "status": "healthy",
        "worker_id": config.WORKER_ID,
        "components": {
            "noise_canceller": "loaded",
            "vad_detector": "loaded",
            "stt_handler": "ready",
            "flow_engine": f"{len(flow_engine.flows)} flows loaded",
            "response_handler": "ready",
            "session_manager": "ready",
            "redis": session_manager.redis_client.ping()
        },
        "capacity": {
            "active_calls": len(active_connections),
            "max_concurrent": config.MAX_CONCURRENT_CALLS,
            "utilization": f"{len(active_connections)/config.MAX_CONCURRENT_CALLS*100:.1f}%"
        },
        "sessions": session_manager.get_stats(),
        "stt_stats": stt_handler.get_stats(),
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
        "sessions": session_manager.get_stats(),
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting FreeSWITCH VoiceBot Server (Multi-Call Mode)")
    logger.info(f"   Server: {config.WS_HOST}:{config.WS_PORT}")
    logger.info(f"   Worker ID: {config.WORKER_ID}")
    logger.info(f"   Max Concurrent: {config.MAX_CONCURRENT_CALLS}")
    logger.info(f"   Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
    logger.info(f"   STT: {config.STT_URL}")
    logger.info(f"   Audio: {config.AUDIO_BASE_PATH}")
    logger.info(f"   NC Model: {config.DF_MODEL} (atten={config.DF_ATTENUATION_LIMIT}dB, gain={config.DF_GAIN}x)")
    logger.info(f"   VAD Threshold: {config.VAD_THRESHOLD}")
    if config.DF_DEBUG_SAVE_DIR:
        logger.info(f"   Debug Audio: {config.DF_DEBUG_SAVE_DIR}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=config.WS_HOST,
        port=config.WS_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
