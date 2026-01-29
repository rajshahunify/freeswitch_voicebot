"""
SIMPLIFIED SERVER FOR TESTING
This version uses simple RMS-based detection instead of VAD
Use this to verify audio flow is working
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import logging
import subprocess
import json
import time
import audioop
from concurrent.futures import ThreadPoolExecutor

import config
from ivr import IntentMatcher, ResponseHandler
from stt_handler import STTHandler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# Initialize components
intent_matcher = IntentMatcher(
    intent_keywords=config.INTENT_KEYWORDS,
    fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
)

response_handler = ResponseHandler(
    audio_base_path=config.AUDIO_BASE_PATH,
    allow_interruptions=config.ALLOW_INTERRUPTIONS
)

stt_handler = STTHandler(
    stt_url=config.STT_URL,
    stt_params=config.STT_PARAMS,
    timeout=config.STT_TIMEOUT
)

logger.info("✓ Components initialized (SIMPLE MODE)")

# Simple RMS-based detection parameters
SILENCE_THRESHOLD = 300  # RMS threshold
SILENCE_CHUNKS = 15      # Number of silent chunks before end
MIN_AUDIO_LENGTH = 32000 # Minimum audio bytes


async def get_active_call_uuid(retries=10, delay=0.2):
    """Get active call UUID"""
    for i in range(retries):
        try:
            cmd = ["fs_cli", "-x", "show channels as json"]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            output = process.stdout.strip()
            
            if output:
                data = json.loads(output)
                if data and "rows" in data and len(data["rows"]) > 0:
                    sorted_calls = sorted(
                        data["rows"],
                        key=lambda x: x.get('created_epoch', 0),
                        reverse=True
                    )
                    return sorted_calls[0]["uuid"]
        except Exception:
            pass
        await asyncio.sleep(delay)
    return None


def process_audio_segment(audio_data, call_uuid):
    """Process audio segment"""
    if not config.ALLOW_INTERRUPTIONS and response_handler.is_speaking(call_uuid):
        logger.debug("Bot is speaking, ignoring user audio")
        return
    
    try:
        logger.info(f"🎯 Processing {len(audio_data)} bytes of audio")
        
        # STT
        text = stt_handler.transcribe(audio_data)
        
        if text:
            # Intent matching
            audio_file = intent_matcher.match_intent(text)
            
            # Play response
            response_handler.play_audio(call_uuid, audio_file, text)
        else:
            logger.warning("STT returned no text")
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)


@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    logger.info("=" * 60)
    logger.info("📞 NEW CALL (SIMPLE MODE)")
    logger.info("=" * 60)
    
    call_uuid = await get_active_call_uuid()
    
    if call_uuid:
        logger.info(f"✓ Connected to call {call_uuid}")
        
        # Play welcome
        subprocess.run(["fs_cli", "-x", f"uuid_break {call_uuid} all"], capture_output=True)
        await asyncio.sleep(0.5)
        response_handler.play_audio(call_uuid, "english_menu.wav")
    else:
        logger.error("No call UUID found")
        await websocket.close()
        return
    
    audio_buffer = bytearray()
    silence_counter = 0
    chunk_count = 0
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                logger.info("Call ended")
                break
            
            if "bytes" in message:
                chunk_count += 1
                raw_chunk = message["bytes"]
                
                # Log progress
                if chunk_count % 50 == 0:
                    logger.info(f"📊 Received {chunk_count} chunks")
                
                # Add to buffer
                audio_buffer.extend(raw_chunk)
                
                # Calculate RMS
                try:
                    rms = audioop.rms(raw_chunk, 2)
                except Exception as e:
                    logger.error(f"RMS error: {e}")
                    rms = 0
                
                # Log RMS values periodically
                if chunk_count % 50 == 0:
                    logger.info(f"🔊 Current RMS: {rms} (threshold: {SILENCE_THRESHOLD})")
                
                # Simple silence detection
                if rms < SILENCE_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0
                    if chunk_count % 20 == 0:
                        logger.debug(f"🗣️ Speech detected (RMS: {rms})")
                
                # Process on silence
                if silence_counter > SILENCE_CHUNKS and len(audio_buffer) > MIN_AUDIO_LENGTH:
                    logger.info(f"🎤 Speech segment detected ({len(audio_buffer)} bytes, {chunk_count} chunks)")
                    
                    chunk_copy = bytes(audio_buffer)
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_audio_segment,
                        chunk_copy,
                        call_uuid
                    )
                    
                    audio_buffer = bytearray()
                    silence_counter = 0
                    chunk_count = 0
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        response_handler.cleanup_call(call_uuid)
        logger.info("Call cleanup complete")


@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "simple"}


if __name__ == "__main__":
    logger.info("🚀 Starting SIMPLE Test Server")
    logger.info("   Using RMS-based detection (no VAD/NC)")
    uvicorn.run(app, host="0.0.0.0", port=8000)