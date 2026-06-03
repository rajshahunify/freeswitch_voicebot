"""
Event Emitter — WebSocket event protocol for the voicebot pipeline

Emits structured JSON events for:
- Speech lifecycle (speech_started, speech_ended, barge_in)
- Pipeline stages (transcription, bot_response, tts_started)
- Metrics (turn_metrics)
- DTMF (keypad presses)

Events are:
1. Logged to the application logger (always)
2. Broadcast to connected WebSocket clients (when dashboard is connected, Phase 2)

Event format matches SiphonAI spec for future compatibility.
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Connected event listeners (WebSocket clients, added in Phase 2)
_listeners: List[Any] = []


def register_listener(listener):
    """Register a WebSocket client to receive events"""
    _listeners.append(listener)
    logger.info(f"Event listener registered. Total listeners: {len(_listeners)}")


def unregister_listener(listener):
    """Remove a WebSocket client from event listeners"""
    if listener in _listeners:
        _listeners.remove(listener)
    logger.info(f"Event listener removed. Total listeners: {len(_listeners)}")


def emit_event(call_uuid: str, event_type: str, data: Optional[Dict] = None):
    """
    Emit a structured event.
    
    Args:
        call_uuid: The call this event belongs to
        event_type: One of: speech_started, speech_ended, transcription,
                    bot_response, tts_started, barge_in, dtmf, turn_metrics
        data: Additional event data (varies by event type)
    
    Event schema:
        {
            "event": str,
            "call_uuid": str,
            "timestamp": float,
            ...data
        }
    """
    event = {
        "event": event_type,
        "call_uuid": call_uuid,
        "timestamp": time.time(),
    }
    
    if data:
        event.update(data)
    
    # Always log the event
    _log_event(event)
    
    # Broadcast to connected listeners (Phase 2: web dashboard)
    _broadcast(event)


def emit_speech_started(call_uuid: str):
    """Emit when VAD detects user speech start"""
    emit_event(call_uuid, "speech_started")


def emit_speech_ended(call_uuid: str, duration_ms: float):
    """Emit when VAD detects silence after speech"""
    emit_event(call_uuid, "speech_ended", {"duration_ms": round(duration_ms, 1)})


def emit_transcription(call_uuid: str, text: str, stt_ms: float, provider: str):
    """Emit when STT produces a transcription"""
    emit_event(call_uuid, "transcription", {
        "text": text,
        "stt_ms": round(stt_ms, 1),
        "provider": provider,
    })


def emit_bot_response(call_uuid: str, text: str, llm_ms: float, provider: str):
    """Emit when LLM produces a response"""
    emit_event(call_uuid, "bot_response", {
        "text": text,
        "llm_ms": round(llm_ms, 1),
        "provider": provider,
    })


def emit_tts_started(call_uuid: str, text: str, tts_ms: float):
    """Emit when TTS begins playback"""
    emit_event(call_uuid, "tts_started", {
        "text": text[:100],  # Truncate for log readability
        "tts_ms": round(tts_ms, 1),
    })


def emit_barge_in(call_uuid: str):
    """Emit when user interrupts bot speech"""
    emit_event(call_uuid, "barge_in")


def emit_dtmf(call_uuid: str, digit: str):
    """Emit when a DTMF keypress is detected"""
    emit_event(call_uuid, "dtmf", {"digit": digit})


def emit_turn_metrics(call_uuid: str, metrics: Dict[str, float]):
    """
    Emit per-turn pipeline metrics.
    
    Args:
        metrics: Dict from LatencyTracker.summary(), e.g.
                 {"nc_ms": 12, "stt_ms": 450, "llm_ms": 280, "tts_ms": 150, 
                  "total_ms": 892, "rtt_ms": 1050, "overhead_ms": 158}
    """
    emit_event(call_uuid, "turn_metrics", metrics)


def _log_event(event: Dict):
    """Log event in a compact, readable format"""
    event_type = event.get("event", "unknown")
    call_uuid = event.get("call_uuid", "?")
    
    # Compact log format per event type
    if event_type == "transcription":
        logger.info(f"[{call_uuid}] 📝 Event: transcription — \"{event.get('text', '')}\" "
                     f"({event.get('stt_ms', 0):.0f}ms, {event.get('provider', '?')})")
    elif event_type == "bot_response":
        text = event.get('text', '')
        logger.info(f"[{call_uuid}] 🤖 Event: bot_response — \"{text[:80]}{'...' if len(text) > 80 else ''}\" "
                     f"({event.get('llm_ms', 0):.0f}ms, {event.get('provider', '?')})")
    elif event_type == "barge_in":
        logger.info(f"[{call_uuid}] 🛑 Event: barge_in")
    elif event_type == "turn_metrics":
        logger.debug(f"[{call_uuid}] 📊 Event: turn_metrics — {json.dumps(event, default=str)}")
    else:
        logger.debug(f"[{call_uuid}] 📡 Event: {event_type}")


def _broadcast(event: Dict):
    """Broadcast event to all connected listeners (Phase 2)"""
    if not _listeners:
        return
    
    event_json = json.dumps(event, default=str)
    disconnected = []
    
    for listener in _listeners:
        try:
            # In Phase 2, listener will be an async WebSocket — we'll need to
            # run this in an event loop. For now, just log.
            pass
        except Exception:
            disconnected.append(listener)
    
    # Cleanup disconnected listeners
    for listener in disconnected:
        _listeners.remove(listener)
