"""
Latency Tracker — Per-request timing for the voicebot pipeline

Tracks:
- Per-component timing (NC, STT, LLM, TTS)
- Total pipeline time
- Round-trip time (RTT): wall-clock from audio arrival to first TTS byte sent
- Hidden overhead (RTT - sum of component times)
"""

import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LatencyTracker:
    """
    Tracks latency for a single request through the pipeline.
    
    Usage:
        tracker = LatencyTracker(call_uuid)
        
        tracker.start("nc")
        # ... noise cancellation ...
        tracker.stop("nc")
        
        tracker.start("stt")
        # ... transcription ...
        tracker.stop("stt")
        
        summary = tracker.summary()
        # {'nc_ms': 12.3, 'stt_ms': 450.1, ..., 'total_ms': 920.5, 'rtt_ms': 1050.2, 'overhead_ms': 129.7}
    """
    
    def __init__(self, call_uuid: str):
        self.call_uuid = call_uuid
        self.pipeline_start = time.time()
        self._starts: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}
        self._first_audio_sent: Optional[float] = None
    
    def start(self, component: str):
        """Mark the start of a component"""
        self._starts[component] = time.time()
    
    def stop(self, component: str):
        """Mark the end of a component and record duration"""
        if component in self._starts:
            self._durations[component] = (time.time() - self._starts[component]) * 1000
            del self._starts[component]
        else:
            logger.warning(f"[{self.call_uuid}] LatencyTracker: stop('{component}') called without start()")
    
    def mark_first_audio_sent(self):
        """Mark when the first TTS audio byte is sent to FreeSWITCH"""
        self._first_audio_sent = time.time()
    
    def get_component_ms(self, component: str) -> float:
        """Get duration of a specific component in milliseconds"""
        return self._durations.get(component, 0.0)
    
    def summary(self) -> Dict[str, float]:
        """
        Return complete latency breakdown.
        
        Returns dict with:
        - {component}_ms: time for each tracked component
        - total_ms: sum of all component times
        - rtt_ms: wall-clock time from pipeline start to first audio sent
        - overhead_ms: rtt_ms - total_ms (thread scheduling, queue delays, etc.)
        """
        result = {}
        
        # Per-component times
        component_total = 0.0
        for component, duration_ms in self._durations.items():
            result[f"{component}_ms"] = round(duration_ms, 1)
            component_total += duration_ms
        
        result["total_ms"] = round(component_total, 1)
        
        # RTT: wall-clock from pipeline start to first audio byte
        if self._first_audio_sent:
            rtt = (self._first_audio_sent - self.pipeline_start) * 1000
        else:
            # If first audio wasn't marked, use current time
            rtt = (time.time() - self.pipeline_start) * 1000
        
        result["rtt_ms"] = round(rtt, 1)
        result["overhead_ms"] = round(rtt - component_total, 1)
        
        return result
    
    def log_summary(self, providers: Optional[Dict[str, str]] = None):
        """
        Log a formatted latency summary.
        
        Args:
            providers: Optional dict of provider info, e.g. {"stt": "remote", "llm": "gemini"}
        """
        s = self.summary()
        providers = providers or {}
        
        parts = []
        for component in ["nc", "stt", "llm", "tts"]:
            key = f"{component}_ms"
            if key in s:
                provider_tag = f" [{providers[component]}]" if component in providers else ""
                parts.append(f"{component.upper()}={s[key]:.0f}ms{provider_tag}")
        
        parts.append(f"Total={s['total_ms']:.0f}ms")
        parts.append(f"RTT={s['rtt_ms']:.0f}ms")
        
        if s.get("overhead_ms", 0) > 10:  # Only log overhead if significant
            parts.append(f"Overhead={s['overhead_ms']:.0f}ms")
        
        logger.info(f"[{self.call_uuid}] ⏱️  Pipeline: {', '.join(parts)}")
