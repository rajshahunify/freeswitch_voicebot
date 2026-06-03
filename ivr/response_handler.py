"""
Response Handler
Manages audio playback responses via FreeSWITCH
"""

import logging
import subprocess
import os
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Handles audio playback responses for calls
    
    Features:
    - Play audio files via FreeSWITCH
    - Track playback state per call
    - Optional interruption support
    - Automatic duration detection
    """
    
    def __init__(self,
                 audio_base_path: str,
                 allow_interruptions: bool = False,
                 speaking_timeout: int = 30):
        """
        Initialize response handler
        
        Args:
            audio_base_path: Base directory for audio files
            allow_interruptions: Allow user to interrupt playback
            speaking_timeout: Max time to hold speaking lock (seconds)
        """
        self.audio_base_path = audio_base_path
        self.allow_interruptions = allow_interruptions
        self.speaking_timeout = speaking_timeout
        
        # Track playback state per call
        self.active_playbacks = {}  # {uuid: filename}
        self.bot_state = {}  # {uuid: {'speaking': bool, 'start_time': float}}
        self._barge_in_flags = {}  # {uuid: bool} — set by mark_stopped, checked by TTS pipeline
        
        logger.info(f"ResponseHandler initialized: interruptions={allow_interruptions}")
    
    def _init_bot_state(self, call_uuid: str):
        """Initialize bot state for a call"""
        if call_uuid not in self.bot_state:
            self.bot_state[call_uuid] = {
                'speaking': False,
                'start_time': None,
                'expected_end_time': 0.0
            }
    
    def is_speaking(self, call_uuid: str) -> bool:
        """
        Check if bot is currently speaking
        
        Args:
            call_uuid: Call identifier
            
        Returns:
            True if bot is speaking
        """
        self._init_bot_state(call_uuid)
        
        expected_end = self.bot_state[call_uuid].get('expected_end_time', 0.0)
        start_time = self.bot_state[call_uuid].get('start_time')
        current_time = time.time()
        
        if current_time < expected_end:
            # Check for timeout
            if start_time and (current_time - start_time) > self.speaking_timeout:
                logger.warning(f"Speaking timeout for {call_uuid}, releasing lock")
                self.bot_state[call_uuid]['expected_end_time'] = 0.0
                return False
            return True
            
        return False
    
    def play_audio(self, 
                   call_uuid: str,
                   filename: str,
                   text: Optional[str] = None) -> bool:
        """
        Play audio file to caller
        
        Args:
            call_uuid: Unique call identifier
            filename: Audio filename — can be:
                      - relative to base path (legacy IVR files)
                      - absolute path (TTS-generated files)
            text: Optional text that triggered this audio (for logging)
            
        Returns:
            True if playback started successfully
        """
        if not call_uuid:
            logger.error("Missing call UUID, cannot play audio")
            return False
        
        self._init_bot_state(call_uuid)
        
        # Check if we should ignore (bot is speaking and interruptions disabled)
        if not self.allow_interruptions and self.is_speaking(call_uuid):
            logger.debug(f"🔇 Ignoring playback request - bot is speaking")
            return False
        
        start_time = time.time()
        
        try:
            # Build full path — support both relative and absolute paths
            if os.path.isabs(filename):
                full_path = filename
            else:
                full_path = os.path.join(self.audio_base_path, filename)
            
            # Log
            if text:
                logger.info(f"🗣️  User: '{text}' → Playing: {os.path.basename(full_path)}")
            else:
                logger.info(f"🤖 Playing: {os.path.basename(full_path)}")
            
            # Stop previous audio if interruptions allowed
            if self.allow_interruptions:
                self._stop_audio(call_uuid)
            
            # Start broadcast
            broadcast_cmd = f"uuid_broadcast {call_uuid} {full_path} aleg"
            process = subprocess.run(
                ["fs_cli", "-x", broadcast_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Get audio duration
            duration = self._get_audio_duration(full_path)
            
            # Update tracker
            self.active_playbacks[call_uuid] = filename
            
            # Check result
            if "+OK" in process.stdout:
                current_time = time.time()
                elapsed_ms = (current_time - start_time) * 1000
                logger.info(f"▶️  Playing {duration:.1f}s audio ({elapsed_ms:.0f}ms setup)")
                
                # Mark bot as speaking with expected end time
                self.bot_state[call_uuid]['expected_end_time'] = current_time + duration
                self.bot_state[call_uuid]['start_time'] = current_time
                self.bot_state[call_uuid]['speaking'] = True
                
                return True
            else:
                logger.warning(f"⚠️  FreeSWITCH error: {process.stdout.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ FreeSWITCH command timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Playback error: {e}")
            return False
    
    def play_audio_queued(self, call_uuid: str, filename: str) -> bool:
        """
        Play audio sequentially — waits for previous audio to finish before starting.
        
        IMPORTANT: uuid_broadcast REPLACES current playback (it does NOT queue).
        So we must wait until the previous segment finishes before sending the next.
        
        Barge-in is still responsive because:
        - mark_stopped() sets expected_end_time = 0, which breaks our wait loop instantly
        - The caller in process_audio_segment checks is_speaking() after each call
        
        Args:
            call_uuid: Unique call identifier
            filename: Absolute path to WAV file
            
        Returns:
            True if played successfully
        """
        if not call_uuid:
            return False
        
        self._init_bot_state(call_uuid)
        
        try:
            # Build path
            full_path = filename if os.path.isabs(filename) else os.path.join(self.audio_base_path, filename)
            
            # Wait for previous audio to finish (with barge-in escape)
            expected_end = self.bot_state[call_uuid].get('expected_end_time', 0.0)
            current_time = time.time()
            
            if current_time < expected_end:
                wait_needed = expected_end - current_time
                logger.debug(f"⏳ Waiting {wait_needed:.1f}s for previous audio to finish")
                
                # Wait in small increments so barge-in (mark_stopped) breaks out fast
                while time.time() < self.bot_state[call_uuid].get('expected_end_time', 0.0):
                    time.sleep(0.05)  # 50ms checks
            
            # After waiting, check if barge-in killed us
            if self.bot_state[call_uuid].get('expected_end_time', 0.0) == 0.0 and expected_end > 0:
                logger.info(f"🛑 Barge-in during wait — skipping playback of {os.path.basename(full_path)}")
                return False
            
            logger.info(f"🤖 Playing: {os.path.basename(full_path)}")
            
            # Now broadcast — previous audio is finished
            broadcast_cmd = f"uuid_broadcast {call_uuid} {full_path} aleg"
            process = subprocess.run(
                ["fs_cli", "-x", broadcast_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Get audio duration
            duration = self._get_audio_duration(full_path)
            self.active_playbacks[call_uuid] = filename
            
            if "+OK" in process.stdout:
                current_time = time.time()
                
                # Set expected end time (not cumulative — we already waited)
                self.bot_state[call_uuid]['expected_end_time'] = current_time + duration
                
                # If this is the first segment, set start_time
                if not self.bot_state[call_uuid].get('start_time'):
                    self.bot_state[call_uuid]['start_time'] = current_time
                
                self.bot_state[call_uuid]['speaking'] = True
                
                logger.info(f"▶️  Playing {duration:.1f}s audio")
                return True
            else:
                logger.warning(f"⚠️  FreeSWITCH broadcast error: {process.stdout.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Playback error: {e}")
            return False
    
    def mark_stopped(self, call_uuid: str):
        """
        Mark playback as stopped (used by barge-in).
        Immediately releases the speaking lock so the pipeline can process new input.
        Also sets the barge-in flag so the TTS pipeline knows to stop.
        
        Args:
            call_uuid: Call identifier
        """
        self._init_bot_state(call_uuid)
        self.bot_state[call_uuid]['speaking'] = False
        self.bot_state[call_uuid]['start_time'] = None
        self.bot_state[call_uuid]['expected_end_time'] = 0.0
        self._barge_in_flags[call_uuid] = True
        logger.debug(f"[{call_uuid}] Speaking lock released (barge-in)")
    
    def was_barge_in(self, call_uuid: str) -> bool:
        """Check if barge-in was triggered for this call (set by mark_stopped)"""
        return self._barge_in_flags.get(call_uuid, False)
    
    def clear_barge_in(self, call_uuid: str):
        """Clear barge-in flag (call after processing)"""
        self._barge_in_flags[call_uuid] = False
    
    def _stop_audio(self, call_uuid: str):
        """
        Stop current audio playback
        
        Args:
            call_uuid: Call identifier
        """
        self._init_bot_state(call_uuid)
        self.bot_state[call_uuid]['expected_end_time'] = 0.0
        try:
            subprocess.run(
                ["fs_cli", "-x", f"uuid_break {call_uuid} all"],
                capture_output=True,
                timeout=2
            )
            logger.debug(f"Stopped audio for {call_uuid}")
        except Exception as e:
            logger.warning(f"Error stopping audio: {e}")
    
    def _get_audio_duration(self, filepath: str) -> float:
        """
        Get audio file duration using ffprobe
        
        Args:
            filepath: Full path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-i", filepath,
                "-show_entries", "format=duration",
                "-v", "quiet",
                "-of", "csv=p=0"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 2.0  # Default fallback
    

    
    def cleanup_call(self, call_uuid: str):
        """
        Cleanup state for ended call
        
        Args:
            call_uuid: Call identifier
        """
        if call_uuid in self.active_playbacks:
            del self.active_playbacks[call_uuid]
        
        if call_uuid in self.bot_state:
            del self.bot_state[call_uuid]
        
        if call_uuid in self._barge_in_flags:
            del self._barge_in_flags[call_uuid]
        
        logger.debug(f"Cleaned up call {call_uuid}")
    
    def get_stats(self) -> dict:
        """Get handler statistics"""
        return {
            'active_calls': len(self.bot_state),
            'active_playbacks': len(self.active_playbacks),
            'allow_interruptions': self.allow_interruptions
        }
    
    def get_call_state(self, call_uuid: str) -> dict:
        """
        Get state for specific call
        
        Args:
            call_uuid: Call identifier
            
        Returns:
            Call state dictionary
        """
        self._init_bot_state(call_uuid)
        return {
            'speaking': self.is_speaking(call_uuid),
            'current_playback': self.active_playbacks.get(call_uuid),
            'state': self.bot_state.get(call_uuid)
        }