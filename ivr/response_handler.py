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
        
        logger.info(f"ResponseHandler initialized: interruptions={allow_interruptions}")
    
    def _init_bot_state(self, call_uuid: str):
        """Initialize bot state for a call"""
        if call_uuid not in self.bot_state:
            self.bot_state[call_uuid] = {
                'speaking': False,
                'start_time': None
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
        
        # Check for timeout
        if self.bot_state[call_uuid]['speaking']:
            start_time = self.bot_state[call_uuid]['start_time']
            if start_time and (time.time() - start_time) > self.speaking_timeout:
                logger.warning(f"Speaking timeout for {call_uuid}, releasing lock")
                self.bot_state[call_uuid]['speaking'] = False
        
        return self.bot_state[call_uuid]['speaking']
    
    def play_audio(self, 
                   call_uuid: str,
                   filename: str,
                   text: Optional[str] = None) -> bool:
        """
        Play audio file to caller
        
        Args:
            call_uuid: Unique call identifier
            filename: Audio filename (relative to base path)
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
            # Build full path
            full_path = os.path.join(self.audio_base_path, filename)
            
            # Log
            if text:
                logger.info(f"🗣️  User: '{text}' → Playing: {filename}")
            else:
                logger.info(f"🤖 Playing: {filename}")
            
            # Stop previous audio if interruptions allowed
            if self.allow_interruptions:
                self._stop_audio(call_uuid)
            
            # Mark bot as speaking
            self.bot_state[call_uuid]['speaking'] = True
            self.bot_state[call_uuid]['start_time'] = time.time()
            
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
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"▶️  Playing {duration:.1f}s audio ({elapsed_ms:.0f}ms setup)")
                
                # Release lock after audio finishes
                threading.Thread(
                    target=self._release_lock_after_duration,
                    args=(call_uuid, duration),
                    daemon=True
                ).start()
                
                return True
            else:
                logger.warning(f"⚠️  FreeSWITCH error: {process.stdout.strip()}")
                self.bot_state[call_uuid]['speaking'] = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ FreeSWITCH command timeout")
            self.bot_state[call_uuid]['speaking'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Playback error: {e}")
            self.bot_state[call_uuid]['speaking'] = False
            return False
    
    def _stop_audio(self, call_uuid: str):
        """
        Stop current audio playback
        
        Args:
            call_uuid: Call identifier
        """
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
    
    def _release_lock_after_duration(self, call_uuid: str, duration: float):
        """
        Release speaking lock after audio duration
        
        Args:
            call_uuid: Call identifier
            duration: Audio duration in seconds
        """
        time.sleep(duration + 0.2)  # Small buffer
        
        if call_uuid in self.bot_state:
            self.bot_state[call_uuid]['speaking'] = False
            logger.debug(f"✓ Released speaking lock for {call_uuid}")
    
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