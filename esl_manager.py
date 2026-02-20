"""
ESL Connection Manager
Replaces all fs_cli subprocess calls with a persistent ESL connection.

Benefits:
- ~10x faster than spawning fs_cli subprocess per command
- Single persistent TCP connection (no connection overhead per call)
- Thread-safe with locking
- No fs_cli binary dependency (Docker-friendly)
"""

import greenswitch
import json
import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ESLManager:
    """
    Thread-safe ESL connection manager for FreeSWITCH.
    
    Provides the same commands as fs_cli but over a persistent
    TCP connection (~2-5ms vs ~50-100ms per command).
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8021, 
                 password: str = 'ClueCon', auto_reconnect: bool = True):
        """
        Initialize ESL Manager.
        
        Args:
            host: FreeSWITCH ESL host
            port: FreeSWITCH ESL port
            password: ESL password
            auto_reconnect: Automatically reconnect on connection loss
        """
        self._host = host
        self._port = port
        self._password = password
        self._auto_reconnect = auto_reconnect
        self._conn: Optional[greenswitch.InboundESL] = None
        self._lock = threading.Lock()
        self._connected = False
    
    def connect(self) -> bool:
        """
        Establish ESL connection to FreeSWITCH.
        
        Returns:
            True if connected successfully
        """
        try:
            self._conn = greenswitch.InboundESL(
                host=self._host,
                port=self._port,
                password=self._password
            )
            self._conn.connect()
            self._connected = True
            logger.info(f"✓ ESL connected to {self._host}:{self._port}")
            return True
        except Exception as e:
            self._connected = False
            logger.error(f"❌ ESL connection failed: {e}")
            return False
    
    def _ensure_connected(self):
        """Ensure ESL connection is alive, reconnect if needed."""
        if not self._connected or self._conn is None:
            if self._auto_reconnect:
                logger.info("🔄 ESL reconnecting...")
                self.connect()
            else:
                raise ConnectionError("ESL not connected")
    
    def send(self, command: str) -> str:
        """
        Send an API command via ESL.
        
        This replaces: subprocess.run(["fs_cli", "-x", command], ...)
        
        Args:
            command: FreeSWITCH API command (without 'api ' prefix)
            
        Returns:
            Command output as string
        """
        with self._lock:
            try:
                self._ensure_connected()
                result = self._conn.send(f"api {command}")
                # greenswitch returns an ESLEvent, extract the body/data
                if hasattr(result, 'data'):
                    return result.data.strip() if result.data else ""
                return str(result).strip()
            except Exception as e:
                logger.error(f"❌ ESL command failed: {command} → {e}")
                self._connected = False
                # One retry on failure
                if self._auto_reconnect:
                    try:
                        self.connect()
                        result = self._conn.send(f"api {command}")
                        if hasattr(result, 'data'):
                            return result.data.strip() if result.data else ""
                        return str(result).strip()
                    except Exception as retry_error:
                        logger.error(f"❌ ESL retry failed: {retry_error}")
                return ""
    
    # =========================================================================
    # HIGH-LEVEL COMMANDS (replacements for specific fs_cli calls)
    # =========================================================================
    
    def show_channels_json(self) -> Dict[str, Any]:
        """
        Get active channels as JSON.
        
        Replaces: fs_cli -x "show channels as json"
        
        Returns:
            Parsed JSON dict with channel data
        """
        output = self.send("show channels as json")
        try:
            if output:
                return json.loads(output)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"No channels or invalid JSON: {e}")
        return {}
    
    def uuid_break(self, uuid: str) -> str:
        """
        Stop audio playback on a channel.
        
        Replaces: fs_cli -x "uuid_break <uuid> all"
        
        Args:
            uuid: Call UUID
            
        Returns:
            FreeSWITCH response
        """
        result = self.send(f"uuid_break {uuid} all")
        logger.debug(f"uuid_break {uuid}: {result}")
        return result
    
    def uuid_broadcast(self, uuid: str, path: str, leg: str = "aleg") -> str:
        """
        Play audio file to a channel.
        
        Replaces: fs_cli -x "uuid_broadcast <uuid> <path> aleg"
        
        Args:
            uuid: Call UUID
            path: Full path to audio file
            leg: Which leg to play on (aleg, bleg, both)
            
        Returns:
            FreeSWITCH response ("+OK" on success)
        """
        result = self.send(f"uuid_broadcast {uuid} {path} {leg}")
        logger.debug(f"uuid_broadcast {uuid} {path}: {result}")
        return result
    
    def uuid_answer(self, uuid: str) -> str:
        """
        Answer a call.
        
        Replaces: fs_cli -x "uuid_answer <uuid>"
        
        Args:
            uuid: Call UUID
            
        Returns:
            FreeSWITCH response
        """
        return self.send(f"uuid_answer {uuid}")
    
    def uuid_audio_fork(self, uuid: str, ws_url: str, 
                         mix_type: str = "mono", 
                         sample_rate: str = "16k") -> str:
        """
        Fork audio to a WebSocket URL.
        
        Replaces: fs_cli -x "uuid_audio_fork <uuid> start <url> <mix> <rate>"
        
        Args:
            uuid: Call UUID
            ws_url: WebSocket URL to stream audio to
            mix_type: mono, stereo, or mixed
            sample_rate: 8k or 16k
            
        Returns:
            FreeSWITCH response
        """
        return self.send(f"uuid_audio_fork {uuid} start {ws_url} {mix_type} {sample_rate}")
    
    def get_status(self) -> str:
        """Get FreeSWITCH status."""
        return self.send("status")
    
    def is_connected(self) -> bool:
        """Check if ESL is connected."""
        return self._connected
    
    def disconnect(self):
        """Disconnect from ESL."""
        self._connected = False
        self._conn = None
        logger.info("ESL disconnected")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_esl_instance: Optional[ESLManager] = None
_esl_lock = threading.Lock()


def get_esl_manager(host: str = '127.0.0.1', port: int = 8021, 
                     password: str = 'ClueCon') -> ESLManager:
    """
    Get or create shared ESLManager instance.
    
    Thread-safe singleton pattern.
    
    Args:
        host: FreeSWITCH ESL host
        port: FreeSWITCH ESL port  
        password: ESL password
        
    Returns:
        Shared ESLManager instance
    """
    global _esl_instance
    
    with _esl_lock:
        if _esl_instance is None:
            _esl_instance = ESLManager(host=host, port=port, password=password)
            _esl_instance.connect()
        
        return _esl_instance
