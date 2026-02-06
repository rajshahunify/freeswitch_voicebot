"""
Session Manager
Handles multiple concurrent call sessions with Redis state management
"""

import redis
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages call sessions across multiple workers
    
    Features:
    - Session creation and lifecycle tracking
    - Redis-backed state storage
    - Session locking for worker coordination
    - Automatic session cleanup
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 session_ttl: int = 3600):
        """
        Initialize session manager
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            session_ttl: Session time-to-live in seconds (1 hour default)
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.session_ttl = session_ttl
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"✓ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def create_session(self, call_uuid: str, metadata: Optional[Dict] = None) -> bool:
        """
        Create a new call session
        
        Args:
            call_uuid: Unique call identifier
            metadata: Optional metadata (caller_number, etc.)
            
        Returns:
            True if session created, False if already exists
        """
        session_key = f"session:{call_uuid}"
        
        # Check if session already exists
        if self.redis_client.exists(session_key):
            logger.warning(f"Session {call_uuid} already exists")
            return False
        
        session_data = {
            'call_uuid': call_uuid,
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'worker_id': None,
            'metadata': metadata or {}
        }
        
        # Store session
        self.redis_client.setex(
            session_key,
            self.session_ttl,
            json.dumps(session_data)
        )
        
        # Add to active sessions set
        self.redis_client.sadd('active_sessions', call_uuid)
        
        logger.info(f"✓ Created session {call_uuid}")
        return True
    
    def get_session(self, call_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Args:
            call_uuid: Call identifier
            
        Returns:
            Session data dict or None if not found
        """
        session_key = f"session:{call_uuid}"
        data = self.redis_client.get(session_key)
        
        if data:
            return json.loads(data)
        return None
    
    def update_session(self, call_uuid: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data
        
        Args:
            call_uuid: Call identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if updated, False if session not found
        """
        session = self.get_session(call_uuid)
        if not session:
            return False
        
        # Update fields
        session.update(updates)
        session['updated_at'] = datetime.utcnow().isoformat()
        
        # Save back to Redis
        session_key = f"session:{call_uuid}"
        self.redis_client.setex(
            session_key,
            self.session_ttl,
            json.dumps(session)
        )
        
        return True
    
    def end_session(self, call_uuid: str) -> bool:
        """
        End a call session
        
        Args:
            call_uuid: Call identifier
            
        Returns:
            True if session ended, False if not found
        """
        session_key = f"session:{call_uuid}"
        
        # Update status to ended
        self.update_session(call_uuid, {'status': 'ended'})
        
        # Remove from active sessions
        self.redis_client.srem('active_sessions', call_uuid)
        
        # Keep session data for a bit (for stats/debugging)
        self.redis_client.expire(session_key, 300)  # 5 minutes
        
        logger.info(f"✓ Ended session {call_uuid}")
        return True
    
    def acquire_session_lock(self, call_uuid: str, worker_id: str, timeout: int = 30) -> bool:
        """
        Acquire lock on a session (for worker coordination)
        
        Args:
            call_uuid: Call identifier
            worker_id: Worker identifier
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        lock_key = f"lock:{call_uuid}"
        
        # Try to acquire lock
        acquired = self.redis_client.set(
            lock_key,
            worker_id,
            nx=True,  # Only set if doesn't exist
            ex=timeout  # Expire after timeout
        )
        
        if acquired:
            self.update_session(call_uuid, {'worker_id': worker_id})
            logger.debug(f"Worker {worker_id} acquired lock on {call_uuid}")
        
        return bool(acquired)
    
    def release_session_lock(self, call_uuid: str, worker_id: str) -> bool:
        """
        Release lock on a session
        
        Args:
            call_uuid: Call identifier
            worker_id: Worker identifier (must match lock holder)
            
        Returns:
            True if lock released, False if not held by this worker
        """
        lock_key = f"lock:{call_uuid}"
        
        # Check if this worker holds the lock
        current_holder = self.redis_client.get(lock_key)
        if current_holder != worker_id:
            logger.warning(f"Worker {worker_id} tried to release lock held by {current_holder}")
            return False
        
        # Release lock
        self.redis_client.delete(lock_key)
        self.update_session(call_uuid, {'worker_id': None})
        
        logger.debug(f"Worker {worker_id} released lock on {call_uuid}")
        return True
    
    def get_active_sessions(self) -> list:
        """
        Get list of all active session UUIDs
        
        Returns:
            List of active call UUIDs
        """
        return list(self.redis_client.smembers('active_sessions'))
    
    def get_session_count(self) -> int:
        """
        Get count of active sessions
        
        Returns:
            Number of active sessions
        """
        return self.redis_client.scard('active_sessions')
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dictionary with stats
        """
        active_count = self.get_session_count()
        active_sessions = self.get_active_sessions()
        
        # Get session details
        sessions_info = []
        for uuid in active_sessions[:10]:  # Limit to 10 for display
            session = self.get_session(uuid)
            if session:
                sessions_info.append({
                    'uuid': uuid,
                    'status': session.get('status'),
                    'worker': session.get('worker_id'),
                    'duration': self._calculate_duration(session.get('created_at'))
                })
        
        return {
            'active_sessions': active_count,
            'sessions': sessions_info,
            'redis_connected': self.redis_client.ping()
        }
    
    def _calculate_duration(self, created_at_str: str) -> float:
        """Calculate session duration in seconds"""
        try:
            created = datetime.fromisoformat(created_at_str)
            duration = (datetime.utcnow() - created).total_seconds()
            return round(duration, 1)
        except:
            return 0.0
    
    def cleanup_stale_sessions(self, max_age_seconds: int = 3600):
        """
        Clean up sessions older than max_age
        
        Args:
            max_age_seconds: Maximum session age in seconds
        """
        active_sessions = self.get_active_sessions()
        cleaned = 0
        
        for uuid in active_sessions:
            session = self.get_session(uuid)
            if session:
                duration = self._calculate_duration(session.get('created_at'))
                if duration > max_age_seconds:
                    self.end_session(uuid)
                    cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale sessions")
        
        return cleaned


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_session_manager_instance: Optional[SessionManager] = None


def get_session_manager(**kwargs) -> SessionManager:
    """
    Get or create shared SessionManager instance
    
    Returns:
        Shared SessionManager instance
    """
    global _session_manager_instance
    
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager(**kwargs)
    
    return _session_manager_instance
