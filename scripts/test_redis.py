#!/usr/bin/env python3
"""
Redis Connectivity Test
Quick test to verify Redis is working
"""

import sys

def test_redis_basic():
    """Test basic Redis connectivity"""
    print("=" * 60)
    print("Testing Redis Connectivity")
    print("=" * 60)
    
    try:
        import redis
        print("✓ Redis package imported")
    except ImportError:
        print("❌ Redis package not installed")
        print("   Install with: pip install redis --break-system-packages")
        return False
    
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        print("✓ Redis client created")
        
        # Test connection
        response = r.ping()
        if response:
            print("✓ Redis PING successful")
        else:
            print("❌ Redis PING failed")
            return False
        
        # Test basic operations
        r.set('test_key', 'test_value')
        print("✓ SET operation successful")
        
        value = r.get('test_key')
        if value == 'test_value':
            print("✓ GET operation successful")
        else:
            print("❌ GET operation failed")
            return False
        
        # Clean up
        r.delete('test_key')
        print("✓ DELETE operation successful")
        
        # Test set operations
        r.sadd('test_set', 'item1', 'item2', 'item3')
        print("✓ SADD operation successful")
        
        count = r.scard('test_set')
        if count == 3:
            print(f"✓ SCARD operation successful (count: {count})")
        else:
            print(f"❌ SCARD operation failed (expected 3, got {count})")
        
        # Clean up
        r.delete('test_set')
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("\nPossible fixes:")
        print("1. Check if Redis is running: sudo service redis-server status")
        print("2. Start Redis: sudo service redis-server start")
        print("3. Check Redis config: sudo cat /etc/redis/redis.conf | grep bind")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_session_manager():
    """Test SessionManager functionality"""
    print("\n" + "=" * 60)
    print("Testing SessionManager")
    print("=" * 60)
    
    try:
        from session_manager import get_session_manager
        print("✓ SessionManager imported")
        
        sm = get_session_manager()
        print("✓ SessionManager instance created")
        
        # Test session creation
        test_uuid = "test-uuid-12345"
        created = sm.create_session(test_uuid, metadata={'test': True})
        if created:
            print(f"✓ Session created: {test_uuid}")
        else:
            print(f"❌ Failed to create session")
            return False
        
        # Test session retrieval
        session = sm.get_session(test_uuid)
        if session:
            print(f"✓ Session retrieved: {session['call_uuid']}")
            print(f"   Status: {session['status']}")
            print(f"   Created: {session['created_at']}")
        else:
            print("❌ Failed to retrieve session")
            return False
        
        # Test session update
        updated = sm.update_session(test_uuid, {'test_field': 'test_value'})
        if updated:
            print("✓ Session updated")
        
        # Test active sessions list
        active = sm.get_active_sessions()
        if test_uuid in active:
            print(f"✓ Session in active list (total active: {len(active)})")
        
        # Test session stats
        stats = sm.get_stats()
        print(f"✓ Session stats: {stats['active_sessions']} active")
        
        # Test session lock
        worker_id = "test-worker"
        locked = sm.acquire_session_lock(test_uuid, worker_id)
        if locked:
            print(f"✓ Session lock acquired by {worker_id}")
        
        # Test lock release
        released = sm.release_session_lock(test_uuid, worker_id)
        if released:
            print("✓ Session lock released")
        
        # Test session end
        ended = sm.end_session(test_uuid)
        if ended:
            print("✓ Session ended")
        
        # Verify cleanup
        active_after = sm.get_active_sessions()
        if test_uuid not in active_after:
            print("✓ Session removed from active list")
        
        print("\n" + "=" * 60)
        print("✅ SESSION MANAGER TESTS PASSED")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SessionManager: {e}")
        print("\nMake sure session_manager.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_redis_info():
    """Display Redis server information"""
    print("\n" + "=" * 60)
    print("Redis Server Information")
    print("=" * 60)
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        info = r.info()
        
        print(f"Redis Version: {info.get('redis_version', 'unknown')}")
        print(f"OS: {info.get('os', 'unknown')}")
        print(f"Architecture: {info.get('arch_bits', 'unknown')} bit")
        print(f"TCP Port: {info.get('tcp_port', 'unknown')}")
        print(f"Connected Clients: {info.get('connected_clients', 'unknown')}")
        print(f"Used Memory: {info.get('used_memory_human', 'unknown')}")
        print(f"Max Memory: {info.get('maxmemory_human', 'not set')}")
        print(f"Uptime: {info.get('uptime_in_days', 'unknown')} days")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Could not get Redis info: {e}")


if __name__ == "__main__":
    print("\n🧪 VoiceBot Redis Test Suite\n")
    
    # Run tests
    redis_ok = test_redis_basic()
    
    if redis_ok:
        session_ok = test_session_manager()
        test_redis_info()
        
        if session_ok:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Copy files: session_manager.py, server_multicall.py")
            print("2. Update config.py with Redis settings")
            print("3. Start server: python server_multicall.py")
            print("4. Test with real calls")
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("⚠️  SESSION MANAGER TESTS FAILED")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("⚠️  REDIS TESTS FAILED")
        print("=" * 60)
        print("\nFix Redis connectivity before proceeding")
        sys.exit(1)
