"""
FreeSWITCH ESL Agent
Handles incoming calls and connects audio to WebSocket server
"""

import greenswitch
import logging

# Import configuration
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize ESL connection
fs = greenswitch.InboundESL(
    host=config.FREESWITCH_HOST,
    port=config.FREESWITCH_PORT,
    password=config.FREESWITCH_PASSWORD
)


def on_call(event):
    """
    Handle incoming call event
    
    Steps:
    1. Answer the call
    2. Fork audio to WebSocket server
    3. Play welcome message (keeps connection alive)
    """
    uuid = event.headers.get("Unique-ID")
    caller_number = event.headers.get("Caller-Caller-ID-Number", "Unknown")
    
    logger.info("=" * 60)
    logger.info(f"📞 INCOMING CALL")
    logger.info(f"   UUID: {uuid}")
    logger.info(f"   From: {caller_number}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Answer the call
        logger.info("📱 Answering call...")
        answer_result = fs.send(f"api uuid_answer {uuid}")
        logger.debug(f"Answer result: {answer_result}")
        
        # Step 2: Fork audio to WebSocket
        # This sends audio stream to our Python server
        ws_url = config.WEBSOCKET_URL
        logger.info(f"🔌 Connecting audio to {ws_url}")
        
        # uuid_audio_fork syntax:
        # uuid_audio_fork <uuid> start <url> <mix_type> <sampling_rate>
        # mix_type: mono (single channel) or stereo
        # sampling_rate: 8k, 16k, 32k, 48k
        fork_cmd = f"api uuid_audio_fork {uuid} start {ws_url} mono 16k"
        fork_result = fs.send(fork_cmd)
        logger.info(f"✓ Audio fork result: {fork_result}")
        
        # Step 3: Play welcome message
        # This keeps the connection alive and gives user something to hear
        # The actual IVR logic happens in the Python server
        welcome_file = "/usr/local/freeswitch/sounds/en/us/callie/ivr/ivr-welcome_to_freeswitch.wav"
        logger.info("🎵 Playing welcome message...")
        broadcast_result = fs.send(f"api uuid_broadcast {uuid} {welcome_file}")
        logger.debug(f"Broadcast result: {broadcast_result}")
        
        logger.info("✓ Call setup complete")
        
    except Exception as e:
        logger.error(f"❌ Error handling call: {e}", exc_info=True)


def main():
    """
    Main function - connects to FreeSWITCH and listens for calls
    """
    logger.info("=" * 60)
    logger.info("🚀 FreeSWITCH ESL Agent Starting")
    logger.info("=" * 60)
    logger.info(f"FreeSWITCH: {config.FREESWITCH_HOST}:{config.FREESWITCH_PORT}")
    logger.info(f"WebSocket: {config.WEBSOCKET_URL}")
    logger.info("=" * 60)
    
    try:
        # Connect to FreeSWITCH
        logger.info("🔌 Connecting to FreeSWITCH ESL...")
        fs.connect()
        logger.info("✓ Connected to FreeSWITCH")
        
        # Subscribe to CHANNEL_PARK events
        # CHANNEL_PARK is triggered when a call is parked/answered
        logger.info("📡 Subscribing to call events...")
        fs.send("events plain CHANNEL_PARK")
        
        # Register event handler
        fs.register_handle("CHANNEL_PARK", on_call)
        logger.info("✓ Event handler registered")
        
        logger.info("=" * 60)
        logger.info("✅ READY - Waiting for incoming calls...")
        logger.info("=" * 60)
        
        # Start event processing loop
        # This blocks and processes events continuously
        fs.process_events()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    finally:
        logger.info("👋 ESL Agent stopped")


if __name__ == "__main__":
    main()
