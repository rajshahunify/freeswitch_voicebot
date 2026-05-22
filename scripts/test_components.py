#!/usr/bin/env python3
"""
Component Testing Script
Tests each component individually to verify functionality
"""

import sys
import logging
import numpy as np
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    logger.info("=" * 60)
    logger.info("Testing Imports...")
    logger.info("=" * 60)
    
    try:
        import config
        logger.info("✓ config imported")
        
        from audio_pipeline import get_noise_canceller, get_vad_detector
        logger.info("✓ audio_pipeline imported")
        
        from ivr import IntentMatcher, ResponseHandler
        logger.info("✓ ivr imported")
        
        from stt_handler import STTHandler
        logger.info("✓ stt_handler imported")
        
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_noise_canceller():
    """Test noise cancellation component"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Noise Canceller...")
    logger.info("=" * 60)
    
    try:
        from audio_pipeline import get_noise_canceller
        import config
        
        # Initialize
        logger.info("Initializing noise canceller...")
        nc = get_noise_canceller(
            model_name=config.DF_MODEL,
            use_gpu=config.DF_USE_GPU
        )
        logger.info("✓ Noise canceller initialized")
        
        # Generate test audio (1 second of noise)
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Random noise
        noise = np.random.randint(-32768, 32767, samples, dtype=np.int16)
        audio_bytes = noise.tobytes()
        
        logger.info(f"Processing {len(audio_bytes)} bytes of test audio...")
        start_time = time.time()
        
        enhanced = nc.process_audio(audio_bytes)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ Processed in {processing_time:.0f}ms")
        logger.info(f"  Input: {len(audio_bytes)} bytes")
        logger.info(f"  Output: {len(enhanced)} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Noise canceller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vad_detector():
    """Test VAD component"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing VAD Detector...")
    logger.info("=" * 60)
    
    try:
        from audio_pipeline import get_vad_detector
        import config
        
        # Initialize
        logger.info("Initializing VAD detector...")
        vad = get_vad_detector(
            threshold=config.VAD_THRESHOLD,
            sample_rate=config.VAD_SAMPLE_RATE
        )
        logger.info("✓ VAD detector initialized")
        
        # Generate test audio chunks
        chunk_size = config.VAD_WINDOW_SIZE
        
        # Silence
        silence = np.zeros(chunk_size, dtype=np.int16)
        silence_bytes = silence.tobytes()
        
        # Noise (simulated speech)
        noise = np.random.randint(-32768, 32767, chunk_size, dtype=np.int16)
        noise_bytes = noise.tobytes()
        
        logger.info("Testing with silence...")
        is_speech, prob = vad.process_chunk(silence_bytes)
        logger.info(f"  Silence: is_speech={is_speech}, prob={prob:.3f}")
        
        logger.info("Testing with noise...")
        is_speech, prob = vad.process_chunk(noise_bytes)
        logger.info(f"  Noise: is_speech={is_speech}, prob={prob:.3f}")
        
        # Test streaming
        logger.info("Testing streaming detection...")
        vad.reset_state()
        
        for i in range(5):
            result = vad.process_stream(noise_bytes)
            logger.info(f"  Frame {i}: {result}")
        
        logger.info("✓ VAD detector working")
        return True
        
    except Exception as e:
        logger.error(f"✗ VAD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_buffer_manager():
    """Test audio buffer management"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Buffer Manager...")
    logger.info("=" * 60)
    
    try:
        from audio_pipeline import AudioBuffer
        
        buffer = AudioBuffer(
            min_length=1000,
            max_length=10000
        )
        logger.info("✓ Buffer initialized")
        
        # Simulate VAD results
        test_chunk = b'\x00' * 512
        
        # Speech start
        result = buffer.add_chunk(test_chunk, {
            'speech_start': True,
            'is_speech': True,
            'speech_end': False
        })
        logger.info(f"  After speech start: {result}")
        
        # Continue speech
        for i in range(5):
            result = buffer.add_chunk(test_chunk, {
                'speech_start': False,
                'is_speech': True,
                'speech_end': False
            })
        
        # Speech end
        result = buffer.add_chunk(test_chunk, {
            'speech_start': False,
            'is_speech': False,
            'speech_end': True
        })
        logger.info(f"  After speech end: {len(result) if result else 0} bytes")
        
        logger.info("✓ Buffer manager working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_matcher():
    """Test intent matching"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Intent Matcher...")
    logger.info("=" * 60)
    
    try:
        from ivr import IntentMatcher
        import config
        
        matcher = IntentMatcher(
            intent_keywords=config.INTENT_KEYWORDS,
            fuzzy_threshold=config.FUZZY_MATCH_THRESHOLD
        )
        logger.info("✓ Intent matcher initialized")
        
        # Test various inputs
        test_cases = [
            "hello",
            "hi there",
            "I need internet",
            "payment information",
            "xyz nonsense"
        ]
        
        for text in test_cases:
            result = matcher.match_intent(text)
            logger.info(f"  '{text}' → {result}")
        
        logger.info("✓ Intent matcher working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Intent matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Configuration...")
    logger.info("=" * 60)
    
    try:
        import config
        
        # Check required settings
        required = [
            'FREESWITCH_HOST',
            'FREESWITCH_PORT',
            'FREESWITCH_PASSWORD',
            'STT_URL',
            'AUDIO_BASE_PATH',
            'SAMPLE_RATE',
            'DF_MODEL',
            'VAD_THRESHOLD'
        ]
        
        for setting in required:
            value = getattr(config, setting)
            logger.info(f"  {setting}: {value}")
        
        logger.info("✓ Configuration complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("FreeSWITCH VoiceBot - Component Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Noise Canceller", test_noise_canceller),
        ("VAD Detector", test_vad_detector),
        ("Buffer Manager", test_buffer_manager),
        ("Intent Matcher", test_intent_matcher),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            logger.error(f"✗ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)