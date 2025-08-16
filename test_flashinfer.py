#!/usr/bin/env python3
"""Test script to check if flashinfer import is causing the hang"""

import time
from loguru import logger

def test_flashinfer():
    logger.info("🔄 Testing flashinfer import...")
    start_time = time.time()
    
    try:
        import flashinfer
        import_time = time.time() - start_time
        logger.info(f"✅ flashinfer imported successfully in {import_time:.1f}s")
        logger.info(f"✅ flashinfer version: {flashinfer.__version__}")
        return True
    except ImportError as e:
        import_time = time.time() - start_time
        logger.info(f"ℹ️  flashinfer not available after {import_time:.1f}s: {e}")
        return True  # This is expected
    except Exception as e:
        import_time = time.time() - start_time
        logger.error(f"❌ flashinfer import failed after {import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting flashinfer test...")
    success = test_flashinfer()
    if success:
        logger.info("🎉 Flashinfer test PASSED")
    else:
        logger.error("💥 Flashinfer test FAILED")
        exit(1)