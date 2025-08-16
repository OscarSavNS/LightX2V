#!/usr/bin/env python3
"""Test script to verify the import fix works"""

import time
from loguru import logger

def test_import():
    logger.info("🔄 Testing import fix...")
    start_time = time.time()
    
    try:
        logger.info("📦 Importing lightx2v.infer...")
        from lightx2v.infer import init_runner
        import_time = time.time() - start_time
        logger.info(f"✅ lightx2v.infer imported successfully in {import_time:.1f}s")
        logger.info(f"✅ init_runner function available: {init_runner}")
        return True
    except Exception as e:
        import_time = time.time() - start_time
        logger.error(f"❌ Import failed after {import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting import test...")
    success = test_import()
    if success:
        logger.info("🎉 Import test PASSED")
    else:
        logger.error("💥 Import test FAILED")
        exit(1)