#!/usr/bin/env python3
"""Test script to check if registry import is causing the hang"""

import time
from loguru import logger

def test_registry():
    logger.info("🔄 Testing registry import...")
    start_time = time.time()
    
    try:
        from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
        import_time = time.time() - start_time
        logger.info(f"✅ ATTN_WEIGHT_REGISTER imported successfully in {import_time:.1f}s")
        logger.info(f"✅ Registry: {ATTN_WEIGHT_REGISTER}")
        return True
    except Exception as e:
        import_time = time.time() - start_time
        logger.error(f"❌ Registry import failed after {import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting registry test...")
    success = test_registry()
    if success:
        logger.info("🎉 Registry test PASSED")
    else:
        logger.error("💥 Registry test FAILED")
        exit(1)