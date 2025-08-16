#!/usr/bin/env python3
"""Test script to check if template import is causing the hang"""

import time
from loguru import logger

def test_template():
    logger.info("🔄 Testing template import...")
    start_time = time.time()
    
    try:
        from lightx2v.common.ops.attn.template import AttnWeightTemplate
        import_time = time.time() - start_time
        logger.info(f"✅ AttnWeightTemplate imported successfully in {import_time:.1f}s")
        logger.info(f"✅ Template: {AttnWeightTemplate}")
        return True
    except Exception as e:
        import_time = time.time() - start_time
        logger.error(f"❌ Template import failed after {import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting template test...")
    success = test_template()
    if success:
        logger.info("🎉 Template test PASSED")
    else:
        logger.error("💥 Template test FAILED")
        exit(1)