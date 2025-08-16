#!/usr/bin/env python3
"""Test script to identify which attn import is hanging"""

import time
from loguru import logger

def test_attn_imports():
    logger.info("🔄 Testing individual attn imports...")
    
    attn_modules = [
        "lightx2v.common.ops.attn.radial_attn",
        "lightx2v.common.ops.attn.ring_attn", 
        "lightx2v.common.ops.attn.sage_attn",
        "lightx2v.common.ops.attn.sparge_attn",
        "lightx2v.common.ops.attn.torch_sdpa",
        "lightx2v.common.ops.attn.ulysses_attn",
    ]
    
    for module_name in attn_modules:
        logger.info(f"📦 Testing import: {module_name}")
        start_time = time.time()
        
        try:
            __import__(module_name)
            import_time = time.time() - start_time
            logger.info(f"✅ {module_name} imported successfully in {import_time:.1f}s")
        except Exception as e:
            import_time = time.time() - start_time
            logger.error(f"❌ {module_name} import failed after {import_time:.1f}s: {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return False
    
    logger.info("🎉 All attn imports completed!")
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting attn import test...")
    success = test_attn_imports()
    if success:
        logger.info("🎉 Attn imports test PASSED")
    else:
        logger.error("💥 Attn imports test FAILED")
        exit(1)