#!/usr/bin/env python3
"""Test script to identify which specific import is hanging"""

import time
from loguru import logger

def test_individual_imports():
    logger.info("🔄 Testing individual imports to identify hang location...")
    
    imports_to_test = [
        ("lightx2v.utils.registry_factory", "RUNNER_REGISTER"),
        ("lightx2v.models.runners.graph_runner", "GraphRunner"),
        ("lightx2v.models.runners.cogvideox.cogvidex_runner", "CogvideoxRunner"),
        ("lightx2v.models.runners.hunyuan.hunyuan_runner", "HunyuanRunner"),
        ("lightx2v.models.runners.wan.wan_audio_runner", "WanAudioRunner"),
        ("lightx2v.models.runners.wan.wan_causvid_runner", "WanCausVidRunner"),
        ("lightx2v.models.runners.wan.wan_distill_runner", "WanDistillRunner"),
        ("lightx2v.models.runners.wan.wan_runner", "WanRunner"),
        ("lightx2v.models.runners.wan.wan_skyreels_v2_df_runner", "WanSkyreelsV2DFRunner"),
    ]
    
    for module_name, class_name in imports_to_test:
        logger.info(f"📦 Testing import: {module_name}")
        start_time = time.time()
        
        try:
            exec(f"from {module_name} import {class_name}")
            import_time = time.time() - start_time
            logger.info(f"✅ {module_name} imported successfully in {import_time:.1f}s")
        except Exception as e:
            import_time = time.time() - start_time
            logger.error(f"❌ {module_name} import failed after {import_time:.1f}s: {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return False
    
    logger.info("🎉 All individual imports completed successfully!")
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting individual import test...")
    # First test common ops
    logger.info("📦 Testing common ops import...")
    start_time = time.time()
    try:
        import lightx2v.common.ops as ops
        import_time = time.time() - start_time
        logger.info(f"✅ lightx2v.common.ops imported in {import_time:.1f}s")
    except Exception as e:
        import_time = time.time() - start_time
        logger.error(f"❌ lightx2v.common.ops import failed after {import_time:.1f}s: {e}")
        exit(1)
    
    success = test_individual_imports()
    if success:
        logger.info("🎉 All imports test PASSED")
    else:
        logger.error("💥 Some imports test FAILED")
        exit(1)