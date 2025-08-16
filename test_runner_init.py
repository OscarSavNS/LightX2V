#!/usr/bin/env python3
"""Test script to verify the runner initialization works"""

import time
import argparse
from loguru import logger

def test_runner_init():
    logger.info("🔄 Testing runner initialization...")
    start_time = time.time()
    
    try:
        logger.info("📦 Importing modules...")
        from lightx2v.infer import init_runner
        from lightx2v.utils.set_config import set_config
        import_time = time.time() - start_time
        logger.info(f"✅ Modules imported in {import_time:.1f}s")
        
        logger.info("⚙️ Creating test configuration...")
        # Simulate the same args used by the server
        class Args:
            def __init__(self):
                self.model_cls = "wan2.1_distill"
                self.task = "i2v"
                self.model_path = "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/distill_models"
                self.config_json = "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/configs/example_config.json"
                self.nproc_per_node = 4
                self.use_prompt_enhancer = False
                self.prompt = ""
                self.negative_prompt = ""
                self.image_path = ""
                self.audio_path = ""
                self.save_video_path = "./output_lightx2v.mp4"
        
        args = Args()
        logger.info(f"📋 Test config: model_cls={args.model_cls}, model_path={args.model_path}")
        
        logger.info("🔧 Creating configuration...")
        config_start = time.time()
        config = set_config(args)
        config_time = time.time() - config_start
        logger.info(f"✅ Configuration created in {config_time:.1f}s")
        
        logger.info("🏗️ Calling init_runner() (this will load the 31GB model)...")
        runner_start = time.time()
        runner = init_runner(config)
        runner_time = time.time() - runner_start
        logger.info(f"✅ Runner initialized in {runner_time:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Total test completed in {total_time:.1f}s")
        logger.info(f"✅ Runner object: {runner}")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Runner init failed after {total_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting runner initialization test...")
    success = test_runner_init()
    if success:
        logger.info("🎉 Runner init test PASSED")
    else:
        logger.error("💥 Runner init test FAILED")
        exit(1)