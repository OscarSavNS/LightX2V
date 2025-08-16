import argparse

import torch.distributed as dist
from loguru import logger

# Lazy imports - moved to function level to avoid hanging during module import
# from lightx2v.common.ops import *
# from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner  # noqa: F401
# from lightx2v.models.runners.graph_runner import GraphRunner
# from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner  # noqa: F401
# from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, Wan22MoeAudioRunner, WanAudioRunner  # noqa: F401
# from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner  # noqa: F401
# from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
# from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
# from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext
# from lightx2v.utils.registry_factory import RUNNER_REGISTER  # Lazy import
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


def init_runner(config):
    import time
    import torch
    
    logger.info(f"🎲 Setting seed: {config.seed}")
    try:
        seed_all(config.seed)
        logger.info(f"✅ Seed {config.seed} set successfully")
    except Exception as e:
        logger.error(f"❌ Failed to set seed: {e}")
        raise
    
    # Lazy import of heavy modules only when actually needed
    logger.info("📦 Lazy importing heavy ML modules...")
    lazy_import_start = time.time()
    
    try:
        logger.info("📦 Importing essential registry modules...")
        # Import specific modules needed for registry registration without triggering heavy imports
        # These imports register the "Default" entries needed for model weight initialization
        from lightx2v.common.ops.conv.conv3d import Conv3dWeight  # Registers "Default" in CONV3D_WEIGHT_REGISTER
        from lightx2v.common.ops.conv.conv2d import Conv2dWeight  # Registers "Default" in CONV2D_WEIGHT_REGISTER
        from lightx2v.common.ops.norm.layer_norm_weight import LayerNormWeight  # Registers "Default" in LN_WEIGHT_REGISTER
        from lightx2v.common.ops.norm.rms_norm_weight import RMSNormWeight  # Registers "Default" in RMS_WEIGHT_REGISTER
        from lightx2v.common.ops.mm.mm_weight import LinearWeight  # Registers "Default" and "Default-Force-FP32" in MM_WEIGHT_REGISTER
        from lightx2v.common.ops.tensor.tensor import TensorOp  # Registers "Default" in TENSOR_REGISTER
        logger.info("✅ Essential registry modules imported")
        
        logger.info("📦 Importing registry...")
        from lightx2v.utils.registry_factory import RUNNER_REGISTER
        logger.info("✅ Registry imported")
        
        logger.info("📦 Skipping GraphRunner and runner class imports during init_runner...")
        # NOTE: We'll import these only when needed for graph mode or when looking up the runner class
        # This avoids triggering the heavy model imports during the initial setup
        logger.info("✅ Runner class imports skipped")
        
        lazy_import_time = time.time() - lazy_import_start
        logger.info(f"✅ All lazy imports completed in {lazy_import_time:.1f}s")
        
    except Exception as e:
        lazy_import_time = time.time() - lazy_import_start
        logger.error(f"❌ Failed to import heavy modules after {lazy_import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        raise
    
    # Log initial memory state
    logger.info("💾 Checking GPU memory state...")
    try:
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 Initial GPU memory: {initial_memory:.2f}GB / {total_memory:.1f}GB")
        else:
            logger.error("❌ CUDA not available!")
            initial_memory = 0
            total_memory = 0
    except Exception as e:
        logger.error(f"❌ Failed to check GPU memory: {e}")
        initial_memory = 0
        total_memory = 0

    logger.info(f"🏗️  Creating runner for model class: {config.model_cls}")
    runner_start_time = time.time()
    
    logger.info("🔍 Checking graph mode...")
    try:
        graph_mode = CHECK_ENABLE_GRAPH_MODE()
        logger.info(f"📊 Graph mode: {graph_mode}")
    except Exception as e:
        logger.error(f"❌ Failed to check graph mode: {e}")
        graph_mode = False
    
    # Import the specific runner classes needed based on config.model_cls
    logger.info(f"📦 Importing specific runner for model class: {config.model_cls}")
    runner_import_start = time.time()
    
    runner_class = None
    try:
        if config.model_cls == "wan2.1_distill":
            from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
            runner_class = WanDistillRunner
        elif config.model_cls == "wan2.1":
            from lightx2v.models.runners.wan.wan_runner import WanRunner
            runner_class = WanRunner
        elif config.model_cls == "wan2.2_moe":
            from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner
            runner_class = Wan22MoeRunner
        elif config.model_cls == "hunyuan":
            from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
            runner_class = HunyuanRunner
        elif config.model_cls == "cogvideox":
            from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner
            runner_class = CogvideoxRunner
        else:
            logger.warning(f"⚠️ Unknown model class {config.model_cls}, falling back to registry lookup")
            # Import all runners for registry lookup fallback
            from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
            from lightx2v.models.runners.wan.wan_runner import WanRunner  # noqa: F401
            runner_class = RUNNER_REGISTER[config.model_cls]
        
        runner_import_time = time.time() - runner_import_start
        logger.info(f"✅ Runner class imported in {runner_import_time:.1f}s: {runner_class}")
        
    except Exception as e:
        runner_import_time = time.time() - runner_import_start
        logger.error(f"❌ Failed to import runner class after {runner_import_time:.1f}s: {e}")
        import traceback
        logger.error(f"📜 Traceback: {traceback.format_exc()}")
        raise

    if graph_mode:
        logger.info("🔄 Graph mode enabled - creating default runner first...")
        try:
            logger.info("📦 Importing GraphRunner for graph mode...")
            from lightx2v.models.runners.graph_runner import GraphRunner
            logger.info("✅ GraphRunner imported")
            
            logger.info("🏗️  Instantiating default runner...")
            instantiate_start = time.time()
            default_runner = runner_class(config)
            instantiate_time = time.time() - instantiate_start
            logger.info(f"✅ Default runner instantiated in {instantiate_time:.1f}s")
            
            logger.info("📦 Initializing default runner modules...")
            module_start_time = time.time()
            default_runner.init_modules()
            module_time = time.time() - module_start_time
            logger.info(f"✅ Default runner modules initialized in {module_time:.1f}s")
            
            logger.info("🔗 Wrapping with GraphRunner...")
            wrap_start = time.time()
            runner = GraphRunner(default_runner)
            wrap_time = time.time() - wrap_start
            logger.info(f"✅ GraphRunner wrapping completed in {wrap_time:.1f}s")
        except Exception as e:
            logger.error(f"❌ Failed in graph mode initialization: {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            raise
    else:
        logger.info("📦 Creating standard runner and initializing modules...")
        try:
            logger.info("🏗️  Instantiating runner...")
            instantiate_start = time.time()
            runner = runner_class(config)
            instantiate_time = time.time() - instantiate_start
            logger.info(f"✅ Runner instantiated in {instantiate_time:.1f}s")
            
            logger.info("📦 Calling runner.init_modules()...")
            module_start_time = time.time()
            runner.init_modules()
            module_time = time.time() - module_start_time
            logger.info(f"✅ Runner modules initialized in {module_time:.1f}s")
        except Exception as e:
            logger.error(f"❌ Failed in standard mode initialization: {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            raise
    
    runner_total_time = time.time() - runner_start_time
    logger.info(f"🎯 Runner initialization completed in {runner_total_time:.1f}s")
    
    # Log final memory state
    logger.info("💾 Checking final GPU memory state...")
    try:
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            memory_used = final_memory - initial_memory
            logger.info(f"💾 Final GPU memory: {final_memory:.2f}GB / {total_memory:.1f}GB (used: +{memory_used:.2f}GB)")
    except Exception as e:
        logger.error(f"❌ Failed to check final GPU memory: {e}")
    
    logger.info("🎉 init_runner completed successfully!")
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=[
            "wan2.1",
            "hunyuan",
            "wan2.1_distill",
            "wan2.1_causvid",
            "wan2.1_skyreels_v2_df",
            "cogvideox",
            "wan2.1_audio",
            "wan2.2_moe",
            "wan2.2",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
        ],
        default="wan2.1",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file for audio-to-video (a2v) task")

    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()

    # set config
    config = set_config(args)

    if config.parallel:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        set_parallel_config(config)

    print_config(config)

    with ProfilingContext("Total Cost"):
        runner = init_runner(config)
        runner.run_pipeline()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
