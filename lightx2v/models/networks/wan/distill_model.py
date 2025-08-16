import glob
import os

import torch
from loguru import logger

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanDistillModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        import time
        start_time = time.time()
        
        logger.info(f"🏗️  WanDistillModel.__init__() starting...")
        logger.info(f"📋 Model path: {model_path}")
        logger.info(f"💻 Device: {device}")
        
        try:
            logger.info("🔄 Calling super().__init__() - this will load the heavy model...")
            super_start = time.time()
            super().__init__(model_path, config, device)
            super_time = time.time() - super_start
            logger.info(f"✅ super().__init__() completed in {super_time:.1f}s")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 WanDistillModel.__init__() completed in {total_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed in WanDistillModel.__init__(): {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            raise

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        import time
        start_time = time.time()
        
        # For the old t2v distill model: https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
        ckpt_path = os.path.join(self.model_path, "distill_model.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"📦 Loading PyTorch weights from {ckpt_path}")
            file_size = os.path.getsize(ckpt_path) / 1024**3
            logger.info(f"📊 File size: {file_size:.2f}GB")
            
            load_start = time.time()
            weight_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            load_time = time.time() - load_start
            logger.info(f"⏱️  Loaded PyTorch file in {load_time:.1f}s ({file_size/load_time:.1f}GB/s)")
            
            logger.info(f"🔄 Converting {len(weight_dict)} tensors to target device...")
            convert_start = time.time()
            weight_dict = {
                key: (weight_dict[key].to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else weight_dict[key].to(GET_SENSITIVE_DTYPE())).pin_memory().to(self.device)
                for key in weight_dict.keys()
            }
            convert_time = time.time() - convert_start
            logger.info(f"✅ Tensor conversion completed in {convert_time:.1f}s")
            
            total_time = time.time() - start_time
            logger.info(f"🎯 Total model loading time: {total_time:.1f}s")
            return weight_dict

        logger.info(f"🔍 Searching for safetensors files in {self.model_path}")
        if self.config.get("enable_dynamic_cfg", False):
            safetensors_path = find_hf_model_path(self.config, self.model_path, "dit_distill_ckpt", subdir="distill_cfg_models")
            logger.info(f"📂 Using dynamic CFG model path: {safetensors_path}")
        else:
            safetensors_path = find_hf_model_path(self.config, self.model_path, "dit_distill_ckpt", subdir="distill_models")
            logger.info(f"📂 Using standard distill model path: {safetensors_path}")

        safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        logger.info(f"📋 Found {len(safetensors_files)} safetensors files to load")
        
        if not safetensors_files:
            # Try loading from current directory
            current_dir_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
            if current_dir_files:
                logger.info(f"📁 Found {len(current_dir_files)} safetensors files in model root directory")
                safetensors_files = current_dir_files
            else:
                logger.error(f"❌ No safetensors files found in {safetensors_path} or {self.model_path}")
                raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        weight_dict = {}
        total_size = 0
        
        for i, file_path in enumerate(safetensors_files):
            file_size = os.path.getsize(file_path) / 1024**3
            total_size += file_size
            logger.info(f"📦 Loading file {i+1}/{len(safetensors_files)}: {os.path.basename(file_path)} ({file_size:.2f}GB)")
            
            file_start = time.time()
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            file_time = time.time() - file_start
            
            weight_dict.update(file_weights)
            logger.info(f"✅ Loaded {len(file_weights)} tensors in {file_time:.1f}s ({file_size/file_time:.1f}GB/s)")
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Total safetensors loading time: {total_time:.1f}s for {total_size:.2f}GB ({total_size/total_time:.1f}GB/s)")
        logger.info(f"📊 Final weight dictionary contains {len(weight_dict)} tensors")
        
        return weight_dict


class Wan22MoeDistillModel(WanDistillModel, WanModel):
    def __init__(self, model_path, config, device):
        WanDistillModel.__init__(self, model_path, config, device)

    @torch.no_grad()
    def infer(self, inputs):
        return WanModel.infer(self, inputs)
