import os

from loguru import logger

from lightx2v.models.networks.wan.distill_model import Wan22MoeDistillModel, WanDistillModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan22StepDistillScheduler, WanStepDistillScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_distill")
class WanDistillRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        import time
        start_time = time.time()
        
        logger.info(f"🔧 WanDistillRunner.load_transformer() starting...")
        logger.info(f"📋 Model path: {self.config.model_path}")
        logger.info(f"💻 Init device: {self.init_device}")
        
        try:
            if self.config.get("lora_configs") and self.config.lora_configs:
                logger.info("📦 Loading transformer with LoRA configurations...")
                
                logger.info("🏗️  Creating WanModel...")
                wan_start = time.time()
                model = WanModel(
                    self.config.model_path,
                    self.config,
                    self.init_device,
                )
                wan_time = time.time() - wan_start
                logger.info(f"✅ WanModel created in {wan_time:.1f}s")
                
                logger.info("🔗 Creating LoRA wrapper...")
                lora_wrapper = WanLoraWrapper(model)
                
                for i, lora_config in enumerate(self.config.lora_configs):
                    logger.info(f"🔄 Processing LoRA config {i+1}/{len(self.config.lora_configs)}...")
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    logger.info(f"📍 LoRA path: {lora_path}, strength: {strength}")
                    
                    load_start = time.time()
                    lora_name = lora_wrapper.load_lora(lora_path)
                    load_time = time.time() - load_start
                    logger.info(f"✅ LoRA loaded in {load_time:.1f}s: {lora_name}")
                    
                    apply_start = time.time()
                    lora_wrapper.apply_lora(lora_name, strength)
                    apply_time = time.time() - apply_start
                    logger.info(f"✅ LoRA applied in {apply_time:.1f}s: {lora_name} with strength: {strength}")
            else:
                logger.info("📦 Loading transformer with WanDistillModel (this loads the 31GB model)...")
                
                logger.info(f"📍 Creating WanDistillModel with path: {self.config.model_path}")
                distill_start = time.time()
                model = WanDistillModel(self.config.model_path, self.config, self.init_device)
                distill_time = time.time() - distill_start
                logger.info(f"✅ WanDistillModel created in {distill_time:.1f}s")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 WanDistillRunner.load_transformer() completed in {total_time:.1f}s")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed in WanDistillRunner.load_transformer(): {e}")
            import traceback
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            raise

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = WanStepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)


class MultiDistillModelStruct(MultiModelStruct):
    def __init__(self, model_list, config, boundary_step_index=2):
        self.model = model_list  # [high_noise_model, low_noise_model]
        assert len(self.model) == 2, "MultiModelStruct only supports 2 models now."
        self.config = config
        self.boundary_step_index = boundary_step_index
        self.cur_model_index = -1
        logger.info(f"boundary step index: {self.boundary_step_index}")

    def get_current_model_index(self):
        if self.scheduler.step_index < self.boundary_step_index:
            logger.info(f"using - HIGH - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config.sample_guide_scale[0]
            if self.cur_model_index == -1:
                self.to_cuda(model_index=0)
            elif self.cur_model_index == 1:  # 1 -> 0
                self.offload_cpu(model_index=1)
                self.to_cuda(model_index=0)
            self.cur_model_index = 0
        else:
            logger.info(f"using - LOW - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config.sample_guide_scale[1]
            if self.cur_model_index == -1:
                self.to_cuda(model_index=1)
            elif self.cur_model_index == 0:  # 0 -> 1
                self.offload_cpu(model_index=0)
                self.to_cuda(model_index=1)
            self.cur_model_index = 1


@RUNNER_REGISTER("wan2.2_moe_distill")
class Wan22MoeDistillRunner(WanDistillRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        use_high_lora, use_low_lora = False, False
        if self.config.get("lora_configs") and self.config.lora_configs:
            for lora_config in self.config.lora_configs:
                if lora_config.get("name", "") == "high_noise_model":
                    use_high_lora = True
                elif lora_config.get("name", "") == "low_noise_model":
                    use_low_lora = True

        if use_high_lora:
            high_noise_model = WanModel(
                os.path.join(self.config.model_path, "high_noise_model"),
                self.config,
                self.init_device,
            )
            high_lora_wrapper = WanLoraWrapper(high_noise_model)
            for lora_config in self.config.lora_configs:
                if lora_config.get("name", "") == "high_noise_model":
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_name = high_lora_wrapper.load_lora(lora_path)
                    high_lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"High noise model loaded LoRA: {lora_name} with strength: {strength}")
        else:
            high_noise_model = Wan22MoeDistillModel(
                os.path.join(self.config.model_path, "distill_models", "high_noise_model"),
                self.config,
                self.init_device,
            )

        if use_low_lora:
            low_noise_model = WanModel(
                os.path.join(self.config.model_path, "low_noise_model"),
                self.config,
                self.init_device,
            )
            low_lora_wrapper = WanLoraWrapper(low_noise_model)
            for lora_config in self.config.lora_configs:
                if lora_config.get("name", "") == "low_noise_model":
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_name = low_lora_wrapper.load_lora(lora_path)
                    low_lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Low noise model loaded LoRA: {lora_name} with strength: {strength}")
        else:
            low_noise_model = Wan22MoeDistillModel(
                os.path.join(self.config.model_path, "distill_models", "low_noise_model"),
                self.config,
                self.init_device,
            )

        return MultiDistillModelStruct([high_noise_model, low_noise_model], self.config, self.config.boundary_step_index)

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = Wan22StepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)
