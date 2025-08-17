import gc

import requests
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger
from requests.exceptions import RequestException

from lightx2v.utils.envs import *
from lightx2v.utils.generate_task_id import generate_task_id
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.utils.utils import cache_video, save_to_video, vae_to_comfyui_image

from .base_runner import BaseRunner


class DefaultRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.has_prompt_enhancer = False
        self.progress_callback = None
        if self.config.task == "t2v" and self.config.get("sub_servers", {}).get("prompt_enhancer") is not None:
            self.has_prompt_enhancer = True
            if not self.check_sub_servers("prompt_enhancer"):
                self.has_prompt_enhancer = False
                logger.warning("No prompt enhancer server available, disable prompt enhancer.")
        if not self.has_prompt_enhancer:
            self.config.use_prompt_enhancer = False
        self.set_init_device()

    def init_modules(self):
        import time
        start_time = time.time()
        
        logger.info("🔧 Initializing runner modules...")
        
        # Log initial memory state
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 Initial GPU memory: {initial_memory:.2f}GB / {total_memory:.1f}GB")
        
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            logger.info("🏗️  Loading model components...")
            model_start = time.time()
            self.load_model()
            model_time = time.time() - model_start
            logger.info(f"✅ Model components loaded in {model_time:.1f}s")
        elif self.config.get("lazy_load", False):
            logger.info("💤 Lazy loading enabled - models will load on demand")
            assert self.config.get("cpu_offload", False)
        
        logger.info("🔗 Setting up inference pipelines...")
        self.run_dit = self._run_dit_local
        self.run_vae_decoder = self._run_vae_decoder_local
        if self.config["task"] == "i2v":
            self.run_input_encoder = self._run_input_encoder_local_i2v
            logger.info("📸 Configured for image-to-video (i2v) pipeline")
        else:
            self.run_input_encoder = self._run_input_encoder_local_t2v
            logger.info("📝 Configured for text-to-video (t2v) pipeline")
        
        # Log final memory state
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            memory_used = final_memory - initial_memory
            logger.info(f"💾 Final GPU memory: {final_memory:.2f}GB / {total_memory:.1f}GB (used: +{memory_used:.2f}GB)")
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Module initialization completed in {total_time:.1f}s")

    def set_init_device(self):
        if self.config.cpu_offload:
            self.init_device = torch.device("cpu")
        else:
            self.init_device = torch.device("cuda")

    def load_vfi_model(self):
        if self.config["video_frame_interpolation"].get("algo", None) == "rife":
            from lightx2v.models.vfi.rife.rife_comfyui_wrapper import RIFEWrapper

            logger.info("Loading RIFE model...")
            return RIFEWrapper(self.config["video_frame_interpolation"]["model_path"])
        else:
            raise ValueError(f"Unsupported VFI model: {self.config['video_frame_interpolation']['algo']}")

    @ProfilingContext("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()
        self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None

    def check_sub_servers(self, task_type):
        urls = self.config.get("sub_servers", {}).get(task_type, [])
        available_servers = []
        for url in urls:
            try:
                status_url = f"{url}/v1/local/{task_type}/generate/service_status"
                response = requests.get(status_url, timeout=2)
                if response.status_code == 200:
                    available_servers.append(url)
                else:
                    logger.warning(f"Service {url} returned status code {response.status_code}")

            except RequestException as e:
                logger.warning(f"Failed to connect to {url}: {str(e)}")
                continue
        logger.info(f"{task_type} available servers: {available_servers}")
        self.config["sub_servers"][task_type] = available_servers
        return len(available_servers) > 0

    def set_inputs(self, inputs):
        self.config["prompt"] = inputs.get("prompt", "")
        self.config["use_prompt_enhancer"] = False
        if self.has_prompt_enhancer:
            self.config["use_prompt_enhancer"] = inputs.get("use_prompt_enhancer", False)  # Reset use_prompt_enhancer from clinet side.
        self.config["negative_prompt"] = inputs.get("negative_prompt", "")
        self.config["image_path"] = inputs.get("image_path", "")
        self.config["save_video_path"] = inputs.get("save_video_path", "")
        self.config["infer_steps"] = inputs.get("infer_steps", self.config.get("infer_steps", 5))
        self.config["target_video_length"] = inputs.get("target_video_length", self.config.get("target_video_length", 81))
        self.config["seed"] = inputs.get("seed", self.config.get("seed", 42))
        self.config["audio_path"] = inputs.get("audio_path", "")  # for wan-audio
        self.config["video_duration"] = inputs.get("video_duration", 5)  # for wan-audio

        # self.config["sample_shift"] = inputs.get("sample_shift", self.config.get("sample_shift", 5))
        # self.config["sample_guide_scale"] = inputs.get("sample_guide_scale", self.config.get("sample_guide_scale", 5))

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def run(self, total_steps=None):
        import time
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        
        logger.info(f"🔄 Starting {total_steps}-step diffusion process...")
        
        for step_index in range(total_steps):
            step_start = time.time()
            progress_percent = ((step_index + 1) / total_steps) * 100
            logger.info(f"🎯 Diffusion step {step_index + 1}/{total_steps} ({progress_percent:.1f}%)")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            step_infer_start = time.time()
            with ProfilingContext4Debug("🚀 infer_main"):
                self.model.infer(self.inputs)
            step_infer_time = time.time() - step_infer_start

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()
            
            step_total_time = time.time() - step_start
            remaining_steps = total_steps - (step_index + 1)
            eta_seconds = remaining_steps * step_total_time
            logger.info(f"✅ Step {step_index + 1} completed in {step_total_time:.1f}s (inference: {step_infer_time:.1f}s, ETA: {eta_seconds:.0f}s)")

            if self.progress_callback:
                self.progress_callback(progress_percent, 100)

        logger.info(f"🎉 All {total_steps} diffusion steps completed!")
        return self.model.scheduler.latents, self.model.scheduler.generator

    def run_step(self):
        self.inputs = self.run_input_encoder()
        self.set_target_shape()
        self.run_dit(total_steps=1)

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            if hasattr(self.model.transformer_infer, "weights_stream_mgr"):
                self.model.transformer_infer.weights_stream_mgr.clear()
            if hasattr(self.model.transformer_weights, "clear"):
                self.model.transformer_weights.clear()
            self.model.pre_weight.clear()
            self.model.post_weight.clear()
            del self.model
        torch.cuda.empty_cache()
        gc.collect()

    @ProfilingContext("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        img = Image.open(self.config["image_path"]).convert("RGB")
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)
        text_encoder_output = self.run_text_encoder(prompt, img)
        torch.cuda.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(clip_encoder_out, vae_encode_out, text_encoder_output, img)

    @ProfilingContext("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        text_encoder_output = self.run_text_encoder(prompt, None)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
        self.init_scheduler()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        if self.config.get("model_cls") == "wan2.2" and self.config["task"] == "i2v":
            self.inputs["image_encoder_output"]["vae_encoder_out"] = None
        latents, generator = self.run(total_steps)
        self.end_run()
        return latents, generator

    @ProfilingContext("Run VAE Decoder")
    def _run_vae_decoder_local(self, latents, generator):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents, generator=generator, config=self.config)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def post_prompt_enhancer(self):
        while True:
            for url in self.config["sub_servers"]["prompt_enhancer"]:
                response = requests.get(f"{url}/v1/local/prompt_enhancer/generate/service_status").json()
                if response["service_status"] == "idle":
                    response = requests.post(
                        f"{url}/v1/local/prompt_enhancer/generate",
                        json={
                            "task_id": generate_task_id(),
                            "prompt": self.config["prompt"],
                        },
                    )
                    enhanced_prompt = response.json()["output"]
                    logger.info(f"Enhanced prompt: {enhanced_prompt}")
                    return enhanced_prompt

    def run_pipeline(self, save_video=True):
        import time
        pipeline_start = time.time()
        
        logger.info(f"🚀 Starting video generation pipeline...")
        logger.info(f"📋 Generation parameters: prompt='{self.config.get('prompt', '')}', steps={self.config.get('infer_steps', 5)}, length={self.config.get('target_video_length', 81)}")
        
        if self.config["use_prompt_enhancer"]:
            logger.info(f"✨ Running prompt enhancer...")
            enhancer_start = time.time()
            self.config["prompt_enhanced"] = self.post_prompt_enhancer()
            enhancer_time = time.time() - enhancer_start
            logger.info(f"✅ Prompt enhancer completed in {enhancer_time:.1f}s")

        logger.info(f"🎭 Running input encoders (CLIP, T5, VAE)...")
        encoder_start = time.time()
        self.inputs = self.run_input_encoder()
        encoder_time = time.time() - encoder_start
        logger.info(f"✅ Input encoders completed in {encoder_time:.1f}s")
        
        logger.info(f"📏 Setting target video dimensions...")
        self.set_target_shape()
        logger.info(f"✅ Target shape set: {self.config.get('target_height', 480)}x{self.config.get('target_width', 832)}")
        
        logger.info(f"🧠 Starting DiT (Diffusion Transformer) inference...")
        dit_start = time.time()
        latents, generator = self.run_dit()
        dit_time = time.time() - dit_start
        logger.info(f"✅ DiT inference completed in {dit_time:.1f}s ({dit_time/60:.1f}min)")

        logger.info(f"🎨 Running VAE decoder to generate video frames...")
        vae_start = time.time()
        images = self.run_vae_decoder(latents, generator)
        vae_time = time.time() - vae_start
        logger.info(f"✅ VAE decoder completed in {vae_time:.1f}s")
        if self.config["model_cls"] != "wan2.2":
            images = vae_to_comfyui_image(images)

        if "video_frame_interpolation" in self.config:
            assert self.vfi_model is not None and self.config["video_frame_interpolation"].get("target_fps", None) is not None
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            images = self.vfi_model.interpolate_frames(
                images,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if save_video:
            if "video_frame_interpolation" in self.config and self.config["video_frame_interpolation"].get("target_fps"):
                fps = self.config["video_frame_interpolation"]["target_fps"]
            else:
                fps = self.config.get("fps", 16)

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Encoding and saving video at {fps} FPS...")
                video_save_start = time.time()

                if self.config["model_cls"] != "wan2.2":
                    save_to_video(images, self.config.save_video_path, fps=fps, method="ffmpeg")  # type: ignore
                else:
                    cache_video(tensor=images, save_file=self.config.save_video_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
                
                video_save_time = time.time() - video_save_start
                logger.info(f"✅ Video encoding completed in {video_save_time:.1f}s")
                logger.info(f"📹 Video saved successfully to: {self.config.save_video_path}")

        # Cleanup and performance summary
        logger.info(f"🧹 Cleaning up GPU memory...")
        del latents, generator
        torch.cuda.empty_cache()
        gc.collect()
        
        total_pipeline_time = time.time() - pipeline_start
        logger.info(f"🎊 PIPELINE COMPLETE! Total time: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f}min)")
        logger.info(f"📊 Performance breakdown: Encoders={encoder_time:.1f}s, DiT={dit_time:.1f}s, VAE={vae_time:.1f}s")

        # Return (images, audio) - audio is None for default runner
        return images, None
