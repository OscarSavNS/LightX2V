<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> Light Video Generation Inference Framework</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Papers](https://img.shields.io/badge/论文集-中文-99cc2)](https://lightx2v-papers-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ English | [中文](README_zh.md) \]**

</div>

--------------------------------------------------------------------------------

**LightX2V** is an advanced lightweight video generation inference framework engineered to deliver efficient, high-performance video synthesis solutions. This unified platform integrates multiple state-of-the-art video generation techniques, supporting diverse generation tasks including text-to-video (T2V) and image-to-video (I2V). **X2V represents the transformation of different input modalities (X, such as text or images) into video output (V)**.

## 💡 Quick Start

For comprehensive usage instructions, please refer to our documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**


## 🤖 Supported Model Ecosystem

### Official Open-Source Models
- ✅ [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
- ✅ [Wan2.1 & Wan2.2](https://huggingface.co/Wan-AI/)
- ✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)
- ✅ [CogVideoX1.5-5B-T2V](https://huggingface.co/THUDM/CogVideoX1.5-5B)

### Quantized Models
- ✅ [Wan2.1-T2V-1.3B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-Lightx2v)
- ✅ [Wan2.1-T2V-14B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-Lightx2v)
- ✅ [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- ✅ [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)

### Distilled Models (**🚀 Recommended: 4-step inference**)
- ✅ [Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v)
- ✅ [Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)
- ✅ [Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v)

🔔 Follow our [HuggingFace page](https://huggingface.co/lightx2v) for the latest model releases from our team.

### Autoregressive Models
- ✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)

## 🚀 Frontend Interfaces

We provide multiple frontend interface deployment options:

- **🎨 Gradio Interface**: Clean and user-friendly web interface, perfect for quick experience and prototyping
  - 📖 [Gradio Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html)
- **🎯 ComfyUI Interface**: Powerful node-based workflow interface, supporting complex video generation tasks
  - 📖 [ComfyUI Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_comfyui.html)
- **🚀 Windows One-Click Deployment**: Convenient deployment solution designed for Windows users, featuring automatic environment configuration and intelligent parameter optimization
  - 📖 [Windows One-Click Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_local_windows.html)

**💡 Recommended Solutions**:
- **First-time Users**: We recommend the Windows one-click deployment solution
- **Advanced Users**: We recommend the ComfyUI interface for more customization options
- **Quick Experience**: The Gradio interface provides the most intuitive operation experience

## 🚀 Core Features

### 🎯 **Ultimate Performance Optimization**
- **🔥 SOTA Inference Speed**: Achieve **~20x** acceleration via step distillation and system optimization (single GPU)
- **⚡️ Revolutionary 4-Step Distillation**: Compress original 40-50 step inference to just 4 steps without CFG requirements
- **🛠️ Advanced Operator Support**: Integrated with cutting-edge operators including [Sage Attention](https://github.com/thu-ml/SageAttention), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Radial Attention](https://github.com/mit-han-lab/radial-attention), [q8-kernel](https://github.com/KONAKONA666/q8_kernels), [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel), [vllm](https://github.com/vllm-project/vllm)

### 💾 **Resource-Efficient Deployment**
- **💡 Breaking Hardware Barriers**: Run 14B models for 480P/720P video generation with only **8GB VRAM + 16GB RAM**
- **🔧 Intelligent Parameter Offloading**: Advanced disk-CPU-GPU three-tier offloading architecture with phase/block-level granular management
- **⚙️ Comprehensive Quantization**: Support for `w8a8-int8`, `w8a8-fp8`, `w4a4-nvfp4` and other quantization strategies

### 🎨 **Rich Feature Ecosystem**
- **📈 Smart Feature Caching**: Intelligent caching mechanisms to eliminate redundant computations
- **🔄 Parallel Inference**: Multi-GPU parallel processing for enhanced performance
- **📱 Flexible Deployment Options**: Support for Gradio, service deployment, ComfyUI and other deployment methods
- **🎛️ Dynamic Resolution Inference**: Adaptive resolution adjustment for optimal generation quality
- **🎞️ Video Frame Interpolation**: RIFE-based frame interpolation for smooth frame rate enhancement


## 🏆 Performance Benchmarks

For detailed performance metrics and comparisons, please refer to our [benchmark documentation](https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/getting_started/benchmark_source.md).

[Detailed Service Deployment Guide →](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_service.html)

## 📚 Technical Documentation

### 📖 **Method Tutorials**
- [Model Quantization](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/quantization.html) - Comprehensive guide to quantization strategies
- [Feature Caching](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/cache.html) - Intelligent caching mechanisms
- [Attention Mechanisms](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/attention.html) - State-of-the-art attention operators
- [Parameter Offloading](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/offload.html) - Three-tier storage architecture
- [Parallel Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html) - Multi-GPU acceleration strategies
- [Changing Resolution Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/changing_resolution.html) - U-shaped resolution strategy
- [Step Distillation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html) - 4-step inference technology
- [Video Frame Interpolation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/video_frame_interpolation.html) - Base on the RIFE technology

### 🛠️ **Deployment Guides**
- [Low-Resource Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_resource.html) - Optimized 8GB VRAM solutions
- [Low-Latency Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_latency.html) - Ultra-fast inference optimization
- [Gradio Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html) - Web interface setup
- [Service Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_service.html) - Production API service deployment
- [Lora Model Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/lora_deploy.html) - Flexible Lora deployment

## 🔥 AMD GPU Support (MI250 Series)

**LightX2V now supports AMD MI250 GPUs with full feature parity!**

### 🚀 **Quick Start - AMD GPU**

Choose your preferred deployment method:

#### **Option 1: Docker Deployment (Recommended)**
```bash
# Using ROCm-compatible vLLM base image
docker-compose -f docker-compose_amd.yml up -d

# Or build custom AMD image
docker build -f Dockerfile_amd -t lightx2v:amd .
```

#### **Option 2: Automated Setup**
```bash
# Download and run the automated setup script
./setup_amd.sh

# Start the server
./run_amd_server.sh
```

#### **Option 3: Manual Installation**
```bash
# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Install ROCm-compatible vLLM
git clone https://github.com/nscaledev/vllm-project-vllm.git
cd vllm-project-vllm && pip install -e .

# Install LightX2V dependencies
pip install -r requirements_amd.txt

# Set environment variables and run
source ~/.lightx2v_amd_env
./run_amd_server.sh
```

### 🎯 **Features Supported on AMD**
- ✅ **Full vLLM Integration**: Prompt enhancement and quantization operations
- ✅ **MI250 Optimization**: Tested on AMD Instinct MI250X GPUs
- ✅ **ROCm 6.1 Support**: Latest ROCm compatibility
- ✅ **Multi-GPU**: Support for all 4 MI250 GPU units
- ✅ **Docker Ready**: Production-ready containerized deployment
- ✅ **Feature Parity**: Same functionality as CUDA version

### 📋 **System Requirements**
- **Hardware**: AMD MI250, MI250X, or compatible ROCm GPUs
- **Software**: ROCm 6.1+, Python 3.10+
- **Memory**: 16GB+ RAM, 32GB+ VRAM (for 14B models)
- **Docker**: ROCm-compatible base image from nscale

### 🔧 **Performance Notes**
- **Environment Variables**: Optimized for MI250 architecture (gfx90a)
- **Memory Management**: HIP allocator configuration for best performance
- **xformers**: Disabled for compatibility (uses PyTorch native attention)

### 📖 **Detailed Documentation**
For comprehensive AMD GPU setup instructions, troubleshooting, and performance optimization, see: **[README_AMD.md](README_AMD.md)**

### 🐛 **Troubleshooting**
- **GPU Detection**: Verify with `rocm-smi` and `python3 test_amd_gpu.py`
- **Memory Issues**: Adjust `HIP_VISIBLE_DEVICES` and memory settings
- **Performance**: Check ROCm version and GPU architecture compatibility

## 🧾 Contributing Guidelines

We maintain code quality through automated pre-commit hooks to ensure consistent formatting across the project.

> [!TIP]
> **Setup Instructions:**
>
> 1. Install required dependencies:
> ```shell
> pip install ruff pre-commit
> ```
>
> 2. Run before committing:
> ```shell
> pre-commit run --all-files
> ```

We appreciate your contributions to making LightX2V better!

## 🤝 Acknowledgments

We extend our gratitude to all the model repositories and research communities that inspired and contributed to the development of LightX2V. This framework builds upon the collective efforts of the open-source community.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/lightx2v&Timeline)

## ✏️ Citation

If you find LightX2V useful in your research, please consider citing our work:

```bibtex
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```

## 📞 Contact & Support

For questions, suggestions, or support, please feel free to reach out through:
- 🐛 [GitHub Issues](https://github.com/ModelTC/lightx2v/issues) - Bug reports and feature requests
- 💬 [GitHub Discussions](https://github.com/ModelTC/lightx2v/discussions) - Community discussions and Q&A

---

<div align="center">
Built with ❤️ by the LightX2V team
</div>
