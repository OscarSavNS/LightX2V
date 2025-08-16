# LightX2V Architecture Overview

This document provides a comprehensive system architecture diagram of the LightX2V video generation inference framework, illustrating the core components, data flow, and optimization techniques.

## System Architecture Diagram

```mermaid
graph TB
    %% Input Layer
    subgraph "Input Layer"
        TextInput[📝 Text Input]
        ImageInput[🖼️ Image Input]
        AudioInput[🎵 Audio Input]
        Config[⚙️ Configuration JSON]
    end

    %% Encoding Layer  
    subgraph "Encoding Layer"
        TextEnc[🔤 Text Encoders<br/>T5-XXL, CLIP, Llama/LLaVA]
        ImageEnc[🖼️ Image Encoders<br/>CLIP Vision, XLM-RoBERTa]
        AudioEnc[🎵 Audio Adapter<br/>Audio-to-Visual Features]
        VAEEnc[🎬 VAE Encoder<br/>Pixel → Latent Space]
    end

    %% Runner Layer (Orchestration)
    subgraph "Runner Layer - Orchestration"
        RunnerFactory[🏭 Runner Factory<br/>RUNNER_REGISTER]
        BaseRunner[🏃 Base Runner]
        
        subgraph "Model Runners"
            WanRunner[🌊 WAN Runner<br/>WAN 2.1/2.2]
            HunyuanRunner[🐉 Hunyuan Runner<br/>Hunyuan Video]
            CogVideoXRunner[🧠 CogVideoX Runner<br/>CogVideoX 1.5]
            AudioRunner[🎵 Audio Runner<br/>Audio-driven]
            DistillRunner[⚡ Distill Runner<br/>4-step inference]
        end
    end

    %% Network Layer
    subgraph "Network Layer - Core Models"
        subgraph "WAN Networks"
            WanModel[🌊 WAN Model<br/>Transformer + Pre/Post]
            WanAudio[🎵 Audio Adapter]
            WanDistill[⚡ Distill Model]
            WanLora[🔧 LoRA Adapter]
        end
        
        subgraph "Hunyuan Networks"
            HunyuanModel[🐉 Hunyuan Model<br/>3D Causal Transformer]
            HunyuanCache[💾 Feature Caching]
        end
        
        subgraph "CogVideoX Networks"
            CogModel[🧠 CogVideoX Model<br/>T5 + Transformer]
        end
    end

    %% Scheduler Layer
    subgraph "Scheduler Layer - Denoising Control"
        BaseScheduler[📅 Base Scheduler]
        
        subgraph "Scheduling Strategies"
            WanScheduler[🌊 WAN Scheduler<br/>DDPM/DDIM]
            HunyuanScheduler[🐉 Hunyuan Scheduler<br/>EulerA/DPM++]
            CogScheduler[🧠 CogVideoX Scheduler<br/>DDIM]
            CacheScheduler[💾 Cache Scheduler<br/>TeaCache/AdaCache]
            DistillScheduler[⚡ Distill Scheduler<br/>4-step]
        end
    end

    %% Optimization Layer
    subgraph "Optimization Layer"
        subgraph "Attention Optimizations"
            FlashAttn[⚡ Flash Attention<br/>Memory Efficient]
            RingAttn[🔄 Ring Attention<br/>Distributed]
            SageAttn[🧠 SAGE Attention<br/>Sparse Patterns]
            SpargeAttn[✨ Sparge Attention<br/>Custom Sparse]
        end
        
        subgraph "Memory Management"
            CPUOffload[💾 CPU Offloading<br/>Weight Streaming]
            DiskOffload[💽 Disk Offloading<br/>Lazy Loading]
            Quantization[🔢 Quantization<br/>INT8/FP8/NvFP4]
        end
        
        subgraph "Caching Systems"
            TeaCache[🍃 TeaCache<br/>Token Similarity]
            AdaCache[🎯 AdaCache<br/>Adaptive Features]
            CustomCache[🔧 Custom Cache<br/>Block-level]
        end
        
        subgraph "Parallel Processing"
            Ulysses[🌊 Ulysses<br/>Sequence Parallel]
            Ring[🔄 Ring<br/>Attention Parallel]
            CFGParallel[🔀 CFG Parallel<br/>Conditional/Unconditional]
        end
    end

    %% Core Processing
    subgraph "Core Processing Pipeline"
        LatentPrep[🎯 Latent Preparation<br/>Noise + Conditioning]
        
        subgraph "Iterative Denoising"
            Step1[📍 Step 1<br/>DiT Forward]
            Step2[📍 Step 2<br/>DiT Forward]
            StepN[📍 Step N<br/>DiT Forward]
        end
        
        NoisePredict[🎯 Noise Prediction<br/>ε-prediction]
        LatentUpdate[🔄 Latent Update<br/>Scheduler Step]
    end

    %% Output Layer
    subgraph "Output Layer"
        VAEDec[🎬 VAE Decoder<br/>Latent → Pixel Space]
        FrameInterp[🎞️ Frame Interpolation<br/>RIFE (Optional)]
        VideoSave[💾 Video Output<br/>MP4 Generation]
    end

    %% Server Infrastructure
    subgraph "Server Infrastructure (Optional)"
        APIServer[🌐 API Server<br/>FastAPI REST]
        TaskManager[📋 Task Manager<br/>Queue Processing]
        GPUManager[🖥️ GPU Manager<br/>Resource Allocation]
        MultiServer[🔀 Multi-Server<br/>Load Balancing]
    end

    %% Data Flow Connections
    TextInput --> TextEnc
    ImageInput --> ImageEnc
    AudioInput --> AudioEnc
    ImageInput --> VAEEnc
    Config --> RunnerFactory
    
    TextEnc --> RunnerFactory
    ImageEnc --> RunnerFactory
    AudioEnc --> RunnerFactory
    VAEEnc --> RunnerFactory
    
    RunnerFactory --> BaseRunner
    BaseRunner --> WanRunner
    BaseRunner --> HunyuanRunner  
    BaseRunner --> CogVideoXRunner
    BaseRunner --> AudioRunner
    BaseRunner --> DistillRunner
    
    WanRunner --> WanModel
    WanRunner --> WanAudio
    WanRunner --> WanDistill
    WanRunner --> WanLora
    
    HunyuanRunner --> HunyuanModel
    HunyuanRunner --> HunyuanCache
    
    CogVideoXRunner --> CogModel
    
    WanModel --> WanScheduler
    HunyuanModel --> HunyuanScheduler
    CogModel --> CogScheduler
    WanDistill --> DistillScheduler
    
    BaseScheduler --> WanScheduler
    BaseScheduler --> HunyuanScheduler
    BaseScheduler --> CogScheduler
    BaseScheduler --> CacheScheduler
    BaseScheduler --> DistillScheduler
    
    WanScheduler --> LatentPrep
    HunyuanScheduler --> LatentPrep
    CogScheduler --> LatentPrep
    
    LatentPrep --> Step1
    Step1 --> Step2
    Step2 --> StepN
    StepN --> NoisePredict
    NoisePredict --> LatentUpdate
    LatentUpdate --> Step1
    
    %% Optimization Integration
    Step1 -.-> FlashAttn
    Step1 -.-> RingAttn
    Step1 -.-> SageAttn
    Step1 -.-> SpargeAttn
    
    Step2 -.-> TeaCache
    Step2 -.-> AdaCache
    Step2 -.-> CustomCache
    
    WanModel -.-> CPUOffload
    WanModel -.-> DiskOffload
    WanModel -.-> Quantization
    
    Step1 -.-> Ulysses
    Step1 -.-> Ring
    Step1 -.-> CFGParallel
    
    StepN --> VAEDec
    VAEDec --> FrameInterp
    FrameInterp --> VideoSave
    
    %% Server connections
    RunnerFactory -.-> APIServer
    APIServer --> TaskManager
    TaskManager --> GPUManager
    APIServer --> MultiServer

    %% Styling
    classDef inputClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef encodingClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef runnerClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef networkClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef schedulerClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef optimizationClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef processingClass fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef outputClass fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    classDef serverClass fill:#fafafa,stroke:#424242,stroke-width:2px

    class TextInput,ImageInput,AudioInput,Config inputClass
    class TextEnc,ImageEnc,AudioEnc,VAEEnc encodingClass
    class RunnerFactory,BaseRunner,WanRunner,HunyuanRunner,CogVideoXRunner,AudioRunner,DistillRunner runnerClass
    class WanModel,WanAudio,WanDistill,WanLora,HunyuanModel,HunyuanCache,CogModel networkClass
    class BaseScheduler,WanScheduler,HunyuanScheduler,CogScheduler,CacheScheduler,DistillScheduler schedulerClass
    class FlashAttn,RingAttn,SageAttn,SpargeAttn,CPUOffload,DiskOffload,Quantization,TeaCache,AdaCache,CustomCache,Ulysses,Ring,CFGParallel optimizationClass
    class LatentPrep,Step1,Step2,StepN,NoisePredict,LatentUpdate processingClass
    class VAEDec,FrameInterp,VideoSave outputClass
    class APIServer,TaskManager,GPUManager,MultiServer serverClass
```

## Architecture Components Overview

### 🎯 **Input Processing**
- **Multi-modal Inputs**: Text prompts, reference images, audio tracks
- **Configuration-driven**: JSON configs control all aspects of generation
- **Adaptive Resolution**: Dynamic resolution handling based on model capabilities

### 🔄 **Orchestration Layer (Runners)**
- **Model Agnostic**: Unified interface for different model architectures
- **Factory Pattern**: Dynamic runner selection based on model class
- **Lifecycle Management**: Handles initialization, execution, and cleanup

### 🧠 **Neural Networks**
- **Transformer-based**: DiT (Diffusion Transformer) architecture
- **Modular Design**: Separate pre/post processing and core transformer
- **Specialized Variants**: Audio adaptation, distillation, LoRA fine-tuning

### 📅 **Scheduling System**
- **Flexible Denoising**: Multiple sampling algorithms (DDPM, DDIM, Euler, DPM++)
- **Step Control**: From 4-step distilled to 50-step high-quality
- **Caching Integration**: Feature reuse across denoising steps

### ⚡ **Optimization Engine**
- **Memory Efficiency**: Advanced attention patterns and quantization
- **Parallel Processing**: Multi-GPU and distributed inference
- **Smart Caching**: Eliminates redundant computations
- **Dynamic Offloading**: CPU/disk storage for memory management

### 🎬 **Output Generation**
- **Progressive Decoding**: Memory-efficient VAE decoding
- **Frame Enhancement**: Optional RIFE-based interpolation
- **Format Flexibility**: Support for various video formats and resolutions

## Key Data Flow Patterns

### 1. **Standard T2V/I2V Flow**
```
Input → Encoding → Runner Selection → Network Processing → Denoising Loop → VAE Decoding → Video Output
```

### 2. **Optimized Flow with Caching**
```
Input → Encoding → [Cache Check] → Partial Processing → [Cache Store] → Output
```

### 3. **Distributed Flow**
```
Input → [Parallel Encoding] → [Ring/Ulysses Attention] → [Synchronized Denoising] → Output
```

### 4. **Memory-Constrained Flow**
```
Input → [CPU Offload] → [Quantized Processing] → [Disk Streaming] → Output
```

## Performance Optimizations

### 🚀 **Speed Optimizations**
- **Flash Attention**: Up to 8x attention speedup
- **Step Distillation**: 40-50 steps → 4 steps (~10x faster)
- **Feature Caching**: Skip redundant computations
- **Parallel Processing**: Multi-GPU acceleration

### 💾 **Memory Optimizations**
- **Quantization**: INT8/FP8 reduces memory by 50-75%
- **CPU Offloading**: Run 14B models on 8GB VRAM
- **Disk Streaming**: Handle models larger than system memory
- **Gradient Checkpointing**: Trade compute for memory

### 🎯 **Quality Optimizations**
- **Adaptive Resolution**: U-shaped resolution strategy
- **CFG Optimization**: Parallel conditional/unconditional inference
- **LoRA Fine-tuning**: Task-specific model adaptation
- **Frame Interpolation**: Smooth motion enhancement

## Model Architecture Details

### WAN Models (WAN 2.1/2.2)
- **Architecture**: Diffusion Transformer (DiT) with 3D convolutions
- **Text Encoder**: T5-XXL (UMT5) for text understanding
- **Image Encoder**: CLIP XLM-RoBERTa + ViT-Huge for visual features
- **VAE**: Custom 3D causal VAE for video encoding/decoding
- **Special Features**: 
  - LoRA adaptation for fine-tuning
  - Audio adapter for audio-driven generation
  - MoE (Mixture of Experts) variants for efficiency
  - Step distillation for 4-step inference

### Hunyuan Video Models
- **Architecture**: 3D causal transformer with attention masking
- **Text Encoders**: Dual system with Llama/LLaVA + CLIP
- **VAE**: 3D causal autoencoder optimized for video
- **Resolution Support**: Dynamic buckets (360p/540p/720p)
- **Special Features**:
  - Feature caching for acceleration
  - Distributed inference support
  - Dynamic resolution adaptation

### CogVideoX Models
- **Architecture**: DiT with specialized temporal attention
- **Text Encoder**: T5 v1.1 XXL for text processing
- **VAE**: CogVideoX-specific VAE architecture
- **Limitations**: Currently supports T2V only (I2V not implemented)

## File Structure Mapping

```
lightx2v/
├── infer.py                    # Main inference entry point
├── models/
│   ├── runners/               # Orchestration layer
│   │   ├── base_runner.py     # Base runner interface
│   │   ├── wan/               # WAN model runners
│   │   ├── hunyuan/           # Hunyuan model runners
│   │   └── cogvideox/         # CogVideoX model runners
│   ├── networks/              # Neural network implementations
│   │   ├── wan/               # WAN architectures
│   │   ├── hunyuan/           # Hunyuan architectures
│   │   └── cogvideox/         # CogVideoX architectures
│   ├── schedulers/            # Denoising schedulers
│   │   ├── scheduler.py       # Base scheduler
│   │   ├── wan/               # WAN schedulers
│   │   ├── hunyuan/           # Hunyuan schedulers
│   │   └── cogvideox/         # CogVideoX schedulers
│   └── input_encoders/        # Text/image/audio encoders
├── common/
│   ├── apis/                  # Service APIs for distributed inference
│   ├── ops/                   # Optimized operators
│   │   ├── attn/             # Attention implementations
│   │   ├── conv/             # Convolution optimizations
│   │   ├── mm/               # Matrix multiplication
│   │   └── norm/             # Normalization layers
│   └── offload/              # Memory management
└── server/                    # Production server infrastructure
```

This architecture enables LightX2V to achieve state-of-the-art video generation performance while maintaining efficiency and scalability across diverse hardware configurations.