#!/bin/bash

# LightX2V Server Launch Script
# Generated automatically by download_model.sh
# Modified for AMD GPU compatibility

# Set environment variables for AMD GPU support
export XFORMERS_FORCE_DISABLE_TRITON=1
export DISABLE_XFORMERS=1
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_PATH=/opt/rocm

echo "Starting LightX2V server with AMD GPU support..."
echo "ROCm PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not found')"
echo "AMD GPUs available: $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Unable to detect')"

# Basic server command
python3 -m lightx2v.api_server \
    --model_path "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v" \
    --model_cls "wan2.1_distill" \
    --task "i2v" \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/configs/example_config.json"

# For multi-GPU setup (uncomment and adjust nproc_per_node):
# python3 -m lightx2v.api_server \
#     --model_path "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v" \
#     --model_cls "wan2.1_distill" \
#     --task "i2v" \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --config_json "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/configs/example_config.json" \
#     --nproc_per_node 2

