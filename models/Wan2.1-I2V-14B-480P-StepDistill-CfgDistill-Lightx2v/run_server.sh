#!/bin/bash

# LightX2V Server Launch Script
# Generated automatically by download_model.sh

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

