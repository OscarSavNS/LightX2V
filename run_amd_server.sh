#!/bin/bash

# AMD GPU-compatible server launcher for LightX2V
# This script sets the proper environment variables to run with ROCm/AMD GPUs

export XFORMERS_FORCE_DISABLE_TRITON=1
export DISABLE_XFORMERS=1
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_PATH=/opt/rocm

# Add debug logging
export AMD_LOG_LEVEL=1
export HIP_PRINT_ENV=1
export PYTHONPATH=$PYTHONPATH:/root/LightX2V

# Add timeout and import optimization
export LIGHTX2V_IMPORT_TIMEOUT=120
export FLASH_ATTN_SKIP_PRECOMPILE=1
export XFORMERS_EFFICIENT_ATTENTION_OVERRIDE=0

# AMD GPU-specific optimizations for large model loading
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=8
export HIP_LAUNCH_BLOCKING=0

echo "=== Starting LightX2V server with AMD GPU support ==="
echo "Environment variables set:"
echo "  XFORMERS_FORCE_DISABLE_TRITON: $XFORMERS_FORCE_DISABLE_TRITON"
echo "  DISABLE_XFORMERS: $DISABLE_XFORMERS"
echo "  AMD_SERIALIZE_KERNEL: $AMD_SERIALIZE_KERNEL"
echo "  HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  AMD_LOG_LEVEL: $AMD_LOG_LEVEL"

echo ""
echo "=== System Diagnostics ==="
echo "ROCm PyTorch detected: $(python3 -c 'import torch; print(torch.__version__)')"
echo "AMD GPUs available: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "ROCm available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU names and memory
echo ""
echo "=== GPU Information ==="
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('No GPUs detected')
"

# Test basic GPU operations
echo ""
echo "=== Testing GPU Operations ==="
python3 -c "
import torch
try:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        print('✓ Basic GPU operations working')
    else:
        print('✗ No GPUs available for testing')
except Exception as e:
    print(f'✗ GPU test failed: {e}')
"

# Check model files with detailed logging
echo ""
echo "=== Model Files Check ==="
echo "⏱️  $(date): Starting model file verification..."
MODEL_PATH="./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/distill_models"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model directory exists: $MODEL_PATH"
    echo "📁 Directory size: $(du -sh $MODEL_PATH | cut -f1)"
    echo "📋 Config file exists: $([ -f "./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/configs/example_config.json" ] && echo "✓" || echo "✗")"
    
    echo "📊 Key model files:"
    if [ -f "$MODEL_PATH/distill_model.safetensors" ]; then
        echo "  ✅ Main DiT model: $(du -sh $MODEL_PATH/distill_model.safetensors | cut -f1) - distill_model.safetensors"
    else
        echo "  ❌ Main DiT model: distill_model.safetensors NOT FOUND"
    fi
    
    if [ -f "$MODEL_PATH/Wan2.1_VAE.pth" ]; then
        echo "  ✅ VAE model: $(du -sh $MODEL_PATH/Wan2.1_VAE.pth | cut -f1) - Wan2.1_VAE.pth"
    else
        echo "  ❌ VAE model: Wan2.1_VAE.pth NOT FOUND"
    fi
    
    if [ -f "$MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth" ]; then
        echo "  ✅ T5 encoder: $(du -sh $MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth | cut -f1) - models_t5_umt5-xxl-enc-bf16.pth"
    else
        echo "  ❌ T5 encoder: models_t5_umt5-xxl-enc-bf16.pth NOT FOUND"
    fi
    
    if [ -f "$MODEL_PATH/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
        echo "  ✅ CLIP encoder: $(du -sh $MODEL_PATH/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth | cut -f1) - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    else
        echo "  ❌ CLIP encoder: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth NOT FOUND"
    fi
    
    echo "📈 Memory requirements estimate:"
    echo "  DiT model: ~31GB GPU memory"
    echo "  T5 encoder: ~11GB GPU memory" 
    echo "  CLIP encoder: ~5GB GPU memory"
    echo "  VAE: ~1GB GPU memory"
    echo "  Total: ~48GB (Available: 256GB across 4x MI250X)"
else
    echo "❌ Model directory not found: $MODEL_PATH"
fi

echo ""
echo "=== Starting Server ==="
echo "⏱️  $(date): Initiating server startup with enhanced monitoring..."

# Function to log system stats
log_system_stats() {
    echo "📊 $(date): System Stats:"
    echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
    if command -v rocm-smi >/dev/null 2>&1; then
        echo "  GPU Memory Usage:"
        rocm-smi --showmeminfo vram | grep -E 'GPU|Used|Total' | head -8 | sed 's/^/    /'
    fi
    echo "  Load: $(uptime | awk -F'load average:' '{print $2}')"
}

# Start server in background to monitor it
echo "🚀 $(date): Starting LightX2V API server..."
log_system_stats

python3 -m lightx2v.api_server \
    --model_path ./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/distill_models \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json ./models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/configs/example_config.json \
    "$@" &

# Monitor startup and log progress
server_pid=$!
echo "🔍 Server PID: $server_pid"
echo "⏱️  $(date): Starting server startup monitoring..."

# Monitor startup progress with extended timeout for 31GB model loading
startup_timeout=900  # 15 minutes for large model loading
elapsed=0
check_interval=30    # Check every 30 seconds

echo "⚠️  Large model loading may take up to 15 minutes..."

while [ $elapsed -lt $startup_timeout ]; do
    if ! kill -0 $server_pid 2>/dev/null; then
        wait $server_pid
        exit_code=$?
        echo "❌ $(date): Server process terminated unexpectedly with exit code $exit_code"
        exit $exit_code
    fi
    
    # Check if server is responding with detailed health check
    health_response=$(curl -s http://localhost:8000/health 2>/dev/null)
    http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    
    if [ "$http_code" = "200" ]; then
        # Parse the health response to check if server is fully ready
        if echo "$health_response" | grep -q '"status":"healthy"'; then
            echo "✅ $(date): Server started successfully after ${elapsed}s"
            echo "🌐 API available at http://0.0.0.0:8000"
            echo "📚 Documentation at http://0.0.0.0:8000/docs"
            echo "🎯 Health status: $health_response"
            
            # Keep server running in foreground
            wait $server_pid
            exit $?
        elif echo "$health_response" | grep -q '"status":"loading"'; then
            echo "🔄 $(date): Server responding but still loading models (${elapsed}s/${startup_timeout}s)"
        else
            echo "🔄 $(date): Server responding but not ready yet (${elapsed}s/${startup_timeout}s)"
        fi
    elif [ "$http_code" = "404" ] || [ "$http_code" = "500" ]; then
        echo "🔄 $(date): Server responding but health endpoint not ready (${elapsed}s/${startup_timeout}s)"
    else
        echo "⏳ $(date): Server startup in progress... (${elapsed}s/${startup_timeout}s)"
    fi
    
    sleep $check_interval
    elapsed=$((elapsed + check_interval))
    log_system_stats
done

echo "❌ $(date): Server startup timed out after ${startup_timeout}s"
echo "💡 This usually indicates model loading issues - check the logs above for details"
echo "🔄 Terminating server process..."
kill -TERM $server_pid 2>/dev/null
sleep 5
kill -KILL $server_pid 2>/dev/null
exit 1
