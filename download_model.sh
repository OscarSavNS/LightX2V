#!/bin/bash

# LightX2V Model Download Script
# Downloads video generation models from HuggingFace for use with LightX2V

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DEFAULT_MODEL_DIR="./models"
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"$HOME/.cache/huggingface/hub"}

print_header() {
    echo -e "${BLUE}"
    echo "██╗     ██╗ ██████╗ ██╗  ██╗████████╗██╗  ██╗██████╗ ██╗   ██╗"
    echo "██║     ██║██╔════╝ ██║  ██║╚══██╔══╝╚██╗██╔╝╚════██╗██║   ██║"
    echo "██║     ██║██║  ███╗███████║   ██║    ╚███╔╝  █████╔╝██║   ██║"
    echo "██║     ██║██║   ██║██╔══██║   ██║    ██╔██╗ ██╔═══╝ ╚██╗ ██╔╝"
    echo "███████╗██║╚██████╔╝██║  ██║   ██║   ██╔╝ ██╗███████╗ ╚████╔╝ "
    echo "╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝  ╚═══╝  "
    echo "                    Model Download Script"
    echo -e "${NC}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is required but not installed."
        exit 1
    fi
    
    # Check if huggingface_hub is available
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        log_warn "huggingface_hub not found. Installing..."
        pip install huggingface_hub[cli]
    fi
    
    log_info "Dependencies check completed."
}

show_model_menu() {
    echo -e "${BLUE}Available Models:${NC}"
    echo "1) WAN 2.1 I2V 14B 480P (Standard)"
    echo "2) WAN 2.1 I2V 14B 480P (Distilled - 4 steps, Recommended)"
    echo "3) WAN 2.1 I2V 14B 720P (Standard)"
    echo "4) WAN 2.1 I2V 14B 720P (Distilled - 4 steps)"
    echo "5) WAN 2.1 T2V 14B (Standard)"
    echo "6) WAN 2.1 T2V 14B (Distilled - 4 steps)"
    echo "7) WAN 2.1 T2V 1.3B (Lightweight)"
    echo "8) Hunyuan Video"
    echo "9) CogVideoX 1.5 5B T2V"
    echo "10) Custom HuggingFace model"
    echo ""
}

get_model_info() {
    case $1 in
        1)
            MODEL_REPO="lightx2v/Wan2.1-I2V-14B-480P-Lightx2v"
            MODEL_CLASS="wan2.1"
            TASK="i2v"
            ;;
        2)
            MODEL_REPO="lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v"
            MODEL_CLASS="wan2.1_distill"
            TASK="i2v"
            ;;
        3)
            MODEL_REPO="lightx2v/Wan2.1-I2V-14B-720P-Lightx2v"
            MODEL_CLASS="wan2.1"
            TASK="i2v"
            ;;
        4)
            MODEL_REPO="lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v"
            MODEL_CLASS="wan2.1_distill"
            TASK="i2v"
            ;;
        5)
            MODEL_REPO="lightx2v/Wan2.1-T2V-14B-Lightx2v"
            MODEL_CLASS="wan2.1"
            TASK="t2v"
            ;;
        6)
            MODEL_REPO="lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v"
            MODEL_CLASS="wan2.1_distill"
            TASK="t2v"
            ;;
        7)
            MODEL_REPO="lightx2v/Wan2.1-T2V-1.3B-Lightx2v"
            MODEL_CLASS="wan2.1"
            TASK="t2v"
            ;;
        8)
            MODEL_REPO="tencent/HunyuanVideo"
            MODEL_CLASS="hunyuan_video"
            TASK="t2v"
            ;;
        9)
            MODEL_REPO="THUDM/CogVideoX1.5-5B"
            MODEL_CLASS="cogvideox1.5_5b"
            TASK="t2v"
            ;;
        10)
            echo -n "Enter HuggingFace model repository (e.g., username/model-name): "
            read MODEL_REPO
            echo -n "Enter model class (wan2.1, wan2.1_distill, hunyuan_video, cogvideox1.5_5b): "
            read MODEL_CLASS
            echo -n "Enter task type (i2v, t2v): "
            read TASK
            ;;
        *)
            log_error "Invalid selection"
            exit 1
            ;;
    esac
}

show_precision_menu() {
    echo -e "${BLUE}Select precision/quantization:${NC}"
    echo "1) Original precision (bf16/fp16)"
    echo "2) FP8 quantized (Recommended for memory efficiency)"
    echo "3) INT8 quantized (Maximum memory efficiency)"
    echo "4) Download all versions"
    echo ""
}

download_model() {
    local repo=$1
    local local_dir=$2
    local subfolder=$3
    
    log_info "Downloading from: $repo"
    log_info "Local directory: $local_dir"
    
    if [ ! -z "$subfolder" ]; then
        log_info "Subfolder: $subfolder"
        # Download specific subfolder
        python3 -c "
import os
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id='$repo',
        local_dir='$local_dir',
        allow_patterns='$subfolder/**',
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print('✅ Download completed successfully!')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"
    else
        # Download entire repository
        python3 -c "
import os
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id='$repo',
        local_dir='$local_dir',
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print('✅ Download completed successfully!')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"
    fi
}

generate_config_example() {
    local model_path=$1
    local model_class=$2
    local task=$3
    local precision=$4
    
    local config_dir="$model_path/configs"
    mkdir -p "$config_dir"
    
    local config_file="$config_dir/example_config.json"
    
    cat > "$config_file" << EOF
{
    "infer_steps": $([ "$model_class" == "wan2.1_distill" ] && echo "4" || echo "40"),
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3", 
    "cross_attn_2_type": "flash_attn3",
    "seed": 42,
    "sample_guide_scale": 5,
    "sample_shift": 5,
    "enable_cfg": $([ "$model_class" == "wan2.1_distill" ] && echo "false" || echo "true"),
    "cpu_offload": false
}
EOF
    
    log_info "Example config saved to: $config_file"
}

generate_server_command() {
    local model_path=$1
    local model_class=$2
    local task=$3
    local precision=$4
    
    local command_file="$model_path/run_server.sh"
    
    cat > "$command_file" << EOF
#!/bin/bash

# LightX2V Server Launch Script
# Generated automatically by download_model.sh

# Basic server command
python3 -m lightx2v.api_server \\
    --model_path "$model_path" \\
    --model_cls "$model_class" \\
    --task "$task" \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --config_json "$model_path/configs/example_config.json"

# For multi-GPU setup (uncomment and adjust nproc_per_node):
# python3 -m lightx2v.api_server \\
#     --model_path "$model_path" \\
#     --model_cls "$model_class" \\
#     --task "$task" \\
#     --host 0.0.0.0 \\
#     --port 8000 \\
#     --config_json "$model_path/configs/example_config.json" \\
#     --nproc_per_node 2

EOF
    
    chmod +x "$command_file"
    log_info "Server launch script saved to: $command_file"
}

print_usage_info() {
    local model_path=$1
    local model_class=$2
    local task=$3
    
    echo ""
    echo -e "${GREEN}🎉 Model download completed!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Navigate to the model directory:"
    echo "   cd $model_path"
    echo ""
    echo "2. Run the server using the generated script:"
    echo "   ./run_server.sh"
    echo ""
    echo "3. Or run manually:"
    echo "   python3 -m lightx2v.api_server \\"
    echo "       --model_path $model_path \\"
    echo "       --model_cls $model_class \\"
    echo "       --task $task \\"
    echo "       --host 0.0.0.0 \\"
    echo "       --port 8000 \\"
    echo "       --config_json $model_path/configs/example_config.json"
    echo ""
    echo -e "${BLUE}API Endpoints:${NC}"
    echo "- Create task: POST http://localhost:8000/v1/tasks/"
    echo "- Check status: GET http://localhost:8000/v1/tasks/{task_id}/status"
    echo "- Download result: GET http://localhost:8000/v1/tasks/{task_id}/result"
    echo ""
}

main() {
    print_header
    
    # Parse command line arguments
    MODEL_DIR="$DEFAULT_MODEL_DIR"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [--model-dir PATH]"
                echo "  --model-dir PATH    Directory to download models (default: $DEFAULT_MODEL_DIR)"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_dependencies
    
    # Create model directory
    mkdir -p "$MODEL_DIR"
    
    # Model selection
    show_model_menu
    echo -n "Select model (1-10): "
    read model_choice
    
    get_model_info $model_choice
    
    # Extract model name from repository
    MODEL_NAME=$(basename "$MODEL_REPO")
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    
    log_info "Selected model: $MODEL_REPO"
    log_info "Model class: $MODEL_CLASS"
    log_info "Task type: $TASK"
    log_info "Download path: $MODEL_PATH"
    
    # Precision selection (only for WAN models)
    if [[ "$MODEL_CLASS" == wan2.1* ]]; then
        show_precision_menu
        echo -n "Select precision (1-4): "
        read precision_choice
        
        case $precision_choice in
            1)
                SUBFOLDER="original"
                PRECISION="original"
                ;;
            2)
                if [[ "$MODEL_REPO" == *"StepDistill"* ]]; then
                    SUBFOLDER="distill_fp8"
                else
                    SUBFOLDER="fp8"
                fi
                PRECISION="fp8"
                ;;
            3)
                if [[ "$MODEL_REPO" == *"StepDistill"* ]]; then
                    SUBFOLDER="distill_int8"
                else
                    SUBFOLDER="int8"
                fi
                PRECISION="int8"
                ;;
            4)
                SUBFOLDER=""
                PRECISION="all"
                ;;
            *)
                log_error "Invalid precision selection"
                exit 1
                ;;
        esac
    else
        SUBFOLDER=""
        PRECISION="all"
    fi
    
    # Confirm download
    echo ""
    echo -e "${YELLOW}Download Summary:${NC}"
    echo "Repository: $MODEL_REPO"
    echo "Local path: $MODEL_PATH"
    echo "Precision: $PRECISION"
    echo ""
    echo -n "Proceed with download? (y/N): "
    read confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log_info "Download cancelled."
        exit 0
    fi
    
    # Start download
    log_info "Starting download..."
    download_model "$MODEL_REPO" "$MODEL_PATH" "$SUBFOLDER"
    
    # Generate configuration and scripts
    generate_config_example "$MODEL_PATH" "$MODEL_CLASS" "$TASK" "$PRECISION"
    generate_server_command "$MODEL_PATH" "$MODEL_CLASS" "$TASK" "$PRECISION"
    
    # Print usage information
    print_usage_info "$MODEL_PATH" "$MODEL_CLASS" "$TASK"
}

# Handle interrupt
trap 'log_warn "Download interrupted by user"; exit 1' INT

# Run main function
main "$@"