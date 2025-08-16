#!/bin/bash

# LightX2V AMD GPU Setup Script
# Automated installation for AMD MI250 series GPUs with ROCm support
# 
# Usage: ./setup_amd.sh [--skip-pytorch] [--skip-vllm] [--help]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROCM_VERSION="6.1"
PYTORCH_VERSION="2.6.0+rocm6.1"
TORCH_VISION_VERSION="0.21.0+rocm6.1"
TORCH_AUDIO_VERSION="2.6.0+rocm6.1"
VLLM_REPO="https://github.com/nscaledev/vllm-project-vllm.git"

# Parse command line arguments
SKIP_PYTORCH=false
SKIP_VLLM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pytorch)
            SKIP_PYTORCH=true
            shift
            ;;
        --skip-vllm)
            SKIP_VLLM=true
            shift
            ;;
        --help)
            echo "LightX2V AMD GPU Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-pytorch    Skip PyTorch installation (if already installed)"
            echo "  --skip-vllm       Skip vLLM installation (if already installed)"
            echo "  --help           Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Verify ROCm installation"
            echo "  2. Install ROCm-compatible PyTorch"
            echo "  3. Install ROCm-compatible vLLM"
            echo "  4. Install LightX2V dependencies"
            echo "  5. Set up environment variables"
            echo "  6. Test GPU functionality"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LightX2V AMD GPU Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. Consider running as a regular user."
fi

# Step 1: Verify ROCm installation
print_status "Checking ROCm installation..."
if ! command -v rocm-smi &> /dev/null; then
    print_error "ROCm not found. Please install ROCm $ROCM_VERSION first."
    echo "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Check AMD GPUs
GPU_COUNT=$(rocm-smi --showid | grep "GPU" | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    print_error "No AMD GPUs detected. Please check your hardware setup."
    exit 1
fi

print_status "Found $GPU_COUNT AMD GPU(s)"
rocm-smi --showproductname | head -5

# Step 2: Install ROCm PyTorch
if [ "$SKIP_PYTORCH" = false ]; then
    print_status "Installing ROCm-compatible PyTorch..."
    
    # Remove existing CUDA PyTorch if present
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "cu"; then
        print_warning "Removing existing CUDA PyTorch..."
        pip uninstall torch torchvision torchaudio -y
    fi
    
    # Install ROCm PyTorch
    pip install torch==$PYTORCH_VERSION torchvision==$TORCH_VISION_VERSION torchaudio==$TORCH_AUDIO_VERSION \
        --index-url https://download.pytorch.org/whl/rocm$ROCM_VERSION
    
    print_status "PyTorch installation completed"
else
    print_status "Skipping PyTorch installation"
fi

# Verify PyTorch installation
if ! python3 -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__} with {torch.cuda.device_count()} AMD GPUs')" 2>/dev/null; then
    print_error "PyTorch AMD GPU support verification failed"
    exit 1
fi

# Step 3: Install ROCm-compatible vLLM
if [ "$SKIP_VLLM" = false ]; then
    print_status "Installing ROCm-compatible vLLM..."
    
    # Remove existing vLLM if present
    pip uninstall vllm -y 2>/dev/null || true
    
    # Clone and install ROCm vLLM
    if [ -d "vllm-project-vllm" ]; then
        print_warning "vLLM directory exists, removing..."
        rm -rf vllm-project-vllm
    fi
    
    git clone $VLLM_REPO
    cd vllm-project-vllm
    
    # Install vLLM
    pip install -e .
    cd ..
    
    print_status "vLLM installation completed"
else
    print_status "Skipping vLLM installation"
fi

# Step 4: Install LightX2V dependencies
print_status "Installing LightX2V dependencies..."
pip install -r requirements_amd.txt

# Step 5: Set up environment variables
print_status "Setting up environment variables..."

# Create environment setup script
cat > ~/.lightx2v_amd_env << 'EOF'
# LightX2V AMD GPU Environment Variables
export XFORMERS_FORCE_DISABLE_TRITON=1
export DISABLE_XFORMERS=1
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_PATH=/opt/rocm
EOF

# Add to bashrc if not already present
if ! grep -q "lightx2v_amd_env" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# LightX2V AMD GPU Environment" >> ~/.bashrc
    echo "source ~/.lightx2v_amd_env" >> ~/.bashrc
    print_status "Environment variables added to ~/.bashrc"
fi

# Source the environment for current session
source ~/.lightx2v_amd_env

# Step 6: Test GPU functionality
print_status "Testing AMD GPU functionality..."

# Make test script executable
chmod +x test_amd_gpu.py

# Run comprehensive test
if python3 test_amd_gpu.py; then
    echo ""
    print_status "✅ AMD GPU setup completed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Test the server: ./run_amd_server.sh"
    echo "3. Check API endpoints: http://localhost:8000/docs"
    echo ""
    echo -e "${BLUE}GPU Information:${NC}"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'AMD GPUs: {torch.cuda.device_count()}'); [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
else
    print_error "GPU functionality test failed. Please check the output above."
    exit 1
fi

echo ""
print_status "Setup completed! 🚀"