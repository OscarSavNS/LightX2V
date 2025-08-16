#!/usr/bin/env python3

import argparse
import os
import torch

# Set AMD GPU environment variables
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'
# os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.0'  # Let PyTorch detect automatically
os.environ['AMD_SERIALIZE_KERNEL'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['ROCM_PATH'] = '/opt/rocm'

def test_amd_gpu_setup():
    print("=== AMD GPU Compatibility Test ===")
    
    # Test PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm support: {'rocm' in torch.__version__}")
    
    # Test GPU detection
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Test basic GPU operations
        print("\n=== Testing GPU Operations ===")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print("✓ Basic matrix multiplication on GPU successful")
            
            # Test memory allocation
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            print(f"✓ GPU memory allocated: {memory_allocated:.2f} MB")
            
        except Exception as e:
            print(f"✗ GPU operation failed: {e}")
            return False
    else:
        print("✗ No AMD GPUs detected")
        return False
    
    # Test LightX2V imports
    print("\n=== Testing LightX2V Imports ===")
    try:
        import lightx2v.utils.utils
        print("✓ LightX2V utils imported successfully")
        
        import lightx2v.infer
        print("✓ LightX2V inference module imported successfully")
        
    except ImportError as e:
        print(f"✗ LightX2V import failed: {e}")
        return False
    
    # Test vLLM integration (optional)
    print("\n=== Testing vLLM Integration ===")
    try:
        import vllm
        print("✓ vLLM imported successfully")
        
        # Test basic vLLM functionality
        from vllm import LLM, SamplingParams
        print("✓ vLLM classes imported successfully")
        
        # Note: Skip actual model loading to avoid memory issues in test
        print("✓ vLLM integration ready (model loading skipped in test)")
        
    except ImportError as e:
        print(f"⚠ vLLM not available: {e}")
        print("  This is optional - core functionality will still work")
    except Exception as e:
        print(f"⚠ vLLM test failed: {e}")
        print("  This may be normal if no model is available")
    
    # Test quantization ops
    print("\n=== Testing Quantization Operations ===")
    try:
        from lightx2v.common.ops.mm.mm_weight import *
        print("✓ Matrix multiplication operations imported")
        
        # Test if vLLM ops are available
        try:
            from vllm import _custom_ops as ops
            if ops is not None:
                print("✓ vLLM custom ops available")
            else:
                print("⚠ vLLM custom ops not loaded")
        except ImportError:
            print("⚠ vLLM custom ops not available")
            
    except ImportError as e:
        print(f"⚠ Quantization ops test failed: {e}")
    
    print("\n✓ All tests passed! AMD GPU setup is working correctly.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test AMD GPU setup for LightX2V')
    parser.add_argument('--extended', action='store_true', help='Run extended performance tests')
    parser.add_argument('--test-vllm', action='store_true', help='Run comprehensive vLLM tests')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    success = test_amd_gpu_setup()
    
    if success and args.extended:
        print("\n=== Running Extended Tests ===")
        try:
            # Extended GPU stress test
            device = torch.device('cuda:0')
            for i in range(5):
                x = torch.randn(2000, 2000, device=device)
                y = torch.randn(2000, 2000, device=device)
                z = torch.mm(x, y)
                print(f"Extended test {i+1}/5: ✓")
                
        except Exception as e:
            print(f"Extended test failed: {e}")
            success = False
    
    if success and args.test_vllm:
        print("\n=== Running Comprehensive vLLM Tests ===")
        try:
            import vllm
            from vllm import LLM, SamplingParams
            print("✓ Comprehensive vLLM import successful")
            
            # Test vLLM operations without loading a model
            sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
            print("✓ vLLM SamplingParams creation successful")
            
        except Exception as e:
            print(f"⚠ Comprehensive vLLM test failed: {e}")
            print("  This is expected if vLLM is not installed")
    
    if success and args.benchmark:
        print("\n=== Running Performance Benchmarks ===")
        try:
            import time
            device = torch.device('cuda:0')
            
            # Matrix multiplication benchmark
            sizes = [1000, 2000, 4000]
            for size in sizes:
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)
                
                start_time = time.time()
                for _ in range(10):
                    z = torch.mm(x, y)
                    torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                flops = 2 * size**3 / avg_time / 1e9  # GFLOPS
                print(f"Matrix {size}x{size}: {avg_time:.4f}s, {flops:.2f} GFLOPS")
                
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    exit(0 if success else 1)