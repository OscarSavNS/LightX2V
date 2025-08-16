#!/usr/bin/env python3

import os
import sys
import time
import signal
import threading
from contextlib import contextmanager

# Set AMD GPU environment variables
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['AMD_SERIALIZE_KERNEL'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['ROCM_PATH'] = '/opt/rocm'

sys.path.append('/root/LightX2V')

@contextmanager
def timeout_import(seconds=30):
    """Context manager to timeout imports"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Import timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_import(module_name, timeout_seconds=30):
    """Test importing a module with timeout"""
    print(f"Testing import: {module_name}", end=" ... ", flush=True)
    start_time = time.time()
    
    try:
        with timeout_import(timeout_seconds):
            # Use exec to import modules properly
            exec(f"import {module_name}")
        
        elapsed = time.time() - start_time
        print(f"✓ Success ({elapsed:.2f}s)")
        return True
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT ({elapsed:.2f}s)")
        return False
        
    except ImportError as e:
        elapsed = time.time() - start_time
        print(f"✗ ImportError ({elapsed:.2f}s): {e}")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error ({elapsed:.2f}s): {e}")
        return False

def main():
    print("=== LightX2V Import Debugging ===")
    print("Testing imports systematically to find hanging import...\n")
    
    # Test basic imports first
    basic_imports = [
        'torch',
        'numpy', 
        'transformers',
        'diffusers',
        'lightx2v',
        'lightx2v.utils',
        'lightx2v.utils.utils',
    ]
    
    print("=== Basic Imports ===")
    for module in basic_imports:
        test_import(module, timeout_seconds=15)
    
    # Test lightx2v submodules
    lightx2v_imports = [
        'lightx2v.common',
        'lightx2v.common.ops',
        'lightx2v.common.ops.attn',
        'lightx2v.common.ops.mm',
        'lightx2v.common.apis',
        'lightx2v.common.models',
    ]
    
    print("\n=== LightX2V Common Modules ===")
    for module in lightx2v_imports:
        test_import(module, timeout_seconds=20)
    
    # Test the problematic inference module
    inference_imports = [
        'lightx2v.infer',
        'lightx2v.models',
        'lightx2v.api_server',
    ]
    
    print("\n=== LightX2V Core Modules (Likely to hang) ===")
    for module in inference_imports:
        test_import(module, timeout_seconds=60)  # Longer timeout for these
    
    # Test vLLM integration
    vllm_imports = [
        'vllm',
        'vllm.model_executor', 
        'vllm.engine',
    ]
    
    print("\n=== vLLM Integration ===")
    for module in vllm_imports:
        test_import(module, timeout_seconds=30)
    
    print("\n=== Import Testing Complete ===")

if __name__ == "__main__":
    main()