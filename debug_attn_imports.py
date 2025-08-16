#!/usr/bin/env python3

import os
import sys
import time
import signal
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
            exec(f"import {module_name}")
        
        elapsed = time.time() - start_time
        print(f"✓ Success ({elapsed:.2f}s)")
        return True, elapsed
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT ({elapsed:.2f}s)")
        return False, elapsed
        
    except ImportError as e:
        elapsed = time.time() - start_time
        print(f"✗ ImportError ({elapsed:.2f}s): {e}")
        return False, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error ({elapsed:.2f}s): {e}")
        return False, elapsed

def main():
    print("=== Attention Module Import Debugging ===")
    print("Testing individual attention modules to find the slow import...\n")
    
    # Test individual attention modules
    attn_modules = [
        'lightx2v.common.ops.attn.flash_attn',
        'lightx2v.common.ops.attn.radial_attn', 
        'lightx2v.common.ops.attn.ring_attn',
        'lightx2v.common.ops.attn.sage_attn',
        'lightx2v.common.ops.attn.sparge_attn',
        'lightx2v.common.ops.attn.torch_sdpa',
        'lightx2v.common.ops.attn.ulysses_attn',
    ]
    
    total_time = 0
    slow_modules = []
    
    for module in attn_modules:
        success, elapsed = test_import(module, timeout_seconds=30)
        total_time += elapsed
        if elapsed > 5.0:  # Flag modules taking more than 5 seconds
            slow_modules.append((module, elapsed))
    
    print(f"\n=== Summary ===")
    print(f"Total import time: {total_time:.2f}s")
    
    if slow_modules:
        print(f"Slow modules (>5s):")
        for module, elapsed in slow_modules:
            print(f"  {module}: {elapsed:.2f}s")
    else:
        print("No particularly slow modules found")
    
    # Test other ops modules
    print(f"\n=== Other Ops Modules ===")
    other_ops = [
        'lightx2v.common.ops.conv',
        'lightx2v.common.ops.mm',
        'lightx2v.common.ops.norm',
        'lightx2v.common.ops.tensor',
    ]
    
    for module in other_ops:
        test_import(module, timeout_seconds=15)

if __name__ == "__main__":
    main()