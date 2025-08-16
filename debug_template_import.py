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
    print("=== Template Import Debugging ===")
    print("Testing the individual imports in flash_attn.py...\n")
    
    # Test the individual components of flash_attn.py
    components = [
        'loguru',
        'lightx2v.utils.registry_factory',
        'lightx2v.common.ops.attn.template',
    ]
    
    for module in components:
        test_import(module, timeout_seconds=30)
    
    print(f"\n=== Testing Specific Template Import ===")
    print("Testing just the template import that might be the issue...")
    
    start_time = time.time()
    try:
        with timeout_import(45):
            from lightx2v.common.ops.attn.template import AttnWeightTemplate
            print(f"✓ Template import successful")
        elapsed = time.time() - start_time
        print(f"✓ Template direct import ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Template import failed ({elapsed:.2f}s): {e}")

if __name__ == "__main__":
    main()