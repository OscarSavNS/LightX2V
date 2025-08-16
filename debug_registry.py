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
    print("=== Registry Factory Import Debugging ===")
    print("Testing the registry factory import that's causing delays...\n")
    
    # Test the registry factory import chain
    registry_imports = [
        'lightx2v.utils',
        'lightx2v.utils.registry_factory',
    ]
    
    for module in registry_imports:
        test_import(module, timeout_seconds=45)
    
    print(f"\n=== Testing Flash Attention Import Components ===")
    
    # Test the components that come before the registry import
    components = [
        'loguru',
        'flash_attn',
    ]
    
    for module in components:
        test_import(module, timeout_seconds=15)
    
    print(f"\n=== Testing if registry import is the issue ===")
    
    # Test importing flash_attn module without the registry
    print("Testing flash_attn module import without registry...")
    start_time = time.time()
    
    try:
        with timeout_import(45):
            exec("""
# Import everything except the registry line
from loguru import logger

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

print("✓ Flash attention imports completed without registry")
""")
        
        elapsed = time.time() - start_time
        print(f"✓ Success without registry import ({elapsed:.2f}s)")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed even without registry ({elapsed:.2f}s): {e}")

if __name__ == "__main__":
    main()