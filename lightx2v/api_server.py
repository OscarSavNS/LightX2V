#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Early debug logging
print("🐍 api_server.py: Starting Python module initialization...", flush=True)
import time
start_time = time.time()

sys.path.insert(0, str(Path(__file__).parent.parent))

print("📦 api_server.py: Importing run_server from lightx2v.server.main...", flush=True)
import_start = time.time()
from lightx2v.server.main import run_server
import_time = time.time() - import_start
print(f"✅ api_server.py: Import completed in {import_time:.1f}s", flush=True)


def main():
    total_init_time = time.time() - start_time
    print(f"🎯 api_server.py: Module initialization completed in {total_init_time:.1f}s", flush=True)
    print("⚙️  api_server.py: Starting argument parsing...", flush=True)
    parser = argparse.ArgumentParser(description="Run LightX2V inference server")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_cls", type=str, required=True, help="Model class name")
    parser.add_argument("--config_json", type=str, help="Path to model config JSON file")
    parser.add_argument("--task", type=str, default="i2v", help="Task type (i2v, etc.)")

    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (GPUs to use)")

    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")

    args = parser.parse_args()
    print("📋 api_server.py: Arguments parsed successfully", flush=True)
    print("🚀 api_server.py: Calling run_server...", flush=True)
    
    run_server(args)


if __name__ == "__main__":
    main()
