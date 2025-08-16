import sys
from pathlib import Path

print("⏳ main.py: Starting imports...", flush=True)
import time
import_start = time.time()

print("📦 main.py: Importing uvicorn...", flush=True)
uvicorn_start = time.time()
import uvicorn
uvicorn_time = time.time() - uvicorn_start
print(f"✅ main.py: uvicorn imported in {uvicorn_time:.1f}s", flush=True)

print("📦 main.py: Importing loguru...", flush=True)
loguru_start = time.time()
from loguru import logger
loguru_time = time.time() - loguru_start
print(f"✅ main.py: loguru imported in {loguru_time:.1f}s", flush=True)

print("📦 main.py: Importing server_config...", flush=True)
config_start = time.time()
from .config import server_config
config_time = time.time() - config_start
print(f"✅ main.py: server_config imported in {config_time:.1f}s", flush=True)

print("📦 main.py: Importing DistributedInferenceService...", flush=True)
service_start = time.time()
from .service import DistributedInferenceService
service_time = time.time() - service_start
print(f"✅ main.py: DistributedInferenceService imported in {service_time:.1f}s", flush=True)

print("📦 main.py: Importing ApiServer...", flush=True)
api_start = time.time()
from .api import ApiServer
api_time = time.time() - api_start
print(f"✅ main.py: ApiServer imported in {api_time:.1f}s", flush=True)

total_import_time = time.time() - import_start
print(f"🎯 main.py: All imports completed in {total_import_time:.1f}s", flush=True)


def run_server(args):
    import time
    inference_service = None
    try:
        logger.info("🚀 Starting LightX2V server...")
        server_start_time = time.time()

        logger.info("⚙️  Configuring server settings...")
        if hasattr(args, "host") and args.host:
            server_config.host = args.host
        if hasattr(args, "port") and args.port:
            server_config.port = args.port
        logger.info(f"📋 Server config: host={server_config.host}, port={server_config.port}")

        logger.info("✅ Validating server configuration...")
        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")
        logger.info("✅ Server configuration validated")

        logger.info("🔧 Creating distributed inference service...")
        inference_service = DistributedInferenceService()
        logger.info("✅ Distributed inference service created")
        
        logger.info("🚀 Starting distributed inference service (this may take several minutes)...")
        startup_start = time.time()
        if not inference_service.start_distributed_inference(args):
            raise RuntimeError("Failed to start distributed inference service")
        startup_time = time.time() - startup_start
        logger.info(f"✅ Inference service started successfully in {startup_time:.1f}s")

        logger.info("📁 Setting up cache directory...")
        cache_dir = Path(server_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Cache directory: {cache_dir}")

        logger.info("🌐 Creating API server...")
        api_server = ApiServer(max_queue_size=server_config.max_queue_size)
        logger.info("🔗 Initializing API services...")
        api_server.initialize_services(cache_dir, inference_service)
        logger.info("✅ API services initialized")

        logger.info("📱 Getting FastAPI app...")
        app = api_server.get_app()
        logger.info("✅ FastAPI app ready")

        total_startup_time = time.time() - server_start_time
        logger.info(f"🎯 Total server initialization completed in {total_startup_time:.1f}s")
        logger.info(f"🌐 Starting uvicorn server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="info")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        if inference_service:
            inference_service.stop_distributed_inference()
    except Exception as e:
        logger.error(f"Server failed: {e}")
        if inference_service:
            inference_service.stop_distributed_inference()
        sys.exit(1)
