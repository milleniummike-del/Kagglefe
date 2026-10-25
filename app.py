"""
LTX-Video 2.3 Main Orchestrator
Starts the dedicated FastAPI Backend Service on port 7860 (with autonomous FIFO queue worker)
and launches the Gradio Frontend on port 3000.
"""

import argparse
import threading
import time

import requests
import uvicorn

from backend import app as backend_app
from frontend import demo as frontend_demo


def run_backend_service(host: str = "127.0.0.1", port: int = 7860):
    """Runs the FastAPI backend service with its background queue worker."""
    print(f"[Orchestrator] Starting LTX Backend Service on http://{host}:{port}...")
    config = uvicorn.Config(backend_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def wait_for_backend(url: str = "http://127.0.0.1:7860/api/health", max_retries: int = 25):
    """Waits for the backend service to report healthy before starting the frontend."""
    for _ in range(max_retries):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print("[Orchestrator] Backend Service is ready.")
                return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser(description="LTX-Video 2.3 Decoupled Service & Frontend")
    parser.add_argument("--port", type=int, default=3000, help="Frontend server port (default: 3000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Frontend server host (default: 0.0.0.0)")
    parser.add_argument("--backend-port", type=int, default=7860, help="Backend service port (default: 7860)")
    args, _ = parser.parse_known_args()

    # 1. Start Backend Service in background thread on port 7860
    backend_thread = threading.Thread(
        target=run_backend_service,
        kwargs={"host": "127.0.0.1", "port": args.backend_port},
        daemon=True
    )
    backend_thread.start()

    # 2. Wait for Backend to report ready
    wait_for_backend(f"http://127.0.0.1:{args.backend_port}/api/health")

    # 3. Launch Frontend on designated port (3000)
    print(f"[Orchestrator] Starting Gradio Frontend on http://{args.host}:{args.port}...")
    frontend_demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
