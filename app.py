"""
LTX-Video 2.3 Main Orchestrator
Starts the dedicated FastAPI Backend Service on port 7860 (with autonomous FIFO queue worker)
and launches the Gradio Frontend on port 3000.
"""

import sys
import subprocess

# Self-healing environment check: auto-bootstrap dependencies if missing on container boot
try:
    import requests
    import gradio as gr
    import fastapi
    import uvicorn
    import PIL
    import soundfile
except ImportError:
    print("[Orchestrator] Missing packages detected. Bootstrapping Python environment...")
    try:
        import pip
    except ImportError:
        subprocess.run(["curl", "-sS", "https://bootstrap.pypa.io/get-pip.py", "-o", "/tmp/get-pip.py"], check=True)
        subprocess.run([sys.executable, "/tmp/get-pip.py", "--break-system-packages"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-r", "requirements.txt"], check=True)
    import requests
    import gradio as gr
    import fastapi
    import uvicorn
    import PIL
    import soundfile

import time
import argparse
import threading

from backend import app as backend_app
from frontend import demo as frontend_demo


def run_backend_service(host: str = "127.0.0.1", port: int = 7860):
    """Runs the FastAPI backend service with its background queue worker."""
    print(f"[Orchestrator] Starting LTX Backend Service on http://{host}:{port}...")
    config = uvicorn.Config(backend_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def wait_for_backend(url: str = "http://127.0.0.1:7860/api/health", max_retries: int = 30):
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
    parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link")
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

    from frontend import CSS as frontend_css

    # 3. Launch Frontend on designated port (3000) with graceful fallback for share
    print(f"[Orchestrator] Starting Gradio Frontend on http://{args.host}:{args.port}...")
    try:
        frontend_demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=True,
            show_error=True
        )
    except Exception as exc:
        print(f"[Orchestrator] Notice: share launch failed ({exc}). Falling back to local host launch on port {args.port}...")
        frontend_demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            show_error=True
        )


if __name__ == "__main__":
    main()
