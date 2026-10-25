# LTX-Video 2.3 — Decoupled Backend Service & Gradio Frontend

A decoupled, production-grade architecture splitting **LTX-Video 2.3** into an autonomous backend queue service and a non-blocking Gradio frontend client.

## Why the Previous Queue in `ltx.py` Did Not Work

In the original `ltx.py`:
1. **Generator Lock Starvation**: `Submit_Prompt_To_Queue` ran as a Gradio streaming generator holding `_prompt_worker_lock` in a `while True:` loop inside a single client request. Other concurrent clicks and other user sessions were blocked.
2. **Connection Fragility**: If the user refreshed the browser or the network briefly disconnected, Gradio canceled the generator, causing the worker loop to terminate and drop remaining jobs.
3. **No Real Decoupling**: Model inference, queue storage, and UI rendering were entangled in the same event loop.

## The Decoupled Architecture

```
┌───────────────────────────────────────┐
│       Gradio Frontend (port 3000)     │
│  - Non-blocking prompt submissions    │
│  - Live queue monitor (gr.Timer 1s)   │
│  - Completed video gallery & player   │
└──────────────────┬────────────────────┘
                   │ HTTP REST (JSON / multipart)
┌──────────────────▼────────────────────┐
│      FastAPI Backend (port 7860)      │
│  - Autonomous FIFO queue worker thread│
│  - /api/queue/submit (instant <20ms)  │
│  - /api/queue/status & /api/cancel    │
│  - /api/project/upload_extract & run  │
│  - Video rendering engine & /videos   │
└───────────────────────────────────────┘
```

### Components

- **`backend.py`**:
  - Standalone FastAPI server on port `7860`.
  - Autonomous FIFO background worker daemon running in its own thread, completely detached from client connections.
  - Endpoints:
    - `POST /api/queue/submit`: Enqueues video requests and returns job position instantly.
    - `GET /api/queue/status`: Real-time state of active, queued, and completed jobs.
    - `POST /api/jobs/{id}/cancel`: Immediate job cancellation.
    - `POST /api/project/upload_extract`: Extracts project ZIP and validates `manifest.json`.
    - `POST /api/project/queue_batch`: Enqueues all project pieces into the queue.
    - `GET /api/videos/{filename}`: Video streaming endpoint.
- **`frontend.py`**:
  - Gradio UI on port `3000`.
  - Submissions return in milliseconds, allowing the user to queue multiple videos back-to-back without UI freezing.
  - Periodic polling (`gr.Timer(1.0)`) provides real-time progress bars, ETAs, and updates the completed video selector.
- **`app.py`**:
  - Orchestrates both services seamlessly (`python3 app.py --port 3000`).

## Running the Architecture

```bash
# Start both backend and frontend together:
python3 app.py --port 3000 --host 0.0.0.0

# Or start services separately:
python3 backend.py   # Runs on http://127.0.0.1:7860
python3 frontend.py  # Runs on http://0.0.0.0:3000
```
