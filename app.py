"""
LTX-Video 2.3 Gradio Studio with Live Multi-Item Video Generation Queue.
Features sequential background queue worker, live real-time queue tracker,
multi-job queueing without UI blocking, and realistic MP4 video generation via FFmpeg.
"""

import os
import time
import math
import uuid
import queue
import random
import argparse
import tempfile
import threading
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import gradio as gr

# Ensure sample assets directory exists
SAMPLE_DIR = Path("sample_assets")
SAMPLE_DIR.mkdir(exist_ok=True)

# Sample images and audio
ASSET_DIR = Path("sample_assets/images")
IMG_NEBULA = str(ASSET_DIR / "ltx_space_nebula_1792873970962.jpg")
IMG_PLANET = str(ASSET_DIR / "ltx_crystal_planet_1792873971765.jpg")
IMG_CYBER_DAY = str(ASSET_DIR / "ltx_cyber_alley_day_1792873972160.jpg")
IMG_CYBER_NIGHT = str(ASSET_DIR / "ltx_cyber_alley_neon_1792873972764.jpg")

AUDIO_COSMIC = str(SAMPLE_DIR / "cosmic_synth.wav")
AUDIO_CYBER = str(SAMPLE_DIR / "cyber_rain.wav")
AUDIO_BASS = str(SAMPLE_DIR / "cinematic_bass.wav")

# -------------------------------------------------------------
# Background FIFO Job Queue & State Management
# -------------------------------------------------------------
job_queue = queue.Queue()
all_jobs = []  # In-memory list of all jobs
jobs_lock = threading.Lock()
current_active_job = None
stop_worker = False
latest_completed_job_id = None


def render_mock_video_file(job: dict) -> str:
    """
    Simulates LTX-Video 2.3 pipeline steps and generates a real MP4 video with FFmpeg.
    """
    params = job["params"]
    fps = int(params.get("fps", 24))
    steps = int(params.get("steps", 30))
    duration = float(params.get("duration", 4.0))
    total_frames = max(12, int(duration * fps))
    aspect_ratio = params.get("aspect_ratio", "16:9")
    motion_strength = float(params.get("motion_strength", 0.75))
    guidance_scale = float(params.get("guidance_scale", 3.5))

    dims_map = {
        "16:9": (854, 480),
        "9:16": (480, 854),
        "1:1": (640, 640),
        "4:3": (720, 540),
    }
    width, height = dims_map.get(aspect_ratio, (854, 480))

    # Stage 1: Audio Analysis
    with jobs_lock:
        job["stage"] = "Analyzing audio Mel-spectrogram & rhythm markers"
        job["progress"] = 5
        job["eta_seconds"] = int(steps * 0.25 + 4)
    time.sleep(0.8)

    # Stage 2: Latent Encoding
    with jobs_lock:
        job["stage"] = "LTX 3D-VAE: Encoding Start & End Frame latents"
        job["progress"] = 18
        job["eta_seconds"] = int(steps * 0.25 + 3)
    time.sleep(0.8)

    # Stage 3: Spatial-Temporal DiT Denoising Diffusion Steps
    for step in range(1, steps + 1):
        if job.get("cancelled", False):
            raise Exception("Job cancelled by user")
        step_progress = int(20 + 62 * (step / steps))
        remaining_steps = steps - step
        eta = max(1, int(remaining_steps * 0.25 + 2))
        with jobs_lock:
            job["stage"] = f"LTX 2.3 DiT Diffusion: Step {step}/{steps} (CFG={guidance_scale:.1f})"
            job["progress"] = step_progress
            job["current_step"] = step
            job["eta_seconds"] = eta
        time.sleep(0.22)

    # Stage 4: Temporal Decoding
    with jobs_lock:
        job["stage"] = "Temporal Latent Expansion & Flow Interpolation"
        job["progress"] = 86
        job["eta_seconds"] = 2
    time.sleep(0.7)

    # Stage 5: Frame Synthesis & Video Encoding
    with jobs_lock:
        job["stage"] = "Multiplexing video frames with audio track"
        job["progress"] = 92
        job["eta_seconds"] = 1

    # Load input images
    def load_or_create(img_path, default_color):
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                return img.resize((width, height), Image.Resampling.LANCZOS)
            except Exception:
                pass
        img = Image.new("RGB", (width, height), color=default_color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(200, 200, 200), width=3)
        return img

    img1 = load_or_create(params.get("image1_path"), (20, 30, 50))
    img2 = load_or_create(params.get("image2_path"), (50, 25, 40))

    temp_dir = tempfile.mkdtemp(prefix="ltx_queue_")
    frame_pattern = os.path.join(temp_dir, "frame_%04d.png")

    for frame_idx in range(total_frames):
        t = frame_idx / float(total_frames - 1) if total_frames > 1 else 0.0
        smooth_t = 0.5 - 0.5 * math.cos(t * math.pi)
        blended = Image.blend(img1, img2, smooth_t)

        zoom_factor = 1.0 + 0.12 * math.sin(t * math.pi) * motion_strength
        if zoom_factor > 1.01:
            crop_w = int(width / zoom_factor)
            crop_h = int(height / zoom_factor)
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            cropped = blended.crop((left, top, left + crop_w, top + crop_h))
            blended = cropped.resize((width, height), Image.Resampling.BILINEAR)

        # Subtle lens flare
        draw = ImageDraw.Draw(blended, "RGBA")
        flare_x = int(width * (0.2 + 0.6 * t))
        flare_y = int(height * 0.3)
        flare_radius = int(80 + 30 * math.sin(t * math.pi * 4))
        draw.ellipse(
            [(flare_x - flare_radius, flare_y - flare_radius),
             (flare_x + flare_radius, flare_y + flare_radius)],
            fill=(255, 180, 80, 35)
        )

        frame_file = frame_pattern % frame_idx
        blended.save(frame_file, "PNG")

    output_mp4 = os.path.join(temp_dir, f"ltx_{job['id']}.mp4")
    audio_path = params.get("audio_path")

    if audio_path and os.path.exists(audio_path):
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", audio_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_mp4
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_mp4
        ]

    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_mp4


def queue_worker_loop():
    """
    Dedicated background worker thread that picks up jobs sequentially (FIFO).
    Each job waits until the preceding job is fully complete.
    """
    global current_active_job, latest_completed_job_id
    while not stop_worker:
        try:
            job = job_queue.get(timeout=0.8)
        except queue.Empty:
            with jobs_lock:
                current_active_job = None
            continue

        if job.get("cancelled", False):
            with jobs_lock:
                job["status"] = "Cancelled"
                job["stage"] = "Cancelled before processing"
            job_queue.task_done()
            continue

        with jobs_lock:
            job["status"] = "Processing"
            job["start_time"] = time.time()
            current_active_job = job

        try:
            video_path = render_mock_video_file(job)
            elapsed = round(time.time() - job["start_time"], 1)
            with jobs_lock:
                job["status"] = "Completed"
                job["stage"] = "Completed successfully"
                job["progress"] = 100
                job["eta_seconds"] = 0
                job["video_path"] = video_path
                job["elapsed_seconds"] = elapsed
                job["telemetry"] = {
                    "job_id": job["id"],
                    "model": "LTX-Video 2.3 (Lightricks DiT)",
                    "resolution": f"{job['params'].get('aspect_ratio')}",
                    "duration_seconds": job["params"].get("duration"),
                    "fps": job["params"].get("fps"),
                    "steps": job["params"].get("steps"),
                    "seed": job["params"].get("seed"),
                    "guidance_scale": job["params"].get("guidance_scale"),
                    "motion_strength": job["params"].get("motion_strength"),
                    "audio_sync": job["params"].get("audio_sync_mode") if job["params"].get("audio_path") else "No Audio",
                    "elapsed_time": f"{elapsed}s",
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                }
                latest_completed_job_id = job["id"]
        except Exception as e:
            with jobs_lock:
                job["status"] = "Failed"
                job["stage"] = f"Error: {str(e)}"
                job["progress"] = 0
                job["eta_seconds"] = 0

        with jobs_lock:
            current_active_job = None
        job_queue.task_done()


# Start background worker daemon
worker_thread = threading.Thread(target=queue_worker_loop, daemon=True)
worker_thread.start()


# -------------------------------------------------------------
# Gradio Actions: Enqueue Job & Queue Polling
# -------------------------------------------------------------
def add_to_queue(
    prompt: str,
    negative_prompt: str,
    image1_path: str,
    image2_path: str,
    audio_path: str,
    conditioning_mode: str,
    audio_sync_mode: str,
    duration: float,
    aspect_ratio: str,
    fps: int,
    steps: int,
    guidance_scale: float,
    motion_strength: float,
    seed: int,
    randomize_seed: bool
):
    """
    Instantly enqueues a new video generation job into the sequential queue.
    Does NOT block the UI—returns immediately so the user can queue more!
    """
    if randomize_seed or seed is None:
        seed = random.randint(100000, 999999)

    job_id = str(uuid.uuid4())[:8]
    prompt_clean = (prompt or "Cinematic scene").strip()

    job_data = {
        "id": job_id,
        "created_at": time.time(),
        "status": "Queued",
        "stage": "Waiting in queue...",
        "progress": 0,
        "current_step": 0,
        "total_steps": int(steps),
        "eta_seconds": 0,
        "cancelled": False,
        "params": {
            "prompt": prompt_clean,
            "negative_prompt": negative_prompt,
            "image1_path": image1_path,
            "image2_path": image2_path,
            "audio_path": audio_path,
            "conditioning_mode": conditioning_mode,
            "audio_sync_mode": audio_sync_mode,
            "duration": float(duration),
            "aspect_ratio": aspect_ratio,
            "fps": int(fps),
            "steps": int(steps),
            "guidance_scale": float(guidance_scale),
            "motion_strength": float(motion_strength),
            "seed": seed,
        },
        "video_path": None,
        "telemetry": None,
    }

    with jobs_lock:
        all_jobs.append(job_data)
        queue_pos = sum(1 for j in all_jobs if j["status"] in ["Queued", "Processing"])

    job_queue.put(job_data)

    msg = f"✅ **Job #{job_id} Enqueued!** (Position in Queue: **#{queue_pos}**)\nThis generation will automatically run once earlier queue items finish. You can queue more videos below!"
    return msg


def cancel_active_or_queued():
    """Cancels the currently running job or removes queued items."""
    with jobs_lock:
        if current_active_job:
            current_active_job["cancelled"] = True
            return f"Requested cancellation for active Job #{current_active_job['id']}."
        # Otherwise cancel last queued job
        for j in reversed(all_jobs):
            if j["status"] == "Queued":
                j["cancelled"] = True
                j["status"] = "Cancelled"
                return f"Cancelled queued Job #{j['id']}."
    return "No active or queued jobs to cancel."


def poll_queue_state(current_selected_video):
    """
    Called every 1 second by gr.Timer to update the visible queue display,
    active job progress bar, queue backlog table, and completed video selector.
    """
    with jobs_lock:
        jobs_copy = list(all_jobs)
        active = current_active_job

    queued_count = sum(1 for j in jobs_copy if j["status"] == "Queued")
    completed_count = sum(1 for j in jobs_copy if j["status"] == "Completed")
    processing_count = 1 if active else 0

    # 1. Build Active Job Monitor HTML
    if active:
        progress_pct = active.get("progress", 0)
        stage_desc = active.get("stage", "Processing...")
        eta = active.get("eta_seconds", 0)
        job_id = active.get("id")
        prompt_snippet = active["params"]["prompt"][:60] + "..." if len(active["params"]["prompt"]) > 60 else active["params"]["prompt"]
        duration = active["params"]["duration"]
        fps = active["params"]["fps"]

        monitor_html = f"""
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(249, 115, 22, 0.4); border-radius: 10px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e;"></span>
                    <strong style="color: #f97316; font-size: 14px;">GENERATING JOB #{job_id}</strong>
                    <span style="color: #94a3b8; font-size: 12px;">({duration}s @ {fps}fps)</span>
                </div>
                <span style="font-family: monospace; font-size: 12px; color: #38bdf8; font-weight: 600;">
                    ETA: ~{eta}s remaining
                </span>
            </div>
            <div style="color: #cbd5e1; font-size: 13px; margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                <em>"{prompt_snippet}"</em>
            </div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 999px; height: 12px; overflow: hidden; margin-bottom: 6px;">
                <div style="background: linear-gradient(90deg, #f97316, #fb923c); height: 100%; width: {progress_pct}%; transition: width 0.3s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; font-family: monospace;">
                <span>{stage_desc}</span>
                <strong>{progress_pct}%</strong>
            </div>
        </div>
        """
    else:
        monitor_html = f"""
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 14px; margin-bottom: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 13px;">
                Worker Idle · No active generation in progress. Add jobs below to start the queue!
            </span>
        </div>
        """

    # 2. Build Queue Status Header & Summary
    summary_html = f"""
    <div style="display: flex; gap: 16px; align-items: center; font-size: 12px; color: #94a3b8; padding: 4px 0 10px 0;">
        <span>Queue Total: <strong style="color: #ffffff;">{len(jobs_copy)}</strong></span>
        <span>·</span>
        <span>Active: <strong style="color: #f97316;">{processing_count}</strong></span>
        <span>·</span>
        <span>Waiting in Queue: <strong style="color: #38bdf8;">{queued_count}</strong></span>
        <span>·</span>
        <span>Completed: <strong style="color: #4ade80;">{completed_count}</strong></span>
    </div>
    """

    # 3. Build Queue Table Rows
    table_rows = []
    # Reverse order so newest are on top, or FIFO
    for idx, j in enumerate(jobs_copy):
        status = j["status"]
        if status == "Processing":
            status_badge = f"⚡ Running ({j.get('progress', 0)}%)"
        elif status == "Queued":
            status_badge = f"⏳ In Queue (Wait #{idx + 1})"
        elif status == "Completed":
            status_badge = f"✅ Ready ({j.get('elapsed_seconds', 0)}s)"
        elif status == "Cancelled":
            status_badge = "❌ Cancelled"
        else:
            status_badge = "⚠️ Failed"

        has_audio = "Yes" if j["params"].get("audio_path") else "None"
        prompt_short = (j["params"]["prompt"][:40] + "...") if len(j["params"]["prompt"]) > 40 else j["params"]["prompt"]

        table_rows.append([
            f"#{j['id']}",
            status_badge,
            prompt_short,
            f"{j['params']['duration']}s",
            f"{j['params']['steps']} steps",
            has_audio,
            time.strftime("%H:%M:%S", time.localtime(j["created_at"]))
        ])

    # 4. Build Completed Video Options
    completed_jobs = [j for j in jobs_copy if j["status"] == "Completed" and j.get("video_path")]
    dropdown_choices = []
    for cj in reversed(completed_jobs):
        p_snippet = cj["params"]["prompt"][:35] + "..." if len(cj["params"]["prompt"]) > 35 else cj["params"]["prompt"]
        label = f"Job #{cj['id']} · {p_snippet} ({cj['params']['duration']}s)"
        dropdown_choices.append((label, cj["id"]))

    # Determine which video to show
    chosen_video_path = None
    chosen_telemetry = None

    if completed_jobs:
        # Default to latest completed or currently selected
        selected_job = None
        if current_selected_video:
            selected_job = next((j for j in completed_jobs if j["id"] == current_selected_video), None)
        if not selected_job:
            selected_job = completed_jobs[-1]  # Most recent

        chosen_video_path = selected_job.get("video_path")
        chosen_telemetry = selected_job.get("telemetry")

    return (
        summary_html + monitor_html,
        table_rows,
        gr.update(choices=dropdown_choices, value=current_selected_video if current_selected_video in [c[1] for c in dropdown_choices] else (dropdown_choices[0][1] if dropdown_choices else None)),
        chosen_video_path,
        chosen_telemetry
    )


def on_select_video(selected_job_id):
    """When user picks a different completed video from the queue dropdown."""
    with jobs_lock:
        job = next((j for j in all_jobs if j["id"] == selected_job_id and j.get("video_path")), None)
        if job:
            return job.get("video_path"), job.get("telemetry")
    return None, None


# -------------------------------------------------------------
# Gradio UI Layout
# -------------------------------------------------------------
custom_css = """
#app-container {
    max-width: 1380px;
    margin: 0 auto;
}
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 18px;
}
.primary-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.4) !important;
}
.queue-card {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px;
}
"""

with gr.Blocks(title="LTX-Video 2.3 — Sequential Queue Studio", css=custom_css, theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate")) as demo:
    gr.HTML("""
    <div class="header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 26px;">🎬</span>
                    <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
                        LTX-Video 2.3 Studio & Video Generation Queue
                    </h1>
                    <span style="background: rgba(249, 115, 22, 0.2); border: 1px solid #f97316; color: #f97316; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        v2.3 DiT
                    </span>
                    <span style="background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; color: #4ade80; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        Sequential FIFO Queue
                    </span>
                </div>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 13.5px;">
                    Queue multiple video generations with prompt, audio sync, and 2 image keyframes. Generations run sequentially in the background while you continue queuing!
                </p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span style="background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px; color: #94a3b8; font-size: 12px; font-family: monospace;">
                    Worker: 1 Concurrency · Strict FIFO
                </span>
            </div>
        </div>
    </div>
    """)

    # Top Section: Live Queue Tracker Panel
    gr.Markdown("### 📊 Live Generation Queue & Background Worker Monitor")
    queue_monitor_html = gr.HTML(value="""
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 14px; margin-bottom: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 13px;">Worker Ready. Queue your first video below!</span>
        </div>
    """)

    queue_table = gr.Dataframe(
        headers=["Job ID", "Queue Status", "Prompt", "Duration", "Steps", "Audio", "Time Queued"],
        datatype=["str", "str", "str", "str", "str", "str", "str"],
        value=[],
        label="Video Queue Backlog & Recent Generations",
        interactive=False,
    )

    with gr.Row():
        cancel_btn = gr.Button("⏹️ Cancel Current / Queued Job", size="sm", variant="secondary")
        status_feedback = gr.Markdown("")

    gr.Markdown("---")

    # Main 2-Column Interface: Inputs on Left, Output Player & Completed Queue on Right
    with gr.Row():
        # Left Column: Inputs & Controls
        with gr.Column(scale=5):
            gr.Markdown("### 📝 1. Prompt Conditioning")
            prompt_input = gr.Textbox(
                label="Video Prompt",
                placeholder="Describe cinematic camera movement, lighting, motion dynamics, and subject progression...",
                lines=3,
                value="Cinematic drone flythrough of glowing deep space stardust nebula, smoothly approaching an alien crystalline planet with emerald aurora skies, ultra high definition 8k."
            )

            with gr.Accordion("Negative Prompt & Quality Filters", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value="blurry, distorted, artifacts, low resolution, flickering, static, camera shake, jitter, deformed"
                )

            gr.Markdown("### 🖼️ 2. Dual Image Conditioning (Start Frame 0 & End Frame N)")
            with gr.Row():
                image1_input = gr.Image(
                    label="Frame 0 (Start Image)",
                    type="filepath",
                    value=IMG_NEBULA if os.path.exists(IMG_NEBULA) else None
                )
                image2_input = gr.Image(
                    label="Frame N (End Image / Style Reference)",
                    type="filepath",
                    value=IMG_PLANET if os.path.exists(IMG_PLANET) else None
                )

            gr.Markdown("### 🎵 3. Audio Upload (Motion & Beat Sync)")
            audio_input = gr.Audio(
                label="Conditioning Audio (WAV / MP3)",
                type="filepath",
                value=AUDIO_COSMIC if os.path.exists(AUDIO_COSMIC) else None
            )

            with gr.Accordion("⚙️ LTX-Video 2.3 Hyperparameters", open=False):
                with gr.Row():
                    conditioning_mode = gr.Radio(
                        choices=["First-to-Last Frame Interpolation", "Image-to-Video + Style Guide"],
                        value="First-to-Last Frame Interpolation",
                        label="Conditioning Strategy"
                    )
                    audio_sync_mode = gr.Dropdown(
                        choices=["Beat-Driven Camera Motion", "Ambient Flow", "Spectral Pulse"],
                        value="Beat-Driven Camera Motion",
                        label="Audio Reactivity Mode"
                    )

                with gr.Row():
                    duration_slider = gr.Slider(
                        minimum=2.0, maximum=10.0, value=4.0, step=0.5,
                        label="Duration (seconds)"
                    )
                    aspect_ratio_radio = gr.Radio(
                        choices=["16:9", "9:16", "1:1", "4:3"],
                        value="16:9",
                        label="Aspect Ratio"
                    )

                with gr.Row():
                    fps_slider = gr.Slider(
                        minimum=15, maximum=60, value=24, step=1,
                        label="Frame Rate (FPS)"
                    )
                    steps_slider = gr.Slider(
                        minimum=10, maximum=50, value=30, step=1,
                        label="Inference Steps"
                    )

                with gr.Row():
                    cfg_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.5, step=0.1,
                        label="Guidance Scale (CFG)"
                    )
                    motion_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                        label="Motion Dynamics / Strength"
                    )

                with gr.Row():
                    seed_input = gr.Number(value=42890, label="Seed", precision=0)
                    randomize_checkbox = gr.Checkbox(label="Randomize Seed", value=True)

            with gr.Row():
                enqueue_btn = gr.Button(
                    "🎬 Queue Video Generation",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"]
                )
                clear_btn = gr.ClearButton(
                    components=[prompt_input, image1_input, image2_input, audio_input],
                    value="Reset Inputs"
                )

            enqueue_msg = gr.Markdown("")

        # Right Column: Output Video Player & Completed Video Queue Browser
        with gr.Column(scale=5):
            gr.Markdown("### 📺 Video Player & Completed Queue Outputs")
            video_selector = gr.Dropdown(
                label="Select Generated Video from Queue",
                choices=[],
                value=None,
                interactive=True,
                info="Browse all completed videos generated by the queue"
            )

            video_output = gr.Video(
                label="LTX-Video 2.3 Synthesized Video Player",
                interactive=False,
                autoplay=True
            )

            telemetry_output = gr.JSON(
                label="Selected Video Generation Telemetry & Metadata"
            )

    # Examples Section
    gr.Markdown("### 💡 Clickable Examples (Load Prompt, Audio & 2 Images)")
    example_list = []
    if os.path.exists(IMG_NEBULA) and os.path.exists(IMG_PLANET) and os.path.exists(AUDIO_COSMIC):
        example_list.append([
            "Cinematic drone flythrough of glowing deep space stardust nebula, smoothly approaching an alien crystalline planet with emerald aurora skies, ultra high definition 8k.",
            "blurry, distorted, artifacts, low resolution",
            IMG_NEBULA,
            IMG_PLANET,
            AUDIO_COSMIC,
            "First-to-Last Frame Interpolation",
            "Beat-Driven Camera Motion",
            4.0,
            "16:9",
            24,
            30,
            3.5,
            0.8,
            10482,
            True
        ])

    if os.path.exists(IMG_CYBER_DAY) and os.path.exists(IMG_CYBER_NIGHT) and os.path.exists(AUDIO_CYBER):
        example_list.append([
            "Time-lapse transition of a bustling Tokyo cyberpunk alleyway transforming from bright morning daylight into a rain-slicked neon illuminated nightscape with glowing reflections.",
            "low contrast, jittery, overexposed",
            IMG_CYBER_DAY,
            IMG_CYBER_NIGHT,
            AUDIO_CYBER,
            "First-to-Last Frame Interpolation",
            "Ambient Flow",
            4.0,
            "16:9",
            24,
            30,
            4.0,
            0.7,
            99201,
            True
        ])

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=[
                prompt_input,
                negative_prompt_input,
                image1_input,
                image2_input,
                audio_input,
                conditioning_mode,
                audio_sync_mode,
                duration_slider,
                aspect_ratio_radio,
                fps_slider,
                steps_slider,
                cfg_slider,
                motion_slider,
                seed_input,
                randomize_checkbox
            ],
            outputs=[video_output, telemetry_output],
            fn=lambda *args: None,
            cache_examples=False
        )

    # ---------------------------------------------------------
    # Event Handlers & Timer Polling
    # ---------------------------------------------------------

    # 1. Enqueue button adds job to FIFO queue instantly
    enqueue_btn.click(
        fn=add_to_queue,
        inputs=[
            prompt_input,
            negative_prompt_input,
            image1_input,
            image2_input,
            audio_input,
            conditioning_mode,
            audio_sync_mode,
            duration_slider,
            aspect_ratio_radio,
            fps_slider,
            steps_slider,
            cfg_slider,
            motion_slider,
            seed_input,
            randomize_checkbox
        ],
        outputs=[enqueue_msg]
    )

    # 2. Cancel button
    cancel_btn.click(
        fn=cancel_active_or_queued,
        outputs=[status_feedback]
    )

    # 3. Video selector dropdown change
    video_selector.change(
        fn=on_select_video,
        inputs=[video_selector],
        outputs=[video_output, telemetry_output]
    )

    # 4. Timer polls queue status every 1.0 second and updates the UI
    queue_timer = gr.Timer(value=1.0)
    queue_timer.tick(
        fn=poll_queue_state,
        inputs=[video_selector],
        outputs=[
            queue_monitor_html,
            queue_table,
            video_selector,
            video_output,
            telemetry_output
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-Video 2.3 Gradio Studio with Queue")
    parser.add_argument("--port", type=int, default=3000, help="Server port (default: 3000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args, _ = parser.parse_known_args()

    # Launch Gradio interface
    print(f"Starting Python Gradio LTX-Video 2.3 Sequential Queue Studio on http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True
    )
