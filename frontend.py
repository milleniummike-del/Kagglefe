"""
LTX-Video 2.3 Gradio Frontend Client
Communicates with the dedicated FastAPI backend queue service on port 7860 via HTTP REST.
Features a visible real-time queue monitor, non-blocking FIFO submission,
batch project processing, and interactive video playback.
"""

import os
import time
from pathlib import Path

import gradio as gr
import requests

BACKEND_URL = os.environ.get("LTX_BACKEND_URL", "http://127.0.0.1:7860")

# Sample assets for examples
ASSET_DIR = Path("sample_assets/images")
SAMPLE_AUDIO_DIR = Path("sample_assets")

IMG_NEBULA = str(ASSET_DIR / "ltx_space_nebula_1792873970962.jpg") if (ASSET_DIR / "ltx_space_nebula_1792873970962.jpg").exists() else None
IMG_PLANET = str(ASSET_DIR / "ltx_crystal_planet_1792873971765.jpg") if (ASSET_DIR / "ltx_crystal_planet_1792873971765.jpg").exists() else None
AUDIO_COSMIC = str(SAMPLE_AUDIO_DIR / "cosmic_synth.wav") if (SAMPLE_AUDIO_DIR / "cosmic_synth.wav").exists() else None


# -------------------------------------------------------------
# Frontend Actions calling Backend REST API
# -------------------------------------------------------------
def submit_prompt_to_backend(
    prompt_text: str,
    image_path: str,
    audio_path: str,
    seed_value: int,
    duration_value: str,
    resolution_value: str,
    aspect_value: str,
    match_audio_value: bool,
):
    """
    Submits a video generation prompt to the backend FIFO queue.
    Returns immediately without blocking the browser!
    """
    prompt_clean = (prompt_text or "").strip()
    if not prompt_clean:
        return "⚠️ Please enter a prompt before clicking Submit to Queue."

    payload = {
        "prompt": prompt_clean,
        "image_path": image_path,
        "audio_path": audio_path,
        "seed": int(seed_value) if seed_value is not None else -1,
        "duration": duration_value,
        "resolution": resolution_value,
        "aspect_ratio": aspect_value,
        "match_audio_dur": bool(match_audio_value),
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/api/queue/submit", json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ **Job #{data['job_id']} Added to Queue!** (Position: **#{data['position']}**)\nThe background worker will generate this video sequentially. You can continue submitting more prompts below!"
        else:
            return f"❌ Backend Error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"❌ Connection Error: Cannot reach backend service at {BACKEND_URL}: {e}"


def cancel_active_or_queued():
    """Requests the backend to cancel the active job or remove a queued job."""
    try:
        status_resp = requests.get(f"{BACKEND_URL}/api/queue/status", timeout=3)
        if status_resp.status_code == 200:
            data = status_resp.json()
            target_id = None
            if data.get("active_job"):
                target_id = data["active_job"]["id"]
            elif data.get("queued_jobs"):
                target_id = data["queued_jobs"][-1]["id"]

            if target_id:
                cancel_resp = requests.post(f"{BACKEND_URL}/api/jobs/{target_id}/cancel", timeout=3)
                if cancel_resp.status_code == 200:
                    return f"🛑 Successfully requested cancellation for Job #{target_id}."
            return "ℹ️ No active or queued jobs found to cancel."
    except Exception as e:
        return f"❌ Error contacting backend: {e}"


def handle_zip_extract(zip_file):
    """Uploads project ZIP to backend and extracts manifest."""
    if not zip_file:
        return "❌ Please upload a project ZIP file first.", ""

    file_path = zip_file if isinstance(zip_file, str) else getattr(zip_file, "name", str(zip_file))
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/api/project/upload_extract",
                files={"file": (os.path.basename(file_path), f, "application/zip")},
                timeout=30
            )
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ Extracted project archive.\n📂 Folder: {data['project_folder']}\n🎬 Found {data['piece_count']} pieces in manifest.json.\n\nReady to Run!", data["project_folder"]
        else:
            return f"❌ Extraction failed: {resp.text}", ""
    except Exception as e:
        return f"❌ Error contacting backend: {e}", ""


def handle_project_run(project_folder):
    """Enqueues all pieces from extracted project into the backend queue."""
    if not project_folder:
        return "❌ No project folder specified. Please extract a ZIP first."

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/project/queue_batch",
            data={"project_folder": project_folder},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return f"🚀 **Enqueued {data['queued_pieces']} Pieces!**\nAll pieces have been added to the sequential generation queue and will be processed in order.\nJob IDs: {', '.join(data['job_ids'][:10])}"
        else:
            return f"❌ Failed to run project: {resp.text}"
    except Exception as e:
        return f"❌ Error contacting backend: {e}"


def poll_queue_updates(current_video_selection):
    """
    Called every 1.0s by gr.Timer to update the live queue dashboard,
    active generation card, backlog table, and completed videos dropdown.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/api/queue/status", timeout=2)
        if resp.status_code != 200:
            return (
                f"<div style='color: #ef4444; padding: 10px;'>Backend Service Status: {resp.status_code}</div>",
                [],
                gr.update(),
                None,
                None
            )
        data = resp.json()
    except Exception:
        return (
            "<div style='color: #eab308; padding: 8px;'>Connecting to LTX-2.3 Backend Service...</div>",
            [],
            gr.update(),
            None,
            None
        )

    active = data.get("active_job")
    queued_jobs = data.get("queued_jobs", [])
    completed_jobs = data.get("completed_jobs", [])
    summary = data.get("summary", {})

    # 1. Build Active Job HTML Card
    if active:
        pct = active.get("progress", 0)
        stage = active.get("stage", "Generating...")
        eta = active.get("eta_seconds", 0)
        job_id = active.get("id")
        prompt_txt = active.get("params", {}).get("prompt", "")
        prompt_snippet = (prompt_txt[:65] + "...") if len(prompt_txt) > 65 else prompt_txt

        monitor_html = f"""
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(249, 115, 22, 0.5); border-radius: 10px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e;"></span>
                    <strong style="color: #f97316; font-size: 14px;">GENERATING JOB #{job_id}</strong>
                </div>
                <span style="font-family: monospace; font-size: 12px; color: #38bdf8; font-weight: 600;">
                    ETA: ~{eta}s remaining
                </span>
            </div>
            <div style="color: #cbd5e1; font-size: 13px; margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                <em>"{prompt_snippet}"</em>
            </div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 999px; height: 12px; overflow: hidden; margin-bottom: 6px;">
                <div style="background: linear-gradient(90deg, #f97316, #fb923c); height: 100%; width: {pct}%; transition: width 0.3s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; font-family: monospace;">
                <span>{stage}</span>
                <strong>{pct}%</strong>
            </div>
        </div>
        """
    else:
        monitor_html = """
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 14px; margin-bottom: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 13px;">
                Worker Idle · Ready to accept jobs. Click "Submit Prompt to Queue" below!
            </span>
        </div>
        """

    summary_html = f"""
    <div style="display: flex; gap: 16px; align-items: center; font-size: 12px; color: #94a3b8; padding: 2px 0 8px 0;">
        <span>Active Worker: <strong style="color: #f97316;">{'Running' if active else 'Idle'}</strong></span>
        <span>·</span>
        <span>Waiting in Queue: <strong style="color: #38bdf8;">{summary.get('queued', 0)}</strong></span>
        <span>·</span>
        <span>Completed Total: <strong style="color: #4ade80;">{summary.get('total_processed', 0)}</strong></span>
    </div>
    """

    # 2. Build Queue Backlog Table
    table_rows = []
    if active:
        p_sub = active["params"]["prompt"][:40] + "..." if len(active["params"]["prompt"]) > 40 else active["params"]["prompt"]
        table_rows.append([
            f"#{active['id']}",
            f"⚡ Processing ({active.get('progress', 0)}%)",
            p_sub,
            active["params"].get("duration", "5s"),
            active["params"].get("resolution", "720p"),
            time.strftime("%H:%M:%S", time.localtime(active.get("started_at", time.time())))
        ])

    for idx, qj in enumerate(queued_jobs):
        p_sub = qj["params"]["prompt"][:40] + "..." if len(qj["params"]["prompt"]) > 40 else qj["params"]["prompt"]
        table_rows.append([
            f"#{qj['id']}",
            f"⏳ Waiting (Position #{idx + 1})",
            p_sub,
            qj["params"].get("duration", "5s"),
            qj["params"].get("resolution", "720p"),
            time.strftime("%H:%M:%S", time.localtime(qj.get("created_at", time.time())))
        ])

    # 3. Completed Videos Options
    dropdown_choices = []
    for cj in completed_jobs:
        if cj.get("video_path") and os.path.exists(cj["video_path"]):
            p_short = cj["params"]["prompt"][:35] + "..." if len(cj["params"]["prompt"]) > 35 else cj["params"]["prompt"]
            label = f"Job #{cj['id']} · {p_short} ({cj.get('elapsed_seconds', 0)}s)"
            dropdown_choices.append((label, cj["id"]))

    chosen_video = None
    chosen_telemetry = None

    if completed_jobs:
        selected_job = None
        if current_video_selection:
            selected_job = next((j for j in completed_jobs if j["id"] == current_video_selection), None)
        if not selected_job:
            for j in completed_jobs:
                if j.get("video_path") and os.path.exists(j["video_path"]):
                    selected_job = j
                    break

        if selected_job and selected_job.get("video_path") and os.path.exists(selected_job["video_path"]):
            chosen_video = selected_job["video_path"]
            chosen_telemetry = selected_job.get("telemetry")

    current_val = current_video_selection if current_video_selection in [c[1] for c in dropdown_choices] else (dropdown_choices[0][1] if dropdown_choices else None)

    return (
        summary_html + monitor_html,
        table_rows,
        gr.update(choices=dropdown_choices, value=current_val),
        chosen_video,
        chosen_telemetry
    )


def on_select_video(selected_job_id):
    """When user picks a different completed video from the queue dropdown."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/queue/status", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            for j in data.get("completed_jobs", []):
                if j["id"] == selected_job_id:
                    return j.get("video_path"), j.get("telemetry")
    except Exception:
        pass
    return None, None


# -------------------------------------------------------------
# Gradio UI Design
# -------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.gradio-container { max-width: 1350px !important; margin: auto !important; }
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.primary-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: white !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.4) !important;
}
.secondary-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(title="LTX-Video 2.3 — Decoupled Queue Studio") as demo:
    gr.HTML("""
    <div class="header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 26px;">🎬</span>
                    <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
                        LTX-Video 2.3 Decoupled Queue Studio
                    </h1>
                    <span style="background: rgba(249, 115, 22, 0.2); border: 1px solid #f97316; color: #f97316; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        FastAPI Service + Gradio Client
                    </span>
                    <span style="background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; color: #4ade80; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        Sequential FIFO Queue
                    </span>
                </div>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 13.5px;">
                    Independent backend worker processes jobs sequentially. Submit multiple video generations without UI lockup or browser connection loss!
                </p>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 6px; color: #38bdf8; font-size: 12px; font-family: monospace;">
                    Backend: http://127.0.0.1:7860
                </span>
            </div>
        </div>
    </div>
    """)

    # Top Section: Real-Time Queue Monitor
    gr.Markdown("### 📊 Live Generation Queue & Background Service Worker")
    queue_monitor_html = gr.HTML(value="""
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 14px; margin-bottom: 12px; text-align: center;">
            <span style="color: #94a3b8; font-size: 13px;">Connecting to backend worker...</span>
        </div>
    """)

    queue_table = gr.Dataframe(
        headers=["Job ID", "Status", "Prompt", "Duration", "Resolution", "Queued At"],
        datatype=["str", "str", "str", "str", "str", "str"],
        value=[],
        label="Sequential Queue Backlog (FIFO Order)",
        interactive=False,
    )

    with gr.Accordion("🧠 Real LTX-2.3 22B Distilled Model & GPU Status", open=False):
        model_status_html = gr.HTML("""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 14px;">
                <div style="font-weight: 600; color: #f97316; margin-bottom: 6px;">LTX-Video 2.3 22B Distilled Engine</div>
                <div style="color: #94a3b8; font-size: 13px;">Real model weights (GGUF Q4_K_M) load automatically when located in <code>models/</code> or specified path.</div>
            </div>
        """)
        with gr.Row():
            custom_model_path = gr.Textbox(
                label="Custom Transformer GGUF Path",
                placeholder="models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf",
                value="models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"
            )
            custom_encoder_path = gr.Textbox(
                label="Custom Text Encoder Path",
                placeholder="/kaggle/tmp/models/gemma-3-12b-it-qat-q4_0-unquantized",
                value=""
            )
        with gr.Row():
            load_model_btn = gr.Button("⚡ Load / Reload Real LTX-2.3 Model", variant="secondary")
            model_load_status = gr.Markdown("")

        def handle_load_model(m_path, enc_path):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/api/model/load",
                    json={"transformer_path": m_path or None, "text_encoder_path": enc_path or None},
                    timeout=60
                )
                if r.status_code == 200:
                    d = r.json()
                    if d.get("real_model_loaded"):
                        return f"✅ **Real LTX-2.3 Model Loaded!** GPU inference is active for subsequent queue jobs."
                    else:
                        return f"⚠️ **Could not load real model:** {d.get('message')}\nUsing fallback synthesizer."
                return f"❌ Request error ({r.status_code}): {r.text}"
            except Exception as e:
                return f"❌ Error contacting backend: {e}"

        load_model_btn.click(
            fn=handle_load_model,
            inputs=[custom_model_path, custom_encoder_path],
            outputs=[model_load_status]
        )

    with gr.Row():
        cancel_btn = gr.Button("🛑 Cancel Active / Queued Job", size="sm", variant="secondary")
        status_feedback = gr.Markdown("")

    gr.Markdown("---")

    # Tabs for Single Generation vs Batch Project Processing
    with gr.Tabs():
        with gr.TabItem("🎬 Single Prompt Generation"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=5):
                    gr.Markdown("### 📝 Video Conditioning Inputs")
                    prompt_input = gr.Textbox(
                        label="Prompt to Submit",
                        placeholder=(
                            "Enter video prompt, then click 'Submit Prompt to Queue'.\n"
                            "Each click adds a new job to the queue without waiting!\n\n"
                            "[VISUAL] A professional speaker talking directly to camera in a modern studio. "
                            "[SPEECH] Welcome everyone to our new AI tutorial! [SOUND] Clear expressive voice."
                        ),
                        lines=4,
                        value="[VISUAL] Cinematic drone flythrough of glowing deep space stardust nebula, smoothly approaching an alien crystalline planet. [SPEECH] Journey to the frontier of deep space. [SOUND] Synth ambient pulse."
                    )

                    with gr.Row():
                        image_input = gr.Image(
                            label="🖼️ Reference Image (Portrait / Start Frame)",
                            type="filepath",
                            value=IMG_NEBULA
                        )
                        audio_input = gr.Audio(
                            label="🎤 Speech / Vocals Audio Track",
                            type="filepath",
                            value=AUDIO_COSMIC
                        )

                    with gr.Row():
                        seed_input = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
                        duration_dropdown = gr.Dropdown(
                            label="⏱️ Duration",
                            choices=[
                                "2 Seconds (49 frames)",
                                "3 Seconds (73 frames)",
                                "5 Seconds (121 frames)",
                                "8 Seconds (193 frames)",
                                "10 Seconds (241 frames)",
                                "15 Seconds (361 frames)",
                                "20 Seconds (481 frames)",
                            ],
                            value="5 Seconds (121 frames)",
                        )
                        match_audio_dur = gr.Checkbox(
                            label="🎵 Auto-match video duration to audio file",
                            value=False
                        )

                    with gr.Row():
                        resolution_dropdown = gr.Dropdown(
                            label="📐 Base Resolution",
                            choices=["1080p", "720p", "540p", "480p", "360p", "180p"],
                            value="720p",
                        )
                        aspect_ratio_dropdown = gr.Dropdown(
                            label="📏 Aspect Ratio",
                            choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                            value="16:9 Landscape",
                        )

                    with gr.Row():
                        queue_btn = gr.Button("➕ Submit Prompt to Queue", variant="primary", size="lg", elem_classes=["primary-btn"])
                        clear_btn = gr.ClearButton(components=[prompt_input, image_input, audio_input], value="🗑️ Clear Inputs")

                    submission_msg = gr.Markdown("")

                # Right Column: Output Video Player & Completed Dropdown
                with gr.Column(scale=5):
                    gr.Markdown("### 📺 Generated Video Player & Output History")
                    video_selector = gr.Dropdown(
                        label="Select Completed Video from Queue",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Browse all completed videos generated by the backend queue"
                    )

                    video_output = gr.Video(
                        label="Generated LTX-Video 2.3 Output",
                        interactive=False,
                        autoplay=True
                    )

                    telemetry_output = gr.JSON(
                        label="Selected Video Generation Telemetry & Metadata"
                    )

        with gr.TabItem("📦 Batch Project Processing (ZIP)"):
            gr.Markdown(
                """
                ### 📦 Process a complete AI video project
                Upload a ZIP archive containing `manifest.json` and referenced image and audio files.
                All pieces will be enqueued into the backend queue and processed sequentially!
                """
            )
            with gr.Row():
                project_zip = gr.File(label="📦 Upload Project ZIP", file_types=[".zip"], type="filepath")
                project_folder_text = gr.Textbox(label="📂 Extracted Project Folder", value="", interactive=False)

            with gr.Row():
                extract_btn = gr.Button("📂 Extract & Validate ZIP", variant="secondary", size="lg")
                run_project_btn = gr.Button("▶️ Queue Batch Project", variant="primary", size="lg", elem_classes=["primary-btn"])

            project_status = gr.Textbox(label="📋 Project Status", lines=8, interactive=False)

            extract_btn.click(
                fn=handle_zip_extract,
                inputs=[project_zip],
                outputs=[project_status, project_folder_text]
            )

            run_project_btn.click(
                fn=handle_project_run,
                inputs=[project_folder_text],
                outputs=[project_status]
            )

    # Clickable Examples
    gr.Markdown("### 💡 Clickable Examples")
    example_list = []
    if IMG_NEBULA and AUDIO_COSMIC:
        example_list.append([
            "[VISUAL] Cinematic space nebula flythrough approaching crystalline alien planet. [SPEECH] Exploration begins.",
            IMG_NEBULA,
            AUDIO_COSMIC,
            42890,
            "5 Seconds (121 frames)",
            "720p",
            "16:9 Landscape",
            False
        ])

    if IMG_PLANET:
        example_list.append([
            "[VISUAL] Crystalline planet landscape with glowing emerald auroras under deep starry skies.",
            IMG_PLANET,
            None,
            10482,
            "3 Seconds (73 frames)",
            "720p",
            "16:9 Landscape",
            False
        ])

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=[
                prompt_input,
                image_input,
                audio_input,
                seed_input,
                duration_dropdown,
                resolution_dropdown,
                aspect_ratio_dropdown,
                match_audio_dur
            ],
            fn=lambda *args: None,
            cache_examples=False
        )

    # ---------------------------------------------------------
    # Event Handlers & Timer Hookup
    # ---------------------------------------------------------

    # 1. Non-blocking prompt submission
    queue_btn.click(
        fn=submit_prompt_to_backend,
        inputs=[
            prompt_input,
            image_input,
            audio_input,
            seed_input,
            duration_dropdown,
            resolution_dropdown,
            aspect_ratio_dropdown,
            match_audio_dur
        ],
        outputs=[submission_msg]
    )

    # 2. Cancel button
    cancel_btn.click(
        fn=cancel_active_or_queued,
        outputs=[status_feedback]
    )

    # 3. Video selection change
    video_selector.change(
        fn=on_select_video,
        inputs=[video_selector],
        outputs=[video_output, telemetry_output]
    )

    # 4. Timer updates queue display every 1.0s
    queue_timer = gr.Timer(value=1.0)
    queue_timer.tick(
        fn=poll_queue_updates,
        inputs=[video_selector],
        outputs=[
            queue_monitor_html,
            queue_table,
            video_selector,
            video_output,
            telemetry_output
        ]
    )


def start_frontend(host="0.0.0.0", port=3000):
    print(f"[Frontend] Starting Gradio UI on http://{host}:{port}")
    demo.launch(
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CSS,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    start_frontend()
