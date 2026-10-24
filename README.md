# LTX-Video 2.3 — Pure Python Gradio Queue Studio

An official-grade, **100% Python** Gradio application for **LTX-Video 2.3 (Lightricks)** video generation with a live, visible **Video Generation Queue** powered by sequential FIFO background processing, prompt conditioning, audio upload for motion sync, and dual keyframe images.

## Structure

```
├── app.py                # Pure Python Gradio application & FIFO queue worker
├── sample_assets/        # Synthetic audio and demo keyframe image assets
│   ├── images/
│   │   ├── ltx_space_nebula_*.jpg
│   │   ├── ltx_crystal_planet_*.jpg
│   │   ├── ltx_cyber_alley_day_*.jpg
│   │   └── ltx_cyber_alley_neon_*.jpg
│   ├── cosmic_synth.wav
│   ├── cyber_rain.wav
│   └── cinematic_bass.wav
├── README.md
├── metadata.json
└── package.json          # Container startup runner (python3 app.py)
```

## Features

- **100% Python Codebase**: Built exclusively with Python (`gradio`, `pillow`, `subprocess` with `ffmpeg`), with zero JavaScript/TypeScript.
- **Visible Multi-Job Video Generation Queue**:
  - **Live Worker Monitor**: Real-time progress bar (0%–100%), pipeline stage description, and remaining ETA countdown.
  - **Interactive Queue Backlog Table**: Shows all jobs in the queue with status badges:
    - `⚡ Running (Step X/30 · Y%)`
    - `⏳ In Queue (Wait #1, #2, ...)`
    - `✅ Ready (Elapsed time)`
    - `❌ Cancelled`
  - **Sequential Processing (FIFO)**: Queue multiple jobs one after another without waiting for the UI. The background worker processes them one by one in order.
  - **Completed Video Selector**: Browse and play all generated videos from the queue history with full video controls, download option, and detailed generation telemetry.
  - **Cancel Control**: Immediate cancellation of the running job or pending queued jobs.
- **Inputs & Conditioning**:
  - **Prompt**: Textarea with cinematic suggestions + Negative prompt filter accordion.
  - **Dual Image Uploads**: `gr.Image` for Frame 0 (Start Image) and Frame N (End Image / Style Reference).
  - **Audio Upload**: `gr.Audio` for conditioning audio with beat-driven camera motion, ambient flow, and spectral pulse sync.
  - **LTX-Video 2.3 Hyperparameters**: Duration, Aspect Ratio (16:9, 9:16, 1:1, 4:3), FPS, Inference Steps (10–50), CFG Guidance Scale, Motion Dynamics, and Seed controls.
- **Mock Video Generation Engine**:
  - Pure Python synthesizer using Pillow and FFmpeg rendering smooth camera motion, easing transitions, and audio multiplexing to standard H.264 MP4.
- **Pre-populated Clickable Examples**:
  - Cosmic Nebula to Crystalline Planet (with synth pad audio)
  - Tokyo Cyberpunk Daylight to Neon Nightfall (with ambient rain audio)

## Running the App

```bash
python3 app.py --port 3000 --host 0.0.0.0
```
