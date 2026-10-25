"""
Audio and Video I/O helpers for LTX-Video and Wan2GP pipelines.
Implements tensor to MP4 saving with FFmpeg encoding.
"""

import os
import shutil
import tempfile
import subprocess
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


def save_video(tensor, save_file: str, fps: float = 24.0, nrow: int = 1, normalize: bool = False, value_range=(-1, 1)):
    """
    Saves a 4D or 5D PyTorch tensor or NumPy array as an MP4 video file using FFmpeg.
    Supports shapes:
      - (B, C, T, H, W)
      - (C, T, H, W)
      - (T, H, W, C)
    """
    frames = None

    if torch is not None and torch.is_tensor(tensor):
        t = tensor.detach().cpu()
        if t.dim() == 5:
            t = t[0]  # Take first batch item -> (C, T, H, W)
        if t.dim() == 4:
            if t.shape[0] in (1, 3, 4):  # (C, T, H, W) -> (T, H, W, C)
                t = t.permute(1, 2, 3, 0)

        if t.dtype == torch.uint8:
            frames = t.numpy()
        else:
            t = t.float()
            if normalize:
                low, high = value_range
                t = (t - low) / max(high - low, 1e-6)
            if t.max() <= 1.0 and t.min() >= 0.0:
                t = t * 255.0
            frames = t.clamp(0, 255).to(torch.uint8).numpy()
    elif isinstance(tensor, np.ndarray):
        frames = tensor
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[0] in (1, 3, 4):
            frames = np.transpose(frames, (1, 2, 3, 0))
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0 if frames.max() <= 1.0 else frames, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported video tensor type: {type(tensor)}")

    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="save_video_")
    pattern = os.path.join(temp_dir, "frame_%05d.png")

    try:
        num_frames = frames.shape[0]
        for i in range(num_frames):
            frame_arr = frames[i]
            if frame_arr.ndim == 2:
                frame_arr = np.repeat(frame_arr[..., None], 3, axis=-1)
            elif frame_arr.shape[-1] == 1:
                frame_arr = np.repeat(frame_arr, 3, axis=-1)
            elif frame_arr.shape[-1] > 3:
                frame_arr = frame_arr[..., :3]

            img = Image.fromarray(frame_arr)
            img.save(pattern % i)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(save_file)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return save_file
