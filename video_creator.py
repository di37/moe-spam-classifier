"""
Turn a folder of numbered frames into an H.264 MP4 and a smaller looping GIF.

    python video_creator.py frames/training_moe media/training_moe

writes media/training_moe.mp4 and media/training_moe.gif.
"""
import argparse
import os
import re

import imageio_ffmpeg
import numpy as np
from PIL import Image

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _frame_number(name):
    match = re.match(r"\d+", name)
    return int(match.group()) if match else None


def sorted_frames(folder):
    """Frame paths ordered by the number their name starts with (not by text or file time)."""
    names = [n for n in os.listdir(folder)
             if n.lower().endswith(FRAME_EXTENSIONS) and _frame_number(n) is not None]
    return [os.path.join(folder, n) for n in sorted(names, key=_frame_number)]


def _load_rgb(path, size=None):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    return rgb.crop((0, 0, *size)) if size else rgb


def write_mp4(paths, out_path, fps, hold_frames):
    """H.264 + yuv420p, which plays inline in browsers and on GitHub."""
    w, h = _load_rgb(paths[0]).size
    size = (w - w % 2, h - h % 2)  # H.264 needs even dimensions
    writer = imageio_ffmpeg.write_frames(out_path, size, fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=1)
    writer.send(None)
    try:
        for path in paths:
            frame = np.ascontiguousarray(np.asarray(_load_rgb(path, size)))
            writer.send(frame)
        for _ in range(hold_frames):  # pause on the last frame
            writer.send(frame)
    finally:
        writer.close()


def write_gif(paths, out_path, fps, every, width, hold_seconds):
    """Every `every`-th frame, scaled to `width`, always ending on the last frame."""
    picked = paths[::every]
    if picked[-1] != paths[-1]:
        picked = picked + [paths[-1]]
    first = _load_rgb(picked[0])
    size = (width, round(first.height * width / first.width))
    frames = [_load_rgb(p).resize(size, Image.LANCZOS) for p in picked]
    durations = [round(1000 * every / fps)] * (len(frames) - 1) + [round(1000 * hold_seconds)]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)


def make_video(folder, out_base, fps=15, hold_last=2.0, gif_width=560, gif_every=3):
    """Write out_base.mp4 and out_base.gif from the frames in `folder`."""
    if gif_every < 1:
        raise ValueError(f"gif_every must be at least 1, got {gif_every}")
    paths = sorted_frames(folder)
    if not paths:
        raise SystemExit(f"no frames found in {folder}")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    write_mp4(paths, out_base + ".mp4", fps, round(hold_last * fps))
    write_gif(paths, out_base + ".gif", fps, gif_every, gif_width, hold_last)
    print(f"wrote {out_base}.mp4 and {out_base}.gif from {len(paths)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", help="folder of frames named <number>.jpg")
    parser.add_argument("out", help="output path without extension")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gif-every", type=int, default=3, choices=range(1, 31), metavar="N",
                        help="keep every N-th frame in the GIF")
    args = parser.parse_args()
    make_video(args.frames, args.out, fps=args.fps, gif_every=args.gif_every)
