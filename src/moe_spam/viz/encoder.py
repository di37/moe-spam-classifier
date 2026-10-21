"""Turning a folder of numbered frames into an H.264 MP4 and a smaller looping GIF."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from moe_spam.errors import MoeSpamError

FPS = 15  # frames per second of every video
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class VideoEncoder:
    """How frames become a video and a GIF.

    Attributes:
        fps: Frames per second.
        hold_last: Seconds the last frame stays on screen.
        gif_width: Width of the GIF in pixels.
        gif_every: Keep every this-many-th frame in the GIF.
    """

    fps: int = FPS
    hold_last: float = 2.0
    gif_width: int = 560
    gif_every: int = 3

    def __post_init__(self):
        """Checks the settings.

        Raises:
            ValueError: If ``fps``, ``gif_width`` or ``gif_every`` is below 1, or
                ``hold_last`` is negative.
        """
        for name in ("fps", "gif_width", "gif_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.hold_last < 0:
            raise ValueError(f"hold_last must be at least 0, got {self.hold_last}")

    def encode(self, folder, out_base):
        """Writes ``out_base.mp4`` and ``out_base.gif`` from the frames in ``folder``.

        Args:
            folder: The folder of frames, named ``<number>.jpg`` (or ``.jpeg``, ``.png``).
            out_base: The output path without its extension.

        Raises:
            MoeSpamError: If the folder is missing or has no frames.
        """
        if not Path(folder).is_dir():
            raise MoeSpamError(f"frames folder not found: {folder}")
        paths = self.sorted_frames(folder)
        if not paths:
            raise MoeSpamError(f"no frames found in {folder}")
        out_base = str(out_base)
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        self._write_mp4(paths, out_base + ".mp4", round(self.hold_last * self.fps))
        self._write_gif(paths, out_base + ".gif")
        print(f"wrote {out_base}.mp4 and {out_base}.gif from {len(paths)} frames")

    @staticmethod
    def sorted_frames(folder):
        """Returns the frame paths, ordered by the number their name starts with."""
        names = [n for n in os.listdir(folder)
                 if n.lower().endswith(FRAME_EXTENSIONS) and _frame_number(n) is not None]
        return [os.path.join(folder, n) for n in sorted(names, key=_frame_number)]

    def _write_mp4(self, paths, out_path, hold_frames):
        """H.264 + yuv420p, which plays inline in browsers and on GitHub."""
        w, h = _load_rgb(paths[0]).size
        size = (w - w % 2, h - h % 2)  # H.264 needs even dimensions
        writer = imageio_ffmpeg.write_frames(out_path, size, fps=self.fps, codec="libx264",
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

    def _write_gif(self, paths, out_path):
        """Every ``gif_every``-th frame, scaled to ``gif_width``, always ending on the last."""
        picked = paths[::self.gif_every]
        if picked[-1] != paths[-1]:
            picked = picked + [paths[-1]]
        first = _load_rgb(picked[0])
        size = (self.gif_width, round(first.height * self.gif_width / first.width))
        frames = [_load_rgb(p).resize(size, Image.LANCZOS) for p in picked]
        durations = ([round(1000 * self.gif_every / self.fps)] * (len(frames) - 1)
                     + [round(1000 * self.hold_last)])
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=True)


def _frame_number(name):
    match = re.match(r"\d+", Path(name).name)
    return int(match.group()) if match else None


def _load_rgb(path, size=None):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    return rgb.crop((0, 0, *size)) if size else rgb
