"""What every video shares: its timeline of beats, its frame files, and the render loop."""

import dataclasses
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from moe_spam.paths import PATHS
from moe_spam.viz.canvas import Canvas
from moe_spam.viz.encoder import FPS, FRAME_EXTENSIONS, VideoEncoder

# The project's videos: wide enough GIFs to read the token panels, every 2nd frame.
VIDEO_ENCODER = VideoEncoder(fps=FPS, gif_width=800, gif_every=2)


def ease(t):
    """Returns ``t`` clamped to [0, 1] and eased in and out (a smoothstep)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


@dataclass(frozen=True)
class Beat:
    """One step of a video's script.

    Attributes:
        step: The name of the step, which picks what is drawn.
        seconds: How long the step animates.
        hold: How long its last frame then stays on screen.
    """

    step: str
    seconds: float
    hold: float

    def frames(self, fps=FPS):
        """Returns ``(animated frames, held frames)`` at ``fps``; at least one animated frame."""
        return max(1, round(self.seconds * fps)), round(self.hold * fps)


def numbered_titles(beats, names):
    """Returns each step's title, numbered in the order ``beats`` first show it.

    Steps with the same name share a number.

    Args:
        beats: The script, in order.
        names: ``{step: name}``.

    Returns:
        ``{step: "<number>. <name>"}`` for every step in ``beats``.
    """
    numbers = {}
    for beat in beats:
        numbers.setdefault(names[beat.step], len(numbers) + 1)
    return {beat.step: f"{numbers[names[beat.step]]}. {names[beat.step]}" for beat in beats}


class FrameWriter:
    """The numbered JPEG files of one video.

    Attributes:
        frame_dir: The folder of the frames.
    """

    def __init__(self, frame_dir):
        """Creates a writer for the frames in ``frame_dir``."""
        self.frame_dir = Path(frame_dir)

    def clear(self):
        """Creates the folder, and deletes frames left from an earlier run (nothing else)."""
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        for ext in FRAME_EXTENSIONS:
            for old in self.frame_dir.glob(f"*{ext}"):
                old.unlink()

    def path(self, number):
        """Returns the path of frame ``number`` (a step, or a running index)."""
        return self.frame_dir / f"{number:07d}.jpg"

    def hold(self, last, count):
        """Repeats frame ``last`` as the next ``count`` frames; returns the next free number."""
        for number in range(last + 1, last + 1 + count):
            shutil.copyfile(self.path(last), self.path(number))
        return last + 1 + count


class Video(ABC):
    """A scripted video: segments of beats, each frame drawn on its own ``Canvas``.

    Subclasses say what the segments are and how to draw one frame; this class runs the
    timeline, writes the frames and encodes the video (the Template Method pattern).

    Attributes:
        name: The video's name: its frames go to ``frames/<name>/``, the video to
            ``media/<name>.mp4`` and ``.gif``.
        paths: Where the frames and the video go.
        fps: Frames per second.
    """

    name = "video"

    def __init__(self, paths=PATHS, fps=FPS):
        """Creates the video, writing under ``paths``, at ``fps`` frames per second."""
        self.paths = paths
        self.fps = fps

    @abstractmethod
    def segments(self):
        """Returns ``[(context, beats)]``: the parts of the video, in order.

        The context is handed to ``draw`` with each of the part's frames.
        """

    @abstractmethod
    def draw(self, canvas, context, beat, t):
        """Draws one frame.

        Args:
            canvas: The frame's ``Canvas``.
            context: The context of the current segment.
            beat: The current ``Beat``.
            t: Progress through the beat's animation, in (0, 1].
        """

    def check(self):
        """Raises if the video cannot be drawn; runs before any old frame is deleted.

        The base video has nothing to check; subclasses override this.
        """
        return None

    def render(self, frame_dir=None):
        """Draws every frame; returns how many there are.

        Args:
            frame_dir: Where the frames go; ``frames/<name>/`` by default.
        """
        self.check()
        writer = FrameWriter(frame_dir or self.paths.frame_dir(self.name))
        writer.clear()
        number = 0
        for context, beats in self.segments():
            for beat in beats:
                frames, hold = beat.frames(self.fps)
                for i in range(frames):
                    with Canvas(writer.path(number)) as canvas:
                        self.draw(canvas, context, beat, (i + 1) / frames)
                    number += 1
                number = writer.hold(number - 1, hold)
        return number

    def encode(self, frame_dir=None, out_base=None):
        """Turns the rendered frames into a video and a GIF, at this video's ``fps``.

        Args:
            frame_dir: Where the frames are; ``frames/<name>/`` by default.
            out_base: The output path without extension; ``media/<name>`` by default.
        """
        dataclasses.replace(VIDEO_ENCODER, fps=self.fps).encode(
            frame_dir or self.paths.frame_dir(self.name), out_base or self.paths.video(self.name))
