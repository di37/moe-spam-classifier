"""Where the project reads its data and writes its outputs.

Every path is relative to the current directory, so run the commands from the repository
root. Keeping them relative also keeps home directories out of the run records.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """The project's input file and output folders.

    Attributes:
        data_file: The SMS Spam Collection, as unzipped from the UCI download.
        results: Run records (CSV and JSON).
        checkpoints: Trained models.
        frames: Rendered video frames, one folder per video.
        media: Finished videos and GIFs.
    """

    data_file: Path = Path("data/raw/sms/SMSSpamCollection")
    results: Path = Path("results")
    checkpoints: Path = Path("checkpoints")
    frames: Path = Path("frames")
    media: Path = Path("media")

    def checkpoint(self, run: str) -> Path:
        """Returns the checkpoint file of a run, e.g. ``checkpoints/moe.pt``."""
        return self.checkpoints / f"{run}.pt"

    def frame_dir(self, video: str) -> Path:
        """Returns the folder that holds a video's frames."""
        return self.frames / video

    def video(self, video: str) -> Path:
        """Returns a video's output path without extension (``.mp4`` and ``.gif`` are added)."""
        return self.media / video


PATHS = ProjectPaths()
