"""Argument types and options shared by the ``moe-spam`` commands."""

import argparse
import math
from abc import ABC, abstractmethod

from moe_spam.viz.encoder import FPS


class Command(ABC):
    """One ``moe-spam`` subcommand: it declares its options and runs itself.

    Attributes:
        name: The subcommand, as typed.
        help: One line on what it does.
    """

    name = ""
    help = ""

    @abstractmethod
    def add_arguments(self, parser):
        """Adds the command's options to its ``argparse`` sub-parser."""

    @abstractmethod
    def run(self, args, parser):
        """Runs the command.

        Args:
            args: The parsed command line.
            parser: The command's sub-parser, to report usage errors with.

        Returns:
            Whatever is useful to a caller in Python (``train`` returns the final metrics);
            the command line ignores it.
        """


def positive_int(text):
    """``argparse`` type for counts: an int of at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def float_at_least(low, inclusive):
    """Returns an ``argparse`` type for finite floats above ``low`` (or at it, if inclusive)."""
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or value < low or (value == low and not inclusive):
            raise argparse.ArgumentTypeError(
                f"must be a number {'at least' if inclusive else 'more than'} {low}, got {text}")
        return value
    parse.__name__ = "number"  # argparse names the type in its error: "invalid number value"
    return parse


def add_fps_option(parser, default=FPS):
    """Adds ``--fps``, from 1 to 60."""
    parser.add_argument("--fps", type=int, default=default, choices=range(1, 61), metavar="FPS",
                        help=f"frames per second, 1 to 60 (default: {default})")


def add_video_options(parser, checkpoint):
    """Adds the options every video command has.

    Args:
        parser: The command's sub-parser.
        checkpoint: The default checkpoint the video is drawn from.
    """
    parser.add_argument("checkpoint", nargs="?", default=str(checkpoint),
                        help=f"trained model to draw (default: {checkpoint})")
    add_fps_option(parser)
    parser.add_argument("--no-video", action="store_true", help="only draw the frames")
