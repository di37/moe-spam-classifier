"""The ``moe-spam`` command: one subcommand per task. Run it from the repository root."""

import argparse
import sys

from moe_spam.cli.lesson_commands import (
    BalancingVideoCommand,
    ExercisesVideoCommand,
    WhyMoeVideoCommand,
)
from moe_spam.cli.training_commands import BaselinesCommand, BenchmarkCommand, TrainCommand
from moe_spam.cli.video_commands import (
    ArchitectureVideoCommand,
    EncodeVideoCommand,
    HowItWorksVideoCommand,
    InferenceVideoCommand,
    ModelDiagramVideoCommand,
)
from moe_spam.errors import MoeSpamError

COMMANDS = (TrainCommand(), BaselinesCommand(), BenchmarkCommand(), ArchitectureVideoCommand(),
            InferenceVideoCommand(), HowItWorksVideoCommand(), ModelDiagramVideoCommand(),
            WhyMoeVideoCommand(), BalancingVideoCommand(), ExercisesVideoCommand(),
            EncodeVideoCommand())


def build_parser():
    """Returns the parser of ``moe-spam`` and its subcommands.

    Returns:
        ``(parser, {name: (command, sub-parser)})``.
    """
    parser = argparse.ArgumentParser(
        prog="moe-spam",
        description="A tiny BERT with mixture-of-experts layers that tells SMS spam from ham, "
                    "and videos of how it trains, routes tokens and runs.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands = {}
    for command in COMMANDS:
        sub = subparsers.add_parser(command.name, help=command.help, description=command.help)
        command.add_arguments(sub)
        commands[command.name] = (command, sub)
    return parser, commands


def run(argv=None):
    """Parses ``argv`` (the command line by default) and runs the chosen subcommand.

    Returns:
        What the subcommand returned (``train`` returns the final metrics).
    """
    parser, commands = build_parser()
    args = parser.parse_args(argv)
    command, sub = commands[args.command]
    return command.run(args, sub)


def main(argv=None):
    """The ``moe-spam`` entry point: runs the subcommand and returns its exit status.

    Returns:
        0 on success; 1 after printing the message of a ``MoeSpamError``, a problem the user
        can fix. Usage errors exit with status 2, as ``argparse`` does.
    """
    try:
        run(argv)
    except MoeSpamError as error:
        print(f"moe-spam: error: {error}", file=sys.stderr)
        return 1
    return 0
