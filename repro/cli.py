"""Argument handling shared by the stage entry points."""

import argparse
import logging

from repro.experiment import Experiment

DEFAULT_WORKERS = 32


def add_workers_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="conversations collected in parallel",
    )


def apply_overrides(
    experiment: Experiment, args: argparse.Namespace, *names: str
) -> None:
    """Copy each command-line option in `names` that was given onto the experiment."""
    for name in names:
        value = getattr(args, name)
        if value:
            setattr(experiment, name, value)


def configure_logging() -> None:
    """Plain key=value log lines on stderr; call once from each entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
