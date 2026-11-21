"""Run an experiment end to end: collect (each configured setting) -> round trip -> analyze.
Every stage is resumable, so rerunning continues where it stopped."""

import argparse
import sys

from repro import experiment
from repro.cli import add_workers_argument, apply_overrides, configure_logging

STAGES = ["collect", "roundtrip", "analyze"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="experiment TOML, e.g. experiments/opus-roundtrip.toml"
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument(
        "--limit",
        type=int,
        help="max units per model and setting (overrides the config)",
    )
    parser.add_argument(
        "--translator",
        help="override the config's round-trip translator (OpenRouter slug or facebook/nllb-*)",
    )
    parser.add_argument("--spacy", help="override the config's spaCy model")
    parser.add_argument("--no-judge", action="store_true", help="skip Jev")
    add_workers_argument(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging()
    # progress shows up in a redirected log as it happens
    sys.stdout.reconfigure(line_buffering=True)

    exp = experiment.load(args.config)
    apply_overrides(exp, args, "limit", "translator", "spacy")
    if args.no_judge:
        exp.judge = False

    # stage modules are imported lazily: roundtrip pulls in torch, analyze spaCy
    if "collect" in args.stages:
        from repro import collect

        for setting in exp.settings:
            collect.run(exp, setting, args.workers)
    if "roundtrip" in args.stages:
        from repro import roundtrip

        roundtrip.run(exp)
    if "analyze" in args.stages:
        from repro import analyze

        analyze.run(exp)


if __name__ == "__main__":
    main()
