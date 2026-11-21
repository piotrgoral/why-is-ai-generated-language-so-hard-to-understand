"""Writing the result tables and printing the original-vs-round-trip summary."""

from pathlib import Path

import pandas as pd

SIGNIFICANCE_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
DISPLAY_WIDTH, DISPLAY_COLUMNS = 250, 20


def significance_stars(p: float) -> str:
    return next((stars for level, stars in SIGNIFICANCE_LEVELS if p < level), "")


def write_results(
    directory: Path,
    units: pd.DataFrame,
    pooled: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    """units.csv (per-unit counts), features.csv (pooled metrics) and diff.csv."""
    directory.mkdir(parents=True, exist_ok=True)
    units.to_csv(directory / "units.csv", index=False)
    pooled.to_csv(directory / "features.csv", index=False)
    differences.to_csv(directory / "diff.csv", index=False)


def print_differences(name: str, differences: pd.DataFrame) -> None:
    """Table of original - round trip per model and metric, starred by p-value."""
    cells = [
        f"{difference:+.2f}{significance_stars(p)}"
        for difference, p in zip(differences.orig_minus_rt, differences.p)
    ]
    table = differences.assign(cell=cells).pivot_table(
        index=["setting", "model"], columns="metric", values="cell", aggfunc="first"
    )
    with pd.option_context(
        "display.width", DISPLAY_WIDTH, "display.max_columns", DISPLAY_COLUMNS
    ):
        print(
            f"\n[{name}] original - round trip (pooled; * p<.05, ** p<.01, *** p<.001)"
        )
        print(table)
