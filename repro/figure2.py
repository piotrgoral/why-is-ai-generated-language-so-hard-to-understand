"""Figure 2: change under round-trip translation, by model, in standard deviations of the original values
across models (App. B.4). For each measure and setting, |round trip - original| is divided by the SD of the
original values across all models passed in; the result is averaged within each of the four measure pairs of
Figure 6, then across pairs; Overall is the mean over the settings the model was run in. Bars: 95% bootstrap
intervals resampling whole units (conversations; essays within prompt), with the SDs held at their point values.

The SD is taken across the models given, so pass every experiment that should share one scale:
    uv run python -m repro.figure2 experiments/haiku-roundtrip.toml experiments/sonnet-roundtrip.toml
Reads <results>/units.csv, written by repro.analyze."""

import argparse
import json

import numpy as np
import numpy.typing as npt
import pandas as pd

from repro import experiment
from repro.analysis.metrics import COUNTS, metrics
from repro.analysis.stats import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    LOWER_QUANTILE,
    UPPER_QUANTILE,
)
from repro.analysis.syntax import REFERENCE_STATS

# the four measure pairs of Figure 6
PAIRS = {
    "n-grams": ["unattested_per_1k_pairs", "trigram_surprisal_bits"],
    "dependencies": ["dep_length_std", "stacked_nouns_per_1k_words"],
    "relatives": ["orc_no_that_per_1k_words", "orc_that_per_1k_words"],
    "verbs": ["denominal_per_1k_words", "inanimate_subj_per_1k_words"],
}
MEASURES = [m for ms in PAIRS.values() for m in ms]
ALL_PAIRS = "all pairs"
MIN_MODELS = 3  # fewer models than this -> warn that the SD scale is unstable

Paired = dict[tuple[str, str], tuple[npt.NDArray, npt.NDArray, npt.NDArray]]


def paired(units: pd.DataFrame) -> Paired:
    """(setting, model) -> (original counts, round-trip counts, bootstrap strata), one row per shared unit."""
    out: Paired = {}
    machine = units[units.source.isin(["orig", "rt"])]
    for (setting, model), u in machine.groupby(["setting", "model"]):
        o, r = (u[u.source == s].set_index("unit_id") for s in ("orig", "rt"))
        ids = sorted(set(o.index) & set(r.index))
        strata = (
            o.loc[ids, "prompt"].to_numpy()
            if setting == "essay"
            else np.zeros(len(ids))
        )
        out[setting, model] = (
            o.loc[ids, list(COUNTS)].to_numpy(float),
            r.loc[ids, list(COUNTS)].to_numpy(float),
            strata,
        )
    return out


def pair_scores(
    orig: npt.NDArray, rt: npt.NDArray, w: list[float], sd: pd.Series
) -> dict[str, float]:
    """|round trip - original| / SD per measure, averaged within each pair, then across pairs."""
    mo, mr = pd.Series(metrics(orig.sum(0), w)), pd.Series(metrics(rt.sum(0), w))
    z = (mr - mo).abs() / sd
    scores = {p: float(np.nanmean(z[ms])) for p, ms in PAIRS.items()}
    return scores | {ALL_PAIRS: float(np.nanmean(list(scores.values())))}


def figure2(
    units: pd.DataFrame,
    w: list[float],
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows (setting, model, group, value, lo, hi, units), incl. the 'overall' setting; and per setting
    the SDs used, with the number of models they span."""
    data = paired(units)
    orig = pd.DataFrame({k: metrics(o.sum(0), w) for k, (o, _, _) in data.items()}).T[
        MEASURES
    ]
    sd = orig.groupby(level=0).std()
    sd["n_models"] = orig.groupby(level=0).size()
    for setting, n in sd.n_models.items():
        if n < MIN_MODELS:
            print(
                f"warning: {setting}: SD across only {n} models; the scale is unstable"
            )

    rng = np.random.default_rng(seed)
    rows = []
    for (setting, model), (o, r, strata) in data.items():
        point = pair_scores(o, r, w, sd.loc[setting, MEASURES])
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
        draws = []
        for _ in range(n_draws):
            i = np.concatenate([rng.choice(g, len(g)) for g in groups])
            draws.append(pair_scores(o[i], r[i], w, sd.loc[setting, MEASURES]))
        draws = pd.DataFrame(draws)
        rows += [
            {
                "setting": setting,
                "model": model,
                "group": group,
                "value": value,
                "lo": draws[group].quantile(LOWER_QUANTILE),
                "hi": draws[group].quantile(UPPER_QUANTILE),
                "units": len(o),
            }
            for group, value in point.items()
        ]
    df = pd.DataFrame(rows)
    overall = (
        df[df.group == ALL_PAIRS]
        .groupby("model")[["value", "lo", "hi"]]
        .mean()
        .reset_index()
        .assign(setting="overall", group=ALL_PAIRS)
    )
    return pd.concat([df, overall], ignore_index=True), sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "configs",
        nargs="+",
        help="experiment TOMLs; the SD is taken across all their models",
    )
    ap.add_argument("--boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--out", default="runs/figure2.csv")
    a = ap.parse_args()

    units = pd.concat(
        [pd.read_csv(experiment.load(c).results / "units.csv") for c in a.configs]
    )
    w = json.loads(REFERENCE_STATS.read_text())["bin_weights"]
    df, sd = figure2(units, w, a.boot, a.seed)
    df.to_csv(a.out, index=False)

    cells = df.assign(
        cell=[
            f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in zip(df.value, df.lo, df.hi)
        ]
    )
    n_models = ", ".join(f"{s}: {n}" for s, n in sd.n_models.items())
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"\nSD of original values across models ({n_models}):")
        print(sd[MEASURES].T.round(3))
        print("\n|round trip - original| in SDs (95% bootstrap interval)")
        print(
            cells.pivot_table(
                index=["setting", "model"],
                columns="group",
                values="cell",
                aggfunc="first",
            )[list(PAIRS) + [ALL_PAIRS]].fillna("")
        )
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
