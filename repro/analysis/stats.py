"""Bootstrap confidence intervals and the paired jackknife test over units (App. B.4)."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from repro.analysis.metrics import COUNTS, metrics

BOOTSTRAP_DRAWS = 1000
LOWER_QUANTILE, UPPER_QUANTILE = 0.025, 0.975  # 95% interval
BOOTSTRAP_SEED = 0


def bootstrap(
    units: pd.DataFrame,
    bin_weights: list[float],
    strata: npt.NDArray,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.Series, pd.Series]:
    """95% CI resampling whole units (conversations; essays within prompt)."""
    rng = np.random.default_rng(seed)
    unit_counts = units[list(COUNTS)].to_numpy(float)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = [
        metrics(
            unit_counts[np.concatenate([rng.choice(g, len(g)) for g in groups])].sum(0),
            bin_weights,
        )
        for _ in range(n_draws)
    ]
    draws = pd.DataFrame(draws)
    return draws.quantile(LOWER_QUANTILE), draws.quantile(UPPER_QUANTILE)


def paired_jackknife(
    orig: pd.DataFrame, rt: pd.DataFrame, bin_weights: list[float]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Two-sided paired jackknife t-test over units of the pooled orig - rt difference.

    Returns the differences, their t statistics and p-values per metric."""
    ids = sorted(set(orig.index) & set(rt.index))
    orig_counts = orig.loc[ids, list(COUNTS)].to_numpy(float)
    rt_counts = rt.loc[ids, list(COUNTS)].to_numpy(float)

    def difference(orig_total: npt.NDArray, rt_total: npt.NDArray) -> pd.Series:
        return pd.Series(metrics(orig_total, bin_weights)) - pd.Series(
            metrics(rt_total, bin_weights)
        )

    orig_total, rt_total = orig_counts.sum(0), rt_counts.sum(0)
    full = difference(orig_total, rt_total)
    leave_one_out = pd.DataFrame(
        [
            difference(orig_total - orig_counts[i], rt_total - rt_counts[i])
            for i in range(len(ids))
        ]
    )
    n = len(ids)
    standard_error = np.sqrt(
        (n - 1) / n * ((leave_one_out - leave_one_out.mean()) ** 2).sum()
    )
    t = full / standard_error.replace(
        0, np.nan
    )  # one unit (or no variation) -> no test
    p = pd.Series(2 * stats.t.sf(t.abs(), n - 1), index=t.index)
    return full, t, p
