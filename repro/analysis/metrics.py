"""Per-unit counts and the difficulty metrics computed from pooled counts (App. B)."""

import numpy as np
import numpy.typing as npt

from repro.analysis.syntax import BINS

NGRAM_COUNTS = ("words", "pairs", "unattested", "bits", "trigrams")
SYNTAX_COUNTS = ("orc_bare", "orc_that", "stacked")
JUDGED_COUNTS = ("denominal", "inanimate")
DEP_LENGTH_SUM_COUNTS = tuple(f"dl_sum{b}" for b in range(BINS))
DEP_LENGTH_ARC_COUNTS = tuple(f"dl_n{b}" for b in range(BINS))
# column order of units.csv
COUNTS = (
    *NGRAM_COUNTS,
    *SYNTAX_COUNTS,
    *JUDGED_COUNTS,
    *DEP_LENGTH_SUM_COUNTS,
    *DEP_LENGTH_ARC_COUNTS,
)
PER_THOUSAND = 1000


def metrics(count_values: npt.ArrayLike, bin_weights: list[float]) -> dict[str, float]:
    """The eight metrics from pooled counts, given in the order of COUNTS."""
    counts = dict(zip(COUNTS, count_values))

    def per_thousand(numerator: str, denominator: str) -> float:
        if not counts[denominator]:
            return np.nan
        return PER_THOUSAND * counts[numerator] / counts[denominator]

    def per_thousand_words(numerator: str) -> float:
        return per_thousand(numerator, "words")

    bins_present = [b for b in range(BINS) if counts[f"dl_n{b}"]]
    if bins_present:
        dep_length_std = sum(
            bin_weights[b] * counts[f"dl_sum{b}"] / counts[f"dl_n{b}"]
            for b in bins_present
        ) / sum(bin_weights[b] for b in bins_present)
    else:
        dep_length_std = np.nan
    return {
        "unattested_per_1k_pairs": per_thousand("unattested", "pairs"),
        "trigram_surprisal_bits": (
            counts["bits"] / counts["trigrams"] if counts["trigrams"] else np.nan
        ),
        "dep_length_std": dep_length_std,
        "orc_no_that_per_1k_words": per_thousand_words("orc_bare"),
        "orc_that_per_1k_words": per_thousand_words("orc_that"),
        "stacked_nouns_per_1k_words": per_thousand_words("stacked"),
        "denominal_per_1k_words": per_thousand_words("denominal"),
        "inanimate_subj_per_1k_words": per_thousand_words("inanimate"),
    }
