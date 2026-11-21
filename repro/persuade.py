"""PERSUADE 2.0 essays: the human baseline and the essay assignments (App. A.4)."""

from pathlib import Path

import pandas as pd

# The seven independent-writing assignments kept in App. A.4.
PERSUADE_PROMPTS = [
    "Cell phones at school",
    "Community service",
    "Distance learning",
    "Grades for extracurricular activities",
    "Mandatory extracurricular activities",
    "Seeking multiple opinions",
    "Summer projects",
]
REQUIRED_METADATA = ("grade_level", "ell_status", "holistic_essay_score")
MIN_HUMAN_SCORE = 5  # student essays scored 5-6 form the human baseline


def load_persuade(path: str | Path) -> pd.DataFrame:
    """Essays on the seven assignments that have all metadata columns filled in."""
    essays = pd.read_csv(path)
    essays = essays[essays.prompt_name.isin(PERSUADE_PROMPTS)]
    present = [column for column in REQUIRED_METADATA if column in essays]
    return essays.dropna(subset=present)


def human_baseline_essays(path: str | Path) -> pd.DataFrame:
    """Student essays scored high enough to serve as the human baseline."""
    essays = load_persuade(path)
    return essays[essays.holistic_essay_score >= MIN_HUMAN_SCORE]
