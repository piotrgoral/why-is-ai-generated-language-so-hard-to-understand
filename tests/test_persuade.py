from repro import persuade


def test_load_persuade_filters(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "prompt_name,grade_level,ell_status,holistic_essay_score\n"
        "Summer projects,10,No,5\n"
        "Other prompt,10,No,5\n"
        "Summer projects,,No,5\n"
    )
    df = persuade.load_persuade(p)
    assert len(df) == 1 and df.iloc[0].prompt_name == "Summer projects"
