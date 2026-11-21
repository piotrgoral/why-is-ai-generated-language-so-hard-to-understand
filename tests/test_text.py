from repro import text


def test_clean_markdown():
    t = "# Title\n\nSee [docs](http://x.com) and `code` and **bold**.\n- item one\n1. item two\n| a | b |"
    blocks = text.clean(t)
    assert blocks == ["See docs and Xyz and bold.", "item one", "item two"]


def test_clean_drops_think_and_code_fences():
    assert text.clean("hidden</think>Hello there.\n```\nx=1\n```") == ["Hello there."]


def test_clean_essay_standin_and_title():
    t = "# My Essay\n\nDear [Name], hello.\n"
    assert text.clean(t, essay=True) == ["Dear Sam, hello."]


def test_clean_essay_keeps_longest_part_around_rule():
    t = "Note to reader\n---\nThis is the real essay body with many words.\n---\nbye"
    assert text.clean(t, essay=True) == ["This is the real essay body with many words."]


def test_clean_student_drops_leading_short_lines():
    t = "Jane\nEnglish\nMay 5\nThis is a long enough opening sentence for the essay."
    assert text.clean(t, student=True) == [
        "This is a long enough opening sentence for the essay."
    ]


def test_sentences_split():
    assert text.sentences(["Hello there. How are you?", "Fine."]) == [
        "Hello there.",
        "How are you?",
        "Fine.",
    ]


def test_clean_keeps_currency_masks_math():
    assert text.clean("Rent is between $500 and $1,000.") == ["Rent is between $500 and $1,000."]
    assert text.clean("Let $x$ be small.") == ["Let Xyz be small."]
