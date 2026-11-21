"""Text cleaning and sentence splitting (simplified App. A.5)."""

import re

from syntok import segmenter

PLACEHOLDER = "Xyz"  # code/URL/math span
STANDIN = "Sam"  # bracketed names in essays

MAX_STUDENT_HEADER_LINES = 4  # title/name/date lines before a student essay
MAX_HEADER_LINE_WORDS = 5  # a header line has fewer than 6 words
HORIZONTAL_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", flags=re.M)
LIST_MARKER = re.compile(r"^(?:>\s*)*(?:[-*+]|\d+[.)])\s+")
NOT_SENTENCES = ("#", "|")  # headings and table rows


def _strip_markup(text: str) -> str:
    """Remove model reasoning, code, images, links and emphasis; mask code/URL/math spans."""
    text = text.split("</think>")[-1]
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`[^`\n]+`|https?://\S+|\$[^$\n]+\$", PLACEHOLDER, text)
    text = re.sub(r"(\*\*|__|\*)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    text = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", text)
    return re.sub(r"[→←↔⇒➜]|[\U0001F000-\U0001FAFF☀-➿]", "\n", text)


def _mask_essay_names(text: str) -> str:
    """Bracketed names become a stand-in; text set apart by a horizontal rule (title
    block, note to reader) is dropped: keep the longest part."""
    text = re.sub(r"Generic_\w+|\[[^\]\n]{1,40}\]", STANDIN, text)
    return max(HORIZONTAL_RULE.split(text), key=len)


def _drop_student_header(lines: list[str]) -> list[str]:
    """Up to four leading title/name/date lines."""
    lines = [line for line in lines if line]
    for _ in range(MAX_STUDENT_HEADER_LINES):
        if lines and len(lines[0].split()) <= MAX_HEADER_LINE_WORDS:
            lines = lines[1:]
    return lines


def clean(text: str, essay: bool = False, student: bool = False) -> list[str]:
    """Markdown/plain text -> list of blocks (paragraphs, list items)."""
    text = _strip_markup(text)
    if essay:
        text = _mask_essay_names(text)
    lines = [line.strip() for line in text.splitlines()]
    if essay and lines and lines[0].startswith("#"):
        lines = lines[1:]  # leading heading is the title
    if student:
        lines = _drop_student_header(lines)
    return [
        LIST_MARKER.sub("", line)
        for line in lines
        if line and not line.startswith(NOT_SENTENCES)
    ]


def sentences(blocks: list[str]) -> list[str]:
    """Split each block into sentences with syntok; spans are cut from the original text."""
    out = []
    for block in blocks:
        for paragraph in segmenter.process(block):
            for sentence in paragraph:
                start = sentence[0].offset
                end = sentence[-1].offset + len(sentence[-1].value)
                if text := block[start:end].strip():
                    out.append(text)
    return out
