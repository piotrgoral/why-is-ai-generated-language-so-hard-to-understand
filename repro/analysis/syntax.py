"""Per-sentence features from the dependency parse (App. B.1, B.3) and the cache of parsed sentences."""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Token

from repro.jsonl import Record, append_jsonl, read_jsonl
from repro.paths import DATA
from repro.text import PLACEHOLDER

logger = logging.getLogger(__name__)

FEATURE_CACHE = DATA / "features.jsonl"
REFERENCE_STATS = DATA / "ref_stats.json"

BINS = 5  # sentence-length bins 1-5, ..., 21-30 words (App. B.3)
BIN_WIDTH = 5
MAX_SENTENCE_WORDS = 30
RELATIVIZERS = {"that", "which", "who", "whom"}
NON_OBJECT_RELATIVIZERS = {"where", "when", "why", "how", "whose"}
ANIMATE_PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "who",
    "someone",
    "somebody",
    "everyone",
    "people",
}
INANIMATE_PRONOUNS = {"it", "this", "that"}
STACKED_NOUN_MIN_COMPOUNDS = 2  # >= 3 nouns in total
DENOMINAL_NOUN_SHARE = (
    0.5  # a verb lemma is a denominal candidate if mostly used as a noun
)
MIN_LEMMA_OCCURRENCES = 3  # lemmas rarer than this get no noun share
PARSE_CHUNK = 2000
PARSE_BATCH = 64

Candidate = tuple[str, str]  # (feature, marked sentence)


def dep_length(doc: Doc) -> tuple[int | None, int, int]:
    """(length bin, summed arc length, number of arcs); punctuation arcs and the root are excluded."""
    n_words = sum(not t.is_punct and not t.is_space for t in doc)
    if not 1 <= n_words <= MAX_SENTENCE_WORDS:
        return None, 0, 0
    arcs = [
        abs(t.i - t.head.i)
        for t in doc
        if t.dep_ != "ROOT"
        and not t.is_punct
        and not t.head.is_punct
        and not t.is_space
    ]
    length_bin = min((n_words - 1) // BIN_WIDTH, BINS - 1)  # last bin is 21-30
    return length_bin, sum(arcs), len(arcs)


def object_relatives(doc: Doc) -> tuple[int, int]:
    """(bare, overt) relative clauses whose verb has its own (non-relativizer) subject
    and no other direct object."""
    bare = overt = 0
    for token in doc:
        if token.dep_ != "relcl":
            continue
        children = list(token.children)
        relativizers = [
            k for k in children if k.lower_ in RELATIVIZERS or k.tag_ in ("WDT", "WP")
        ]
        if any(k.lower_ in NON_OBJECT_RELATIVIZERS for k in children):
            continue
        has_own_subject = any(
            k.dep_ == "nsubj" and k not in relativizers for k in children
        )
        has_other_object = any(
            k.dep_ in ("dobj", "nsubjpass") and k not in relativizers for k in children
        )
        if has_own_subject and not has_other_object:
            overt += bool(relativizers)
            bare += not relativizers
    return bare, overt


def stacked_nouns(doc: Doc) -> int:
    """Noun heads preceded by >= 2 adjacent noun compound modifiers (>= 3 nouns)."""
    count = 0
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and token.dep_ != "compound":
            i = token.i - 1
            while (
                i >= 0
                and doc[i].dep_ == "compound"
                and doc[i].pos_ in ("NOUN", "PROPN")
                and token.i >= doc[i].head.i > i
            ):
                i -= 1
            count += token.i - 1 - i >= STACKED_NOUN_MIN_COMPOUNDS
    return count


def mark(doc: Doc, verb: Token, subject: Token | None = None) -> str:
    """Sentence with the verb as [[verb]] and its subject as {{subject}} (spaCy tokens compare by index)."""
    subject_index = subject.i if subject is not None else -1

    def render(token: Token) -> str:
        if token.i == subject_index:
            return f"{{{{{token.text}}}}}{token.whitespace_}"
        if token.i == verb.i:
            return f"[[{token.text}]]{token.whitespace_}"
        return token.text_with_ws

    return "".join(render(token) for token in doc)


def _is_inanimate_subject(subject: Token) -> bool:
    return (
        subject.lower_ in INANIMATE_PRONOUNS
        or subject.text == PLACEHOLDER
        or (
            subject.pos_ in ("NOUN", "PROPN")
            and subject.ent_type_ != "PERSON"
            and subject.lower_ not in ANIMATE_PRONOUNS
        )
    )


def candidates(doc: Doc, noun_share: dict[str, float]) -> list[Candidate]:
    """Verbs that may be denominal, and active verbs with an inanimate subject, to be judged by Jev."""
    found: list[Candidate] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        if noun_share.get(token.lemma_.lower(), 0) > DENOMINAL_NOUN_SHARE:
            found.append(("denominal", mark(doc, token)))
        if any(k.dep_ in ("auxpass", "nsubjpass") for k in token.children):
            continue
        for subject in (k for k in token.children if k.dep_ == "nsubj"):
            if _is_inanimate_subject(subject):
                found.append(("inanimate", mark(doc, token, subject)))
    return found


def sentence_features(doc: Doc, noun_share: dict[str, float]) -> Record:
    length_bin, dep_sum, dep_count = dep_length(doc)
    bare, overt = object_relatives(doc)
    return {
        "dl_bin": length_bin,
        "dl_sum": dep_sum,
        "dl_n": dep_count,
        "orc_bare": bare,
        "orc_that": overt,
        "stacked": stacked_nouns(doc),
        "cands": candidates(doc, noun_share),
    }


def featurize(
    texts: Iterable[str],
    nlp: Language,
    noun_share: dict[str, float],
    cache_path: Path = FEATURE_CACHE,
) -> dict[str, Record]:
    """Parse-based features for every distinct sentence, cached in `cache_path`."""
    texts = list(texts)

    def cache_key(text: str) -> str:
        return hashlib.md5(f"{nlp.meta['name']}|{text}".encode()).hexdigest()

    cache = {row["h"]: row for row in read_jsonl(cache_path)}
    todo = sorted({t for t in texts if cache_key(t) not in cache})
    for start in range(0, len(todo), PARSE_CHUNK):
        chunk = todo[start : start + PARSE_CHUNK]
        rows = [
            {"h": cache_key(text)} | sentence_features(doc, noun_share)
            for text, doc in zip(chunk, nlp.pipe(chunk, batch_size=PARSE_BATCH))
        ]
        append_jsonl(cache_path, rows)
        cache.update((row["h"], row) for row in rows)
        logger.info("parsed=%d total=%d sentences", start + len(chunk), len(todo))
    return {text: cache[cache_key(text)] for text in set(texts)}


def reference_stats(
    nlp: Language,
    reference: list[str],
    parse_words: int,
    cache_path: Path = REFERENCE_STATS,
) -> dict[str, Any]:
    """Noun share per lemma (denominal candidates) and Wikipedia arc weights per length bin (App. B.3)."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    subset, n_words = [], 0
    for sentence in reference:
        subset.append(sentence)
        n_words += len(sentence.split())
        if n_words >= parse_words:
            break
    pos_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    arcs_per_bin = np.zeros(BINS)
    for doc in nlp.pipe(subset, batch_size=PARSE_BATCH):
        for token in doc:
            if token.pos_ in ("NOUN", "VERB"):
                pos_counts[token.lemma_.lower()][token.pos_] += 1
        length_bin, _, n_arcs = dep_length(doc)
        if length_bin is not None:
            arcs_per_bin[length_bin] += n_arcs
    stats = {
        "noun_share": {
            lemma: counts["NOUN"] / counts.total()
            for lemma, counts in pos_counts.items()
            if counts.total() >= MIN_LEMMA_OCCURRENCES
        },
        "bin_weights": (arcs_per_bin / arcs_per_bin.sum()).tolist(),
    }
    cache_path.write_text(json.dumps(stats))
    return stats
