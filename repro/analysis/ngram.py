"""Interpolated modified Kneser-Ney trigram model and the n-gram features built on it (App. B.2)."""

import math
import re
from collections import Counter
from collections.abc import Iterable

from repro.text import PLACEHOLDER, STANDIN

START, END, UNKNOWN = "<s>", "</s>", "<unk>"
# n-grams containing these are not counted
SKIPPED_WORDS = {PLACEHOLDER.lower(), STANDIN.lower()}
ORDER = 3
MAX_DISCOUNTED_COUNT = 3  # discounts D1, D2, D3+
COUNT_OF_COUNTS_LIMIT = 4  # count-of-counts up to n4 estimate the discounts


def words(text: str) -> list[str]:
    """Lowercased words (with apostrophes) and numbers."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


class KneserNey:
    """Interpolated modified Kneser-Ney trigram model, estimated the way KenLM's lmplz does (Heafield, 2011):
    lower orders use continuation counts except for <s>-initial n-grams, which keep raw counts; three
    discounts per order (D1, D2, D3+) from count-of-counts; unigrams are interpolated with a uniform
    distribution over the vocabulary (incl. <s> and <unk>); sentences start with a single <s>."""

    def __init__(self, sentences: Iterable[list[str]]) -> None:
        self.counts = self._count_ngrams(sentences)
        self.discounts: dict[int, list[float]] = {}
        self.context_totals: dict[int, Counter[tuple[str, ...]]] = {}
        self.backoff_weights: dict[int, dict[tuple[str, ...], float]] = {}
        for order, counts in self.counts.items():
            self.discounts[order] = self._estimate_discounts(counts)
            totals, discounted = Counter(), Counter()
            for gram, count in counts.items():
                context = gram[:-1]
                totals[context] += count
                discounted[context] += self.discounts[order][
                    min(count, MAX_DISCOUNTED_COUNT)
                ]
            self.context_totals[order] = totals
            self.backoff_weights[order] = {
                context: discounted[context] / totals[context] for context in totals
            }
        self.vocab = len(self.counts[1]) + 2  # + <s>, <unk>

    @staticmethod
    def _count_ngrams(
        sentences: Iterable[list[str]],
    ) -> dict[int, Counter[tuple[str, ...]]]:
        """Raw trigram counts and continuation counts for the lower orders."""
        counts: dict[int, Counter[tuple[str, ...]]] = {
            order: Counter() for order in range(1, ORDER + 1)
        }
        for sentence in sentences:
            if not sentence:
                continue
            padded = [START, *sentence, END]
            counts[3].update(zip(padded, padded[1:], padded[2:]))
            counts[2][START, sentence[0]] += 1  # raw count: nothing can precede <s>
        for _, v, w in counts[3]:
            counts[2][v, w] += 1  # number of distinct left extensions
        for _, w in counts[2]:
            counts[1][(w,)] += 1
        return counts

    @staticmethod
    def _estimate_discounts(counts: Counter[tuple[str, ...]]) -> list[float]:
        """[0, D1, D2, D3+] from the count-of-counts n1..n4."""
        n = Counter(c for c in counts.values() if c <= COUNT_OF_COUNTS_LIMIT)
        y = n[1] / (n[1] + 2 * n[2]) if n[1] + n[2] else 0
        return [0] + [
            k - (k + 1) * y * n[k + 1] / n[k] if n[k] else 0
            for k in range(1, MAX_DISCOUNTED_COUNT + 1)
        ]

    def p(self, context: tuple[str, ...], word: str) -> float:
        """P(word | context) for a context of 0-2 words."""
        probability = 1 / self.vocab
        for order in range(1, len(context) + 2):
            history = context[len(context) - order + 1 :]
            if total := self.context_totals[order][history]:
                count = self.counts[order][(*history, word)]
                discount = self.discounts[order][min(count, MAX_DISCOUNTED_COUNT)]
                probability = (count - discount) / total + (
                    self.backoff_weights[order][history] * probability
                )
        return probability

    def surprisal(self, sentence: list[str]) -> tuple[float, int]:
        """(sum of bits, number of scored words incl. </s>) over trigrams without placeholder words."""
        padded = [START, *sentence, END]
        bits = [
            -math.log2(self.p(tuple(padded[max(0, i - 2) : i]), padded[i]))
            for i in range(1, len(padded))
            if not SKIPPED_WORDS & set(padded[max(0, i - 2) : i + 1])
        ]
        return sum(bits), len(bits)


def ngram_features(
    text: str, model: KneserNey, bigrams: set[tuple[str, str]]
) -> dict[str, float]:
    """Word, bigram and trigram-surprisal counts of `text` against a reference corpus."""
    tokens = words(text)
    pairs = [p for p in zip(tokens, tokens[1:]) if not SKIPPED_WORDS & set(p)]
    bits, trigrams = model.surprisal(tokens)
    return {
        "words": sum(t not in SKIPPED_WORDS for t in tokens),
        "pairs": len(pairs),
        "unattested": sum(p not in bigrams for p in pairs),
        "bits": bits,
        "trigrams": trigrams,
    }
