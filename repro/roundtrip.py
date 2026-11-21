"""Split replies into sentences and paraphrase each by round-trip translation EN->JA->EN (Sec. 3.4)
-> data/sentences.jsonl. Student essays are added as source 'human' (no round trip).

The translator is an OpenRouter model (default tencent/hy-mt2-30b-a3b) or, for a `facebook/nllb-*` name,
NLLB-200 run locally as in the paper; `modal:facebook/nllb-*` runs NLLB on Modal (modal_nllb.py). See materials/translation_model/ for the comparison."""

import argparse
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from typing import Any

import pandas as pd

from repro import experiment
from repro.cli import apply_overrides, configure_logging
from repro.experiment import Experiment
from repro.jsonl import Record, append_jsonl, read_jsonl
from repro.persuade import human_baseline_essays
from repro.records import Reply
from repro.text import clean, sentences

logger = logging.getLogger(__name__)

ENGLISH = "eng_Latn"
JAPANESE_PIVOT = "jpn_Jpan"
LANGUAGE_NAMES = {ENGLISH: "English", JAPANESE_PIVOT: "Japanese"}
PROMPT = (
    "Translate the following segment into {}, without additional explanation.\n\n{}"
)
JAPANESE_CHARACTERS = re.compile(r"[぀-ヿ一-鿿]")

MODAL_PREFIX = "modal:"
NLLB_PREFIX = "facebook/nllb"
NLLB_MAX_TOKENS = 256
OPENROUTER_MAX_TOKENS = 512
NLLB_BATCH_SIZE = 32
MODAL_CHUNK_SIZE = 64

# A round trip whose length differs from the original by more than this factor is
# discarded; such pairs leave both versions out (Fig. 10).
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.5, 2
UNITS_PER_FLUSH = 4
REMOTE_UNITS_PER_FLUSH = 256  # ~5k sentences: enough chunks to fill all containers


class Translator(ABC):
    """Translates lists of sentences; `round_trip` paraphrases them via a pivot language."""

    remote = False  # True: the work happens elsewhere, so flush in large batches
    requires_pivot_in_japanese = (
        False  # True: untranslated pivot text gives no round trip
    )

    @abstractmethod
    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]: ...

    def round_trip(
        self, texts: list[str], pivot: str = JAPANESE_PIVOT
    ) -> tuple[list[str], list[str]]:
        """(texts translated into `pivot`, those translated back into English)."""
        pivot_texts = self(texts, ENGLISH, pivot)
        if self.requires_pivot_in_japanese:
            pivot_texts = [
                t if JAPANESE_CHARACTERS.search(t) else "" for t in pivot_texts
            ]
        return pivot_texts, self(pivot_texts, pivot, ENGLISH)


def translate_in_length_sorted_chunks(
    texts: list[str],
    chunk_size: int,
    translate_chunks: Callable[[list[list[str]]], list[list[str]]],
) -> list[str]:
    """Translate `texts` in chunks of similar length (they pad less), in the original order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    chunks = [order[k : k + chunk_size] for k in range(0, len(order), chunk_size)]
    translated_chunks = translate_chunks(
        [[texts[i] for i in chunk] for chunk in chunks]
    )
    out = [""] * len(texts)
    for chunk, translations in zip(chunks, translated_chunks):
        for i, translation in zip(chunk, translations):
            out[i] = translation
    return out


class OpenRouterTranslator(Translator):
    """One request per sentence at temperature 0 (Hy-MT2 prompt template, non-Chinese pairs)."""

    requires_pivot_in_japanese = True

    def __init__(self, model: str, workers: int = 32) -> None:
        self.model, self.workers = model, workers

    def _translate_one(self, text: str, tgt: str) -> str:
        from repro.llm import client, retry

        response = retry(
            lambda: client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(LANGUAGE_NAMES[tgt], text),
                    }
                ],
                max_tokens=OPENROUTER_MAX_TOKENS,
                temperature=0,
            )
        )
        return (response.choices[0].message.content or "").strip()

    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(
                pool.map(lambda t: self._translate_one(t, tgt) if t else "", texts)
            )


class NLLBTranslator(Translator):
    """NLLB-200 run locally; the model is loaded on first use (or by `load()`)."""

    def __init__(self, name: str, batch_size: int = NLLB_BATCH_SIZE) -> None:
        self.name, self.batch_size = name, batch_size
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        # 3.3B: ~6.6 GB in fp16 vs ~13 GB in fp32
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.name, dtype=dtype)
            .to(self.device)
            .eval()
        )

    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]:
        self.load()
        self.tokenizer.src_lang = src
        with self._torch.inference_mode():
            return translate_in_length_sorted_chunks(
                texts,
                self.batch_size,
                lambda chunks: [self._translate_batch(c, tgt) for c in chunks],
            )

    def _translate_batch(self, batch: list[str], tgt: str) -> list[str]:
        encoded = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=NLLB_MAX_TOKENS,
        ).to(self.device)
        generated = self._model.generate(
            **encoded,
            max_length=NLLB_MAX_TOKENS,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt),
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class ModalNLLBTranslator(Translator):
    """NLLB on the deployed Modal app (modal_nllb.py): length-sorted chunks fanned out over autoscaled GPU containers.
    Same translation code as NLLBTranslator; `modal deploy modal_nllb.py` first."""

    remote = True

    def __init__(self, name: str, chunk_size: int = MODAL_CHUNK_SIZE) -> None:
        import modal

        self.nllb = modal.Cls.from_name("nllb", "NLLB")(name=name)
        self.chunk_size = chunk_size

    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]:
        def translate_chunks(chunks: list[list[str]]) -> list[list[str]]:
            # list(): drain the generator, otherwise Modal warns about an unclosed async generator at exit
            return list(
                self.nllb.translate.starmap([(chunk, src, tgt) for chunk in chunks])
            )

        return translate_in_length_sorted_chunks(
            texts, self.chunk_size, translate_chunks
        )


def translator(name: str) -> Translator:
    """The translator named by an experiment's `translator` setting."""
    if name.startswith(MODAL_PREFIX):
        return ModalNLLBTranslator(name.removeprefix(MODAL_PREFIX))
    if name.startswith(NLLB_PREFIX):
        return NLLBTranslator(name)
    return OpenRouterTranslator(name)


def unit_sentences(rows: list[Reply], essay: bool) -> list[Record]:
    """Sentence rows (without translations) for the replies of one unit."""
    return [
        {
            "setting": row["setting"],
            "model": row["model"],
            "unit_id": row["unit_id"],
            "prompt": row["prompt"],
            "sent_id": f"{row['turn']}.{k}",
            "orig": sentence,
        }
        for row in rows
        for k, sentence in enumerate(sentences(clean(row["text"], essay=essay)))
    ]


def accepted_round_trip(original: str, round_trip: str) -> str | None:
    """The round trip, or None when its length is too far from the original's."""
    ratio = len(round_trip.split()) / max(len(original.split()), 1)
    return round_trip if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO else None


def human_baseline_rows(essays: pd.DataFrame) -> list[Record]:
    """Sentences of the student essays, without a round trip."""
    return [
        {
            "setting": "essay",
            "model": "human",
            "unit_id": str(essay.essay_id_comp),
            "prompt": essay.prompt_name,
            "sent_id": f"0.{k}",
            "orig": sentence,
            "rt": None,
        }
        for essay in essays.itertuples()
        for k, sentence in enumerate(
            sentences(clean(essay.full_text, essay=True, student=True))
        )
    ]


def add_human_baseline(exp: Experiment, done: set[tuple[str, str]]) -> None:
    """Store the student essays once, if the experiment has a baseline and none is stored yet."""
    has_baseline = any(model == "human" for model, _ in done)
    if exp.human_baseline and exp.persuade and not has_baseline:
        essays = human_baseline_essays(exp.persuade)
        append_jsonl(exp.sents, human_baseline_rows(essays))
        logger.info("added human baseline essays=%d", len(essays))


def _unit_key(reply: Reply) -> tuple[str, str, str]:
    return reply["setting"], reply["model"], reply["unit_id"]


def pending_units(exp: Experiment, done: set[tuple[str, str]]) -> list[list[Reply]]:
    """Replies grouped by unit, for the units without sentences yet."""
    replies = sorted(
        (r for r in read_jsonl(exp.raw) if (r["model"], r["unit_id"]) not in done),
        key=_unit_key,
    )
    return [list(group) for _, group in groupby(replies, key=_unit_key)]


def translate_units(
    exp: Experiment, units: list[list[Reply]], translate: Translator, per_flush: int
) -> None:
    """Round-trip the sentences of `per_flush` units at a time, appending them as they finish."""
    for start in range(0, len(units), per_flush):
        batch = units[start : start + per_flush]
        rows = [
            row
            for unit in batch
            for row in unit_sentences(unit, unit[0]["setting"] == "essay")
        ]
        pivots, round_trips = translate.round_trip([row["orig"] for row in rows])
        for row, pivot, round_trip in zip(rows, pivots, round_trips):
            row["rt"] = accepted_round_trip(row["orig"], round_trip)
            row["pivot"], row["translator"] = pivot, exp.translator
        append_jsonl(exp.sents, rows)
        logger.info(
            "translated experiment=%s units=%d total=%d",
            exp.name,
            min(start + per_flush, len(units)),
            len(units),
        )


def run(exp: Experiment, units_per_flush: int = UNITS_PER_FLUSH) -> None:
    """Add the human baseline and the round trips of every unit not yet in exp.sents."""
    done = {(row["model"], row["unit_id"]) for row in read_jsonl(exp.sents)}
    add_human_baseline(exp, done)
    units = pending_units(exp, done)
    if not units:
        logger.info("nothing to translate experiment=%s", exp.name)
        return
    translate = translator(exp.translator)
    if translate.remote:
        units_per_flush = max(units_per_flush, REMOTE_UNITS_PER_FLUSH)
    logger.info("round trip experiment=%s translator=%s", exp.name, exp.translator)
    translate_units(exp, units, translate, units_per_flush)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="experiment TOML (default: data/raw.jsonl -> data/sentences.jsonl)",
    )
    parser.add_argument(
        "--translator",
        help="OpenRouter slug (default tencent/hy-mt2-30b-a3b) or a local NLLB model: "
        "facebook/nllb-200-3.3B as in the paper, facebook/nllb-200-distilled-600M for a fast pass; "
        "modal:facebook/nllb-200-3.3B runs it on the deployed Modal app (modal_nllb.py)",
    )
    parser.add_argument(
        "--persuade", help="PERSUADE 2.0 CSV: adds student essays as the human baseline"
    )
    parser.add_argument(
        "--units-per-flush",
        type=int,
        default=UNITS_PER_FLUSH,
        help="units translated and saved together",
    )
    args = parser.parse_args()
    configure_logging()

    exp = experiment.load(args.config)
    apply_overrides(exp, args, "translator", "persuade")
    run(exp, args.units_per_flush)


if __name__ == "__main__":
    main()
