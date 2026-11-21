"""Collect chat conversations (App. A.2) and essays (App. A.4) -> data/raw.jsonl, one row per reply."""

import argparse
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from repro import experiment, llm
from repro.cli import add_workers_argument, apply_overrides, configure_logging
from repro.experiment import Experiment
from repro.jsonl import append_jsonl, read_jsonl
from repro.llm import Message
from repro.persuade import load_persuade
from repro.prompts import (
    CHAT_SYSTEM,
    ESSAY_SYSTEM,
    SIM_USER_FALLBACK,
    SIM_USER_MODEL,
    SIM_USER_SYSTEM,
    TOPICS,
    ModelSpec,
    Topic,
)
from repro.records import SETTINGS, Reply, Setting

logger = logging.getLogger(__name__)

MAX_REPLIES = 12  # replies per chat; the other cap is experiment.max_output_tokens
ESSAYS_PER_PROMPT = 10
SIM_USER_ATTEMPTS = 2
SIM_USER_MAX_CHARS = 2000
NO_LIMIT = 10**9


@dataclass(frozen=True)
class Unit:
    """One conversation or essay to collect: `collect()` returns its reply rows."""

    model: ModelSpec
    unit_id: str
    collect: Callable[[], list[Reply]]


def simulate_user(system: str, conversation: list[Message]) -> str:
    """The next user message, written by the simulated user.

    The simulated user sees the conversation with roles swapped. Anthropic models need the
    first turn to be a user turn, so a neutral cue precedes the (fixed) opening message."""
    swapped = [
        {
            "role": "assistant" if message["role"] == "user" else "user",
            "content": message["content"],
        }
        for message in conversation
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "(start)"},
        *swapped,
    ]
    for _ in range(SIM_USER_ATTEMPTS):
        text, _ = llm.chat(SIM_USER_MODEL, messages, reasoning=False)
        if text and len(text) <= SIM_USER_MAX_CHARS:
            return text
    return SIM_USER_FALLBACK


def _reply_row(
    setting: Setting,
    model_slug: str,
    unit_id: str,
    prompt: str,
    turn: int,
    text: str,
    output_tokens: int,
) -> Reply:
    return {
        "setting": setting,
        "model": model_slug,
        "unit_id": unit_id,
        "prompt": prompt,
        "turn": turn,
        "text": text,
        "output_tokens": output_tokens,
    }


def run_chat(model: ModelSpec, topic: Topic, max_output_tokens: int) -> list[Reply]:
    """A chat of up to MAX_REPLIES replies, ending early once the token budget is spent."""
    simulated_user_system = SIM_USER_SYSTEM.format(
        persona=topic.persona, topic=topic.name
    )
    system = CHAT_SYSTEM.format(assistant=model.assistant, vendor=model.vendor)
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": topic.opening},
    ]
    rows: list[Reply] = []
    tokens_spent = 0
    for turn in range(MAX_REPLIES):
        reply, tokens = llm.chat(model.slug, messages)
        tokens_spent += tokens
        messages.append({"role": "assistant", "content": reply})
        rows.append(
            _reply_row(
                "chat", model.slug, topic.topic_id, topic.topic_id, turn, reply, tokens
            )
        )
        if turn == MAX_REPLIES - 1 or tokens_spent >= max_output_tokens:
            break
        user_message = simulate_user(simulated_user_system, messages[1:])
        messages.append({"role": "user", "content": user_message})
    return rows


def run_essay(
    model: ModelSpec, prompt_name: str, assignment: str, index: int
) -> list[Reply]:
    """One essay written for `assignment`."""
    system = ESSAY_SYSTEM.format(assistant=model.assistant, vendor=model.vendor)
    text, tokens = llm.chat(
        model.slug,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": assignment},
        ],
    )
    unit_id = f"{prompt_name}#{index}"
    return [_reply_row("essay", model.slug, unit_id, prompt_name, 0, text, tokens)]


def chat_units(models: list[ModelSpec], max_output_tokens: int) -> list[Unit]:
    return [
        Unit(model, topic.topic_id, partial(run_chat, model, topic, max_output_tokens))
        for model in models
        for topic in TOPICS
    ]


def essay_units(models: list[ModelSpec], persuade_csv: str | None) -> list[Unit]:
    if not persuade_csv:
        raise SystemExit("persuade (PERSUADE 2.0 CSV) is required for essays")
    assignments = load_persuade(persuade_csv).groupby("prompt_name").assignment.first()
    return [
        Unit(
            model,
            f"{prompt_name}#{index}",
            partial(run_essay, model, prompt_name, assignment, index),
        )
        for model in models
        for prompt_name, assignment in assignments.items()
        for index in range(ESSAYS_PER_PROMPT)
    ]


def units_to_collect(
    units: list[Unit], done: set[tuple[str, str]], limit: int | None
) -> list[Unit]:
    """Units not yet collected, at most `limit` per model."""
    selected: list[Unit] = []
    per_model: dict[str, int] = {}
    for unit in units:
        slug = unit.model.slug
        if (slug, unit.unit_id) in done or per_model.get(slug, 0) >= (
            limit or NO_LIMIT
        ):
            continue
        per_model[slug] = per_model.get(slug, 0) + 1
        selected.append(unit)
    return selected


def collected_units(path: Path, setting: Setting) -> set[tuple[str, str]]:
    """(model, unit_id) of every unit of `setting` already stored in `path`."""
    return {
        (row["model"], row["unit_id"])
        for row in read_jsonl(path)
        if row["setting"] == setting
    }


def run(exp: Experiment, setting: Setting, workers: int = 32) -> None:
    """Collect the units of `setting` missing from exp.raw; failed units are dropped and
    collected again on the next invocation."""
    models = exp.model_specs()
    llm.check_models([model.slug for model in models])
    units = (
        chat_units(models, exp.max_output_tokens)
        if setting == "chat"
        else essay_units(models, exp.persuade)
    )
    todo = units_to_collect(units, collected_units(exp.raw, setting), exp.limit)
    logger.info(
        "collecting experiment=%s setting=%s units=%d", exp.name, setting, len(todo)
    )

    write_lock = threading.Lock()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(unit.collect): unit for unit in todo}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                rows = future.result()
            except Exception:  # one failed conversation must not stop the rest
                logger.exception(
                    "unit failed experiment=%s model=%s unit_id=%s",
                    exp.name,
                    unit.model.slug,
                    unit.unit_id,
                )
                continue
            with write_lock:
                append_jsonl(exp.raw, rows)
            logger.info(
                "unit done model=%s unit_id=%s replies=%d",
                unit.model.slug,
                unit.unit_id,
                len(rows),
            )
    logger.info(
        "usage experiment=%s setting=%s %s", exp.name, setting, llm.usage_summary()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="experiment TOML (default: all models, data/raw.jsonl)"
    )
    parser.add_argument(
        "--setting", choices=SETTINGS, help="default: the config's settings"
    )
    parser.add_argument(
        "--models", nargs="*", help="OpenRouter slugs (overrides the config)"
    )
    parser.add_argument("--persuade", help="PERSUADE 2.0 CSV (essay setting)")
    parser.add_argument("--limit", type=int, help="max units per model (smoke tests)")
    add_workers_argument(parser)
    args = parser.parse_args()
    configure_logging()

    exp = experiment.load(args.config)
    apply_overrides(exp, args, "models", "persuade", "limit")
    if args.setting is None and args.config is None:
        parser.error("--setting is required without --config")
    for setting in [args.setting] if args.setting else exp.settings:
        run(exp, setting, args.workers)


if __name__ == "__main__":
    main()
