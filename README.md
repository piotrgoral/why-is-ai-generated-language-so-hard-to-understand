# Minimal reproduction: "Why is AI-generated language so hard to understand?" (Exp. 1)

> Unofficial reproduction of Experiment 1 from
> **Clark, Martin & Mahowald (2026), [*Why is AI-generated language so hard to understand?*](https://arxiv.org/abs/2610.29162)**, arXiv:2610.29162.
> Not affiliated with or endorsed by the authors. All credit for the study design, prompts and measures goes to them;
> any errors here are ours. See [Deviations from the paper](#deviations-from-the-paper) and [Citation](#citation).

Chat + essay settings of Experiment 1 (`materials/paper/2610.29162v1.md`): collect model text via OpenRouter,
paraphrase it by round-trip translation (EN→JA→EN), measure the difficulty-linked features with spaCy,
and test original vs. round trip per model.

## Setup

```sh
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY and JEV_API_KEY
```

Run commands with `uv run --env-file .env ...`. Data needed:

- **PERSUADE 2.0** (essays only): `persuade_2.0_human_scores_demo_id_github.csv` from the PERSUADE corpus release, passed as `--persuade persuade_2.0.csv`.
- **OpenSubtitles** (chat n-grams): [OPUS OpenSubtitles v2018 mono English](https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz) (3.6 GB) at `data/opensubtitles/en.txt.gz`.

## Quick start

```sh
# 1. collect model text (resumable; --limit 1 for a smoke test)
uv run --env-file .env python -m repro.collect --setting chat --models openai/gpt-5.6-luna --limit 1
uv run --env-file .env python -m repro.collect --setting essay --persuade persuade_2.0.csv

# 2. split sentences and round-trip them (+ student essays as the human baseline)
uv run --env-file .env python -m repro.roundtrip --persuade persuade_2.0.csv

# 3. features, Jev classifier, bootstrap CIs, paired jackknife tests -> results/features.csv, results/diff.csv
uv run --env-file .env python -m repro.analyze   # --spacy en_core_web_sm --no-judge for a fast pass
```

The round trip uses HY-MT2-30B-A3B via OpenRouter by default; pass `--translator facebook/nllb-200-3.3B` to run NLLB-200 locally as in the paper.

## Experiment configs

The commands above run the default experiment (all models, `data/raw.jsonl` → `data/sentences.jsonl` → `results/`).
A config in `experiments/<name>.toml` picks models, settings and stage options and keeps its outputs in
`runs/<name>/` (`raw.jsonl`, `sentences.jsonl`, `results/`). Reference samples and the spaCy/Jev caches in `data/` are shared.
`experiments/` (except `example.toml`) and `runs/` are gitignored, so local configs stay local; copy `experiments/example.toml` to start.

```sh
uv run --env-file .env python -m repro.run experiments/example.toml --limit 1          # smoke run
uv run --env-file .env python -m repro.run experiments/example.toml                    # full run (resumable)
uv run --env-file .env python -m repro.run experiments/example.toml --stages analyze   # one stage only
```

`repro.run` also takes `--translator`, `--spacy` and `--no-judge` overrides; each stage module accepts `--config` too.

To re-translate conversations an experiment already collected, give a new config the same models and settings,
`raw = "runs/<other>/raw.jsonl"` and a different `[roundtrip] translator`; it collects nothing and writes its own
`runs/<name>/sentences.jsonl` and `results/`.

### Optional: NLLB-200 3.3B on Modal

`modal_nllb.py` serves NLLB (the same `NLLBTranslator` code, fp16 on the first free of L4 / A10G / L40S, up to 10 autoscaled
containers, weights cached in the `nllb-hf-cache` volume). Translator `modal:facebook/nllb-200-3.3B` sends length-sorted
64-sentence chunks to it in parallel; several experiments can run at once against the same deployment.

```sh
modal deploy modal_nllb.py          # once (and after changing NLLBTranslator); `modal run modal_nllb.py` for a smoke test
uv run --env-file .env python -m repro.run experiments/<name>.toml --stages roundtrip analyze   # with translator = "modal:facebook/nllb-200-3.3B"
```

## Code layout

| file | role |
|---|---|
| `repro/prompts.py` | models, system prompts and simulated-user prompt (verbatim, App. A), chat topics (Table 5a), Jev questions |
| `repro/paths.py`, `repro/jsonl.py`, `repro/text.py`, `repro/persuade.py` | default paths, JSONL helpers, markdown cleaning + sentence splitting (App. A.5), PERSUADE loader |
| `repro/records.py`, `repro/cli.py` | typed shape of `raw.jsonl` rows; argument handling shared by the stages |
| `repro/llm.py` | OpenRouter client with 3 attempts (5 s / 20 s); prompt caching for Anthropic models; token/cost tally printed after each collect setting |
| `repro/jev.py` | Jev 1.13.0 client (TypeSafe System One API) for denominal-verb / inanimate-subject candidates (App. B.1) |
| `repro/collect.py` | 12-turn chats with simulated user, 10 essays × 7 assignments → `data/raw.jsonl` |
| `repro/roundtrip.py` | per-sentence round trip (OpenRouter HY-MT2 or local NLLB) → `data/sentences.jsonl` (+ `pivot`, `translator`) |
| `repro/experiment.py` | experiment configs (`experiments/*.toml`) and their output paths |
| `repro/run.py` | runs an experiment's stages: collect → round trip → analyze |
| `repro/analyze.py` | orchestrates the analysis → `results/` |
| `repro/analysis/` | `reference.py` (reference samples), `ngram.py` (modified Kneser–Ney trigram, KenLM-style), `syntax.py` (parse features), `judge.py` (Jev), `metrics.py`, `stats.py` (bootstrap, jackknife), `tables.py`, `report.py` |

Caches in `data/`: `ref_wikipedia.txt` and `ref_opensubtitles.txt` (~500k-word samples; n-gram references for
essays and chat), `ref_stats.json` (Wikipedia noun shares and dependency-length weights), `features.jsonl`, `jev.jsonl`.
Delete them after changing `--ref-words`, `--ref-parse-words` or the spaCy model.

## Deviations from the paper
- OpenRouter instead of Bedrock + the Pi harness; Opus 4.5, Sonnet 4.5, Sonnet 5.5, Haiku 4.5, Haiku 5.5, GPT-4.1 (+ mini, nano), GPT-4o (three versions) and GPT-3.5 Turbo are added (3.5 as `gpt-3.5-turbo-16k`: OpenRouter serves `-0613` with a 4k context, too short for 12-turn chats); Olmo 3 and the base models are not included (not on OpenRouter / need text completion). Code setting omitted.
- The simulated user is GPT-5.6 Luna (reasoning off) instead of Claude Sonnet 4.6, about 15x cheaper; unlike the paper's choice it is also one of the studied models.
- Chat personas are ours (the paper doesn't publish them); the simulated user gets a `(start)` cue before the fixed opening message, since its role-swapped history would otherwise begin with an assistant turn.
- The round trip uses Tencent HY-MT2-30B-A3B on OpenRouter (temperature 0, one sentence per request, prompt `Translate the following segment into {Japanese|English}, without additional explanation.`) instead of NLLB-200 3.3B: it keeps meaning better and runs off-machine, but its round trips are ~15% longer (`materials/translation_model/`). Set `translator = "facebook/nllb-200-3.3B"` under `[roundtrip]` (or `--translator`) to run NLLB as in the paper, in fp16 on GPU/MPS (fp32 on CPU). `modal:facebook/nllb-200-3.3B` runs the same on Modal (see above). A Japanese step without Japanese characters gives no round trip.
- The modified Kneser–Ney trigram is a Python re-implementation of KenLM's `lmplz` estimate (checked against `lmplz -o 3` on both references: per-word log10 probabilities agree within 2e-5). The OpenSubtitles sample is uniformly random subtitle lines; the paper does not say how it sampled. No OpenSubtitles human baseline for the non-n-gram features in chat.
- Markdown and essay cleaning is simplified (no encoding repair or quote/whitespace normalization).
- Denominal-verb candidates use noun shares from 100k parsed Wikipedia words for every setting (the paper uses the setting's own human corpus). Round trips that come back as several sentences are parsed as one text. Dependency-length bin weights are renormalised over the bins a source has. The jackknife test is over all units, not stratified by prompt.
- Object relatives are a parse heuristic: adverbial ("the way you talk") and pied-piped ("in which he lives") relatives are excluded, preposition-stranded ones ("the song you dreamed of") are counted. After changing this or the `$...$` math masking in `repro/text.py`, delete `data/features.jsonl` and re-run roundtrip + analyze.

## Citation

If you use this code, please cite the original paper:

```bibtex
@misc{clark2026aigeneratedlanguagehardunderstand,
  title         = {Why is AI-generated language so hard to understand?},
  author        = {Thomas Hikaru Clark and Matias B. Martin and Kyle Mahowald},
  year          = {2026},
  eprint        = {2610.29162},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.29162}
}
```

## Acknowledgments

The prompts in `repro/prompts.py` (App. A), the chat topics (Table 5a) and the copy of the paper in
`materials/paper/` are taken from Clark, Martin & Mahowald (2026), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Experiments also use the PERSUADE 2.0 corpus
and OPUS OpenSubtitles v2018.
