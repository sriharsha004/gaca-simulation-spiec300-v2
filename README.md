# GACA Simulation SPIEC-300-v2

This repository contains the executable simulation harness and raw outputs for the manuscript:

**A Graph-Augmented Cognitive Agent Architecture for Professional Performance Evaluation: Design and a Simulation Study with Mixed Results**

The repository is intended to let reviewers and readers inspect and recompute every reported simulation number from raw logs.

## Contents

- `corpus.jsonl`: 300 synthetic interaction records, 100 each for teaching, healthcare, and customer service.
- `results_main.jsonl`: 1,200 main-condition result rows, 300 interactions x 4 conditions.
- `results_ablation.jsonl`: 1,500 ablation result rows, 300 interactions x 5 configurations.
- `results_scalability.jsonl`: 5 scalability-sweep result rows.
- `analysis_report.json`: output of `analyze.py`, used as the source for manuscript tables.
- `corpus_gen.py`: synthetic corpus generator.
- `kg.py`, `ontology.py`, `retrieval.py`, `taxonomy.py`: graph schema, ontology, retrieval, and taxonomy code.
- `prompts.py`, `rules.py`, `llm_client.py`: prompt templates, rule templates/checker, and LLM client.
- `run_conditions.py`, `run_ablations.py`, `run_scalability.py`: experiment runners.
- `analyze.py`: analysis script that recomputes report metrics from JSONL logs.
- `run_*_stdout*.txt`, `analyze_stdout*.txt`: captured terminal output from the executed runs.

## Executed Run

The logged results in this repository were produced with:

- Provider: OpenAI API
- Model: `gpt-5.4-mini`
- Environment variable: `GACA_MODEL=gpt-5.4-mini`
- Run date: 2026-07-19
- Mock mode: false for all reported result rows

The final integrity checks for the included artifacts were:

```text
     300 corpus.jsonl
    1200 results_main.jsonl
    1500 results_ablation.jsonl
       5 results_scalability.jsonl
    3005 total

results_main.jsonl:0
results_ablation.jsonl:0
mock rows: 0
any_mock in GACA: False
```

One interruption occurred during the ablation run before additional API credits were added:

```text
RuntimeError: LLM call failed after 5 retries: Error code: 429 - insufficient_quota
```

The ablation runner is checkpointed; after credits were added it resumed from the existing `results_ablation.jsonl` checkpoint and completed to 1,500 rows.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your-openai-api-key>
export GACA_MODEL=gpt-5.4-mini
```

If `OPENAI_API_KEY` is not set, the code enters clearly labeled mock mode. Mock-mode output is for plumbing checks only and must not be reported as a real result.

## Recompute the Report

To recompute the manuscript metrics from the included raw logs:

```bash
python3 analyze.py
```

This writes `analysis_report.json`.

## Re-run the Full Pipeline

To regenerate all outputs from scratch, remove stale generated result files and run:

```bash
python3 kg.py
python3 corpus_gen.py --all --n 100
python3 run_conditions.py
python3 run_ablations.py
python3 run_scalability.py --domain healthcare
python3 analyze.py
```

The runners are checkpointed and resumable.

## Methodological Notes

Ground truth in `corpus.jsonl` is produced by construction: each synthetic transcript was generated to embed a pre-selected performance-gap category from the 24-category taxonomy in `taxonomy.py`. Recommendation quality is scored by the disclosed LLM-judge protocol in `prompts.py`, not by human raters.

The corpus-generation model and the tested-condition model were from the same model family in the reported run. This is a disclosed limitation of the included results.

