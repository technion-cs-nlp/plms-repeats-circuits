# Evaluation Experiment

Evaluation of protein language models (ESM3, ESM-C) on masked-token prediction over repeat and baseline datasets. The pipeline has three steps: **evaluate**, **filter**, and **analyze**.

This script generates:
- **Table 1. Model accuracies on repeat prediction tasks** — from the analyze step (`tasks_performance.csv`)
- **Datasets for counterfactuals methods experiment** — from the filter step (high-accuracy subsamples written to `datasets/{name}/{model}/counterfactuals/`)

## Pipeline Overview

```
evaluate → filter, analyze
```

1. **evaluate** — Run model predictions on all datasets; outputs per-position predictions. Must run first.
2. **filter** — Filter high-accuracy proteins and subsample to create datasets for counterfactuals methods; outputs go to `datasets/{name}/{model}/counterfactuals/`.
3. **analyze** — Aggregate performance across tasks and produce summary statistics for Table 1.

Filter and analyze have no order relative to each other (both depend only on evaluate).

---

## Datasets

Expected datasets for evaluation are in `datasets/<repeat_type>/evaluation/`.

---

## Files

| File | Description |
|------|-------------|
| `run.py` | Pipeline driver. Runs evaluate, filter, analyze. `--datasets` applies to evaluate and filter; analyze always uses all datasets. |
| `evaluate.py` | Masked-token prediction on repeat positions (or random for baseline). |
| `filter.py` | Filter high-accuracy proteins, subsample; output counterfactuals datasets. |
| `analyze.py` | Aggregate accuracy across tasks → `tasks_performance.csv` (Table 1). |
| `utils.py` | Shared helpers |

---

## Running the Pipeline

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets_root` | Root folder for input datasets | `{repo}/datasets` |
| `--results_root` | Root folder for outputs | `{repo}/results` |
| `--model_type` | Model: `esm3` or `esm-c` | `esm3` |
| `--steps` | Steps to run: `evaluate`, `filter`, `analyze` | all three |
| `--datasets` | Datasets for evaluate/filter: `synthetic`, `identical`, `approximate`, `baseline` | all four |
| `--batch_size` | Batch size for evaluate | `16` |

**Note:** The analyze step requires all four datasets. If `analyze` is in `--steps`, `--datasets` must include all four.

**Examples:**

```bash
python run.py --model_type esm3 --steps evaluate filter analyze --datasets synthetic identical approximate baseline
# Full pipeline: esm3, all 4 datasets, all steps

python run.py --model_type esm3 --steps evaluate
# Evaluate only (filter and analyze skipped)

python run.py --model_type esm-c --steps evaluate filter --datasets synthetic identical
# esm-c, only synthetic and identical (analyze skipped; analyze needs all datasets)

python run.py --model_type esm3 --datasets_root /path/to/datasets --results_root /path/to/results
# Custom input/output roots
```
