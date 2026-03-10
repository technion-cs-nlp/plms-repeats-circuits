# Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models

[Paper on arXiv](https://arxiv.org/abs/2602.23179v1) | [Dataset on HuggingFace](https://huggingface.co/datasets/galkesten/plms_repeats_circuits/tree/main)


## Repo structure

```
├── plms_repeats_circuits/   # General attribution patching code for circuit discovery; shared code for experiments
│   ├── EAP/                 # Circuit discovery
│   └── utils/               # Shared experiment utilities
├── scripts/
│   ├── initial_datasets_creation/    # Code to generate the original datasets
│   ├── evaluation_experiment/        # Evaluates ESM3/ESM-C on the repeat prediction task
│   ├── counterfactual_experiment/    # Tests different counterfactuals for circuit discovery
│   ├── circuit_discovery_experiment/ # Circuit discovery on the repeat prediction task + circuit comparison
│   ├── component_recurrence_stats/   # Aggregates which components recur across seeds; used to select components for downstream analysis
│   ├── attention_heads_clustering_experiments/  # Attention heads clustering
│   ├── interactions_graph_experiment/           # Interaction graph between attention-head clusters and MLP clusters
│   ├── logit_lens_experiments/       # Logit lens for the repeat prediction task
│   └── neurons_analysis_experiment/  # Neuron–concept matching and analysis
├── datasets/                # All datasets
├── ThirdParty/              # Fork of TransformerLens containing ESM-C and ESM3 support
├── results/                 # Outputs from all experiments
├── pyproject.toml           # Project metadata and dependencies (Poetry)
└── poetry.lock              # Locked dependency versions for reproducible installs
```

The `plms_repeats_circuits` package contains code for **attribution patching** and **integrated gradients attribution patching** on nodes, neurons, and edges, plus shared utilities for all experiments. The attribution patching code is mainly based on [EAP-IG](https://github.com/hannamw/EAP-IG/tree/main).

## Set up the environment

This project uses [Poetry](https://python-poetry.org/) for dependency management. See the [official documentation](https://python-poetry.org/docs/).

**Clone the repo and init submodule:**

```bash
git clone <repo-url>
cd plms-repeats-circuits
git submodule update --init --recursive
```

### Option 1: Poetry virtual environment

Poetry creates and manages its own virtual environment. Dependencies are installed into it.

```bash
poetry install
# Optional: dev and Jupyter deps (for notebooks, tests)
poetry install --with dev jupyter
```

**Run scripts** — either:

```bash
poetry run python scripts/...
```

or spawn a shell with the env activated:

```bash
poetry shell
python scripts/...
```

Tip: `poetry config virtualenvs.in-project true` creates `.venv` inside the project so you can also `source .venv/bin/activate`.

### Option 2: Conda environment + Poetry

Use Conda for the Python version, Poetry for dependencies (installs into the active Conda env).

```bash
conda create -n plms-repeats-circuits python=3.10
conda activate plms-repeats-circuits

# Tell Poetry to use the active env instead of creating its own
poetry config virtualenvs.create false

poetry install
```

To revert later (if you switch to Option 1): `poetry config virtualenvs.create true`

**Run scripts** — with the env activated:

```bash
conda activate plms-repeats-circuits
python scripts/...
```

## Datasets and results on Hugging Face

Datasets and results are hosted at [galkesten/plms_repeats_circuits](https://huggingface.co/datasets/galkesten/plms_repeats_circuits/tree/main). To run the full experiments, download the datasets. For partial experiments, also download results—some scripts (e.g. circuit comparison, neurons analysis) depend on outputs like discovered circuits stored in `results/`.

```bash
# Download to datasets/ and results/ (default)
python download_from_hf.py

# Download to custom dirs
python download_from_hf.py --datasets-dir ./datasets_test --results-dir ./results_test
```

By default, files go to `datasets/` and `results/` in the project root. Use `--datasets-dir` and `--results-dir` to choose other paths. Use `--force` to overwrite existing files. For rate limits, pass `--token YOUR_TOKEN` or set `HF_TOKEN` in your environment.

## Experiments

### Model Performance on Repeat Prediction Tasks

Code to run evaluation of protein language models (ESM3, ESM-C) on masked-token prediction over repeat and baseline datasets (Table 1 in the paper). It also creates filtered datasets per model used in counterfactual experiments and circuit discovery. Run it with e.g. `python scripts/evaluation_experiment/run.py --model_type esm3` or `python scripts/evaluation_experiment/run.py --model_type esm-c`. See [scripts/evaluation_experiment/README.md](scripts/evaluation_experiment/README.md) for details.

### Counterfactual Experiment

Code to run counterfactual corruption experiments on protein repeat regions. The experiment corrupts positions in one of the repeat instances (using methods such as masking, BLOSUM substitution, permutation, etc.) and measures whether the model's prediction at a masked position inside the clean repeat region changes. It produces: the corruption comparison plots for ESM3 (Appendix, Figure 19), and the circuit discovery datasets used for circuit discovery, circuit comparison, and all downstream analyses in Sections 4 and 5 of the paper. For ESM3, run the full set of methods: `python scripts/counterfactual_experiment/run.py --model_type esm3`. For ESM-C, we ran only the BLOSUM 100% method: `python scripts/counterfactual_experiment/run.py --model_type esm-c --eval_methods blosum --filter_methods blosum`. See [scripts/counterfactual_experiment/README.md](scripts/counterfactual_experiment/README.md) for details.

### Circuit Discovery and Comparison

Code to discover minimal circuits at the level of edges, nodes, or neurons via attribution patching (EAP-IG) on ESM3 and ESM-C. Attribution patching scores components by how much they affect the model's output; the pipeline runs a binary search to find the minimal circuit that reproduces full-model behavior above a faithfulness threshold. It also includes code for comparing circuits using IoU, recall, and cross-task faithfulness (how well a circuit from one task transfers to another). Uses circuit_discovery datasets from the counterfactual experiment. Run with e.g. `python scripts/circuit_discovery_experiment/run.py --graph_type nodes --methods blosum --repeat_types identical --model_types esm3`. See [scripts/circuit_discovery_experiment/README.md](scripts/circuit_discovery_experiment/README.md) for full usage and commands to produce all circuits from the paper.

### Component Recurrence Stats

An analysis script that computes statistics used by downstream experiments (Sections 4 and 5) to select components for analysis. Aggregates which attention heads and neurons appear in minimal circuits across seeds; each seed uses different dataset examples, so recurring components are more likely to be important. Run with e.g. `python scripts/component_recurrence_stats/run_component_recurrence_stats.py --graph_type nodes --repeat_type approximate --model_type esm-c --counterfactual_type blosum`. See [scripts/component_recurrence_stats/README.md](scripts/component_recurrence_stats/README.md) for details.

### Attention Heads Clustering

Clusters attention heads by behavioral features (copy score, pattern matching, position, amino-acid bias, etc.) to identify Induction Heads, Relative Position Heads, and AA-biased Heads (Section 4 of the paper). Requires circuit discovery datasets and component recurrence CSVs. Run with e.g. `python scripts/attention_heads_clustering_experiments/run.py --model_types esm3 esm-c --counterfactual_type blosum --repeat_type approximate`. See [scripts/attention_heads_clustering_experiments/README.md](scripts/attention_heads_clustering_experiments/README.md) for details.

### Interactions Graph

Creates interaction graph (Figure 7+16 in the paper). Requires circuit discovery (edges), attention heads clustering, and component recurrence. Run with e.g. `python scripts/interactions_graph_experiment/run.py --repeat_type approximate --counterfactual_type blosum`. See [scripts/interactions_graph_experiment/README.md](scripts/interactions_graph_experiment/README.md) for details.

### Logit Lens

Computes logit lens metrics (top-1 accuracy, correct-token probability) for residual stream, attention heads, and MLPs across layers. Produces the logit lens across layers and clusters plot (paper figures 8 and 17). Requires circuit discovery datasets, component recurrence, and attention heads clustering. Run with e.g. `python scripts/logit_lens_experiments/run.py --repeat_type approximate --counterfactual_type blosum`. See [scripts/logit_lens_experiments/README.md](scripts/logit_lens_experiments/README.md) for details.

### Neurons Analysis Experiment

Analyzes which concepts (repeat tokens, biochemical properties, special tokens) best predict neuron activations via AUROC. Uses circuit discovery datasets, component recurrence (neurons), and optionally RADAR for additional suspected repeats. Produces concepts bar chart, AUROC scatter by layer bin, layer×cluster heatmap, and ESM-C vs. ESM-3 biochemical comparison (Figures 6, 9, 15, 18). Run with e.g. `python scripts/neurons_analysis_experiment/run.py --repeat_type approximate --counterfactual_type blosum --steps run analyze compare`. See [scripts/neurons_analysis_experiment/README.md](scripts/neurons_analysis_experiment/README.md) for details.

## Citation

```bibtex
@misc{kestenpomeranz2026inductionmeetsbiologymechanisms,
  title={Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models},
  author={Gal Kesten-Pomeranz and Yaniv Nikankin and Anja Reusch and Tomer Tsaban and Ora Schueler-Furman and Yonatan Belinkov},
  year={2026},
  eprint={2602.23179},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.23179},
}
```

## License

This project uses the TransformerLens fork in `ThirdParty/TransformerLens`, which is multi-licensed. See [ThirdParty/TransformerLens/README.md](ThirdParty/TransformerLens/README.md) for full details:

- **TransformerLens Core**: MIT License — see [ThirdParty/TransformerLens/LICENSE](ThirdParty/TransformerLens/LICENSE)
- **ESMC 300M**: [EvolutionaryScale Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)
- **ESM-3 & ESMC 600M**: [EvolutionaryScale Cambrian Non-Commercial License](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement) (non-commercial use only)