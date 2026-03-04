# plms-repeats-circuits

This repository accompanies the paper *Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models*.  
Code, data, and instructions will be uploaded soon.

The `plms_repeats_circuits` package contains code for **attribution patching** and **integrated gradients attribution patching** on nodes, neurons, and edges, plus shared utilities for all experiments. The attribution patching code is mainly based on [EAP-IG](https://github.com/hannamw/EAP-IG/tree/main).

## Repo structure

<!-- TODO: Add detailed repo structure -->

## Model Performance on Repeat Prediction Task Experiment

Code to run evaluation of protein language models (ESM3, ESM-C) on masked-token prediction over repeat and baseline datasets (Table 1 in the paper). It also creates filtered datasets per model used in counterfactual experiments and circuit discovery. Run it with e.g. `python scripts/evaluation_experiment/run.py --model_type esm3` or `python scripts/evaluation_experiment/run.py --model_type esm-c`. See [scripts/evaluation_experiment/README.md](scripts/evaluation_experiment/README.md) for details.

## Counterfactual Experiment

Code to run counterfactual corruption experiments on protein repeat regions. The experiment corrupts positions in one of the repeat instances (using methods such as masking, BLOSUM substitution, permutation, etc.) and measures whether the model's prediction at a masked position inside the clean repeat region changes. It produces: the corruption comparison plots for ESM3 (Appendix, Figure 19), and the circuit discovery datasets used for circuit discovery, circuit comparison, and all downstream analyses in Sections 4 and 5 of the paper. For ESM3, run the full set of methods: `python scripts/counterfactual_experiment/run.py --model_type esm3`. For ESM-C, we ran only the BLOSUM 100% method: `python scripts/counterfactual_experiment/run.py --model_type esm-c --eval_methods blosum --filter_methods blosum`. See [scripts/counterfactual_experiment/README.md](scripts/counterfactual_experiment/README.md) for details.
