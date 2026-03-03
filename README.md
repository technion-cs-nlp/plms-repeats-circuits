# plms-repeats-circuits

This repository accompanies the paper *Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models*.  
Code, data, and instructions will be uploaded soon.

## Model Performance on Repeat Prediction Task Experiment

Code to run evaluation of protein language models (ESM3, ESM-C) on masked-token prediction over repeat and baseline datasets (Table 1 in the paper). It also creates filtered datasets per model used in counterfactual experiments and circuit discovery. Run it with e.g. `python scripts/evaluation_experiment/run.py --model_type esm3` or `python scripts/evaluation_experiment/run.py --model_type esm-c`. See [scripts/evaluation_experiment/README.md](scripts/evaluation_experiment/README.md) for details.
