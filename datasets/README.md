# Datasets

This document describes the datasets used in the paper. For a description of every CSV column see [`DATASET_FIELDS.md`](DATASET_FIELDS.md).

## Evaluation Datasets

The datasets are organized by repeat type. For each repeat type, an evaluation dataset was created using the method described in [`scripts/initial_datasets_creation/DATASETS_CREATION.MD`](../scripts/initial_datasets_creation/DATASETS_CREATION.MD) and in the paper. These evaluation datasets are used to produce the model accuracy results on repeat prediction tasks reported in Table 1 of the paper. Each repeat type folder contains an `evaluation/` subfolder with the corresponding evaluation CSV.
`baseline/evaluation/baseline.csv` contains 100k protein sequences sampled from UniRef50 (length range 50–500) with no repeat filtering applied. It contains a single column `seq` and serves as a baseline to compare model behavior on arbitrary proteins against its behavior on repeat-containing sequences on repeat prediction tasks, results are reported in appendix.

## Counterfactuals Datasets

Counterfactuals datasets live under `datasets/{repeat_type}/{model}/counterfactuals/` and are **per model** (esm3, esm-c). They are produced by selecting successful instances for repeat prediction tasks — proteins where the model achieves high accuracy on masked-token prediction. Because model performance differs, each model gets its own filtered subset. See [`scripts/evaluation_experiment/README.md`](../scripts/evaluation_experiment/README.md) for the filter logic and pipeline. Note that for the counterfactual datasets we sampled only 5000 instances for approximate and random (synthetic) to match the identical case, which had about 4000–5000 instances.

## Circuit Discovery Datasets

Circuit Discovery datasets live under `datasets/{repeat_type}/{model}/circuit_discovery/` and are **per model** (esm3, esm-c), since they were produced from the counterfactuals datasets. Note that to get these datasets you need to run the counterfactual experiment (see [`scripts/counterfactual_experiment/README.md`](../scripts/counterfactual_experiment/README.md)). In our pipeline we first evaluated the model on repeat prediction tasks, then chose successful instances per model, then evaluated a counterfactual method for circuit discovery (explained in the appendix). We found four counterfactual methods that worked well: blosum, blosum-opposite50, mask, permutation. We ran circuit discovery on all four for ESM3, as described in the appendix. We chose only examples where each counterfactual caused the model to change its original prediction, and for a fair comparison we chose the same examples for all four—where the model changes its prediction for all four methods. We did not rerun circuit discovery after choosing the blosum counterfactual, therefore you see four datasets under esm3/. For ESM-C we already chose the blosum counterfactual based on ESM3, so there you see only one dataset; we also chose only examples where the counterfactual changed the model's prediction.  

Note that for all experiments in parts 4–5 (attention head analysis, neuron analysis, mechanism analysis) we used only the blosum counterfactual datasets in ESM3.

## RADAR Results Datasets

Under `datasets/approximate/esm3/radar/` and `datasets/approximate/esm-c/radar/` we provide RADAR results run on the blosum counterfactual circuit discovery datasets for both models. These results are used in the neuron analysis experiment to avoid using **suspected repeat positions**—positions that RADAR identified as repeats but that did not survive our filtering/sampling for the circuit discovery dataset. See Appendix I.3 (Neuron Classification Details, *Exclusion of ambiguous tokens in concept evaluation*) in the paper.