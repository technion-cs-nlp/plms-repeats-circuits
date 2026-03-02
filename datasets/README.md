# Datasets

This document describes the datasets used in the paper. For a description of every CSV column see [`DATASET_FIELDS.md`](DATASET_FIELDS.md).

## Evaluation Datasets

The datasets are organized by repeat type. For each repeat type, an evaluation dataset was created using the method described in [`scripts/initial_datasets_creation/DATASETS_CREATION.MD`](../scripts/initial_datasets_creation/DATASETS_CREATION.MD) and in the paper. These evaluation datasets are used to produce the model accuracy results on repeat prediction tasks reported in Table 1 of the paper. Each repeat type folder contains an `evaluation/` subfolder with the corresponding evaluation CSV.
`baseline/evaluation/baseline.csv` contains 100k protein sequences sampled from UniRef50 (length range 50–500) with no repeat filtering applied. It contains a single column `seq` and serves as a baseline to compare model behavior on arbitrary proteins against its behavior on repeat-containing sequences on repeat prediction tasks, results are reported in appendix.

