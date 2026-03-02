# Dataset Creation

This document describes how the initial evaluation datasets were created.

---

## Synthetic Evaluation Dataset

Run the following command to generate the synthetic dataset:

```bash
python scripts/initial_datasets_creation/create_synthetic_repeats_dataset.py \
  --num_samples 20000 \
  --repeat_segment_length_range 10 30 \
  --space_length_range 0 50 5 \
  --repeat_times 2 \
  --sequence_length 200 \
  --output_file_name synthetic_repeats_eval \
  --output_folder datasets/synthetic/evaluation
```

---

## Identical and Approximate Evaluation Datasets

### 1. Download UniRef50

Download the UniRef50 FASTA file from https://www.uniprot.org/help/downloads.

### 2. Sample 10M proteins

Install [seqkit](https://github.com/shenwei356/seqkit) and use it to sample 10 million proteins in the length range 50–400:

```bash
seqkit seq -m 50 -M 400 <input.fasta> \
  | seqkit sample -n 10000000 -s 42 \
  > <output.fasta>
```

### 3. Run RADAR

Run RADAR on the 10M sampled proteins. You can run it locally using https://github.com/AndreasHeger/radar, or use the EMBL-EBI web server at https://www.ebi.ac.uk/jdispatcher/pfa/radar (note: the web server may not be practical for 10M sequences).

### 4. Parse RADAR output to CSV

Write a script to parse the RADAR output and save results to a CSV file, keeping only proteins for which at least one repeat was detected. The CSV must contain the following columns:

`cluster_id`, `rep_id`, `tax`, `seq`, `seq_len`, `repeat_key`, `repeat_length`, `repeat_times`, `repeat_locations`, `min_space`, `max_space`, `repeat_alignments`, `total_score`, `diagonal`, `BW-FROM`, `BW-TO`, `level`, `scores`

See `datasets/DATASET_FIELDS.md` for a description of each column.

### 5. Filter the RADAR results

**Identical repeats:**

```bash
python scripts/initial_datasets_creation/filter_radar_repeats_dataset.py \
  --input_csv <radar results csv> \
  --output_csv datasets/identical/evaluation/identical_repeats_eval.csv \
  --min_repeats_times 2 \
  --max_repeats_times 2 \
  --min_seq_len 50 \
  --min_mutation_percentage 0 \
  --max_mutation_percentage 0 \
  --filter_releated_repeats \
  --trim_percentiles_lower 0.05 \
  --trim_percentiles_upper 0.95
```

**Approximate repeats:**

```bash
python scripts/initial_datasets_creation/filter_radar_repeats_dataset.py \
  --input_csv <radar results csv> \
  --output_csv datasets/approximate/evaluation/approximate_repeats_eval_all.csv \
  --min_repeats_times 2 \
  --max_repeats_times 2 \
  --min_seq_len 50 \
  --min_mutation_percentage 0 \
  --max_mutation_percentage 50 \
  --filter_releated_repeats \
  --trim_percentiles_lower 0.05 \
  --trim_percentiles_upper 0.95
```

### 6. Balance the approximate dataset by identity

The filtered approximate dataset is skewed towards low-identity repeat pairs. To balance the distribution across identity bins, downsample the low-identity bins so their total count matches the number of high-identity examples (identity ≥ 75%):

```bash
python scripts/initial_datasets_creation/sample_based_on_identity_percentage.py \
  --input_csv datasets/approximate/evaluation/approximate_repeats_eval_all.csv \
  --output_csv datasets/approximate/evaluation/approximate_repeats_eval.csv \
  --random_state 42
```
