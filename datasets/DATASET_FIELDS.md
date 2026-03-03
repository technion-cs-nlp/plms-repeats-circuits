# Dataset Fields Reference

| Field | Type | Description |
|---|---|---|
| `cluster_id` | string | Identifier of the sequence cluster this entry belongs to (e.g. a UniRef50 cluster ID or a synthetic key like `10_0_0`). |
| `rep_id` | string | Identifier of the specific representative sequence within its cluster (e.g. a UniProt accession or a synthetic key). |
| `tax` | string | Taxonomic label of the organism the sequence comes from. Set to `UNKNOWN` for synthetic sequences. |
| `seq` | string | Full amino acid sequence in single-letter code. |
| `seq_len` | int | Total length of `seq` in residues. |
| `repeat_key` | string | A representative repeat unit sequence. For synthetic and identical repeats this corresponds to the exact repeated segment used in the sequence. For approximate repeats this is one observed instance of the repeat selected as the reference for alignment. |
| `repeat_length` | int | Length of one repeat unit in residues (equals `len(repeat_key)`). |
| `repeat_times` | int | Number of repeat occurrences present in `seq`. |
| `repeat_locations` | list of `[start, end]` | 0-based, inclusive positions of each repeat occurrence in `seq`. `seq[start:end+1]` gives the repeat segment. Sorted by start position. |
| `min_space` | int | Minimum gap in residues between consecutive repeat occurrences, computed as `repeat_locations[i+1][0] − repeat_locations[i][1]` across all adjacent pairs. |
| `max_space` | int | Maximum gap in residues between consecutive repeat occurrences. |
| `repeat_alignments` | list of strings | A list of all repeat instances in their aligned form, as produced by RADAR. Entries may contain gap characters. |
| `total_score` | float | RADAR Total Score: the total score of all repeat units in the multiple alignment. For the calculation of this score, a profile is calculated for the repeat and every repeat unit is scored against the profile. The total score is the sum of the individual scores.  |
| `diagonal` | int | RADAR Diagonal:  the offset of the suboptimal alignment used for calculating the template repeat unit. The diagonal is usually roughly equal to the length of the repeat, but doesn't necessarily have to be so. For example in non-contiguous repeats, the repeat length is shorter than the offset of two repeats (i.e. the diagonal). |
| `BW-FROM` | int | RADAR BW-From: 1-based index of the first residue of the template repeat unit |
| `BW-TO` | int | RADAR BW-To: 1-based index of the last residue of the template repeat unit. |
| `level` | int | Produced by RADAR. |
| `scores` | list of `[score, z_score]` | One `[score, z_score]` pair per repeat occurrence, in the same order as `repeat_locations`. `score` is the raw score of that unit against the profile; `z_score` is the number of standard deviations above the mean for shuffled sequences scored against the same profile. |
| `mutation_percentage` | float | Average percentage of alignment positions that differ between repeat units (indels + substitutions combined), averaged over all pairwise combinations. |
| `indels_count` | int | Total number of gap positions summed across all pairwise comparisons.|
| `substitutions_count` | int | Total number of substitution positions summed across all pairwise comparisons. |
| `indels_percentage` | float | Indels as a fraction of all mutations: `indels_count / (indels_count + substitutions_count) × 100`. |
| `substitutions_percentage` | float | Substitutions as a fraction of all mutations: `substitutions_count / (indels_count + substitutions_count) × 100`. |
| `identity_percentage` | float | Average percentage of alignment positions that are identical (same amino acid letter) between repeat units. |
| `similarity_percentage` | float | Average percentage of alignment positions where the two residues belong to the same physicochemical group (aliphatic GAVLI, aromatic FYW, sulfur-containing CM, hydroxylic ST, basic KRH, acidic/amidic DENQ, or imino P). Identical residues are also considered similar. |
| `blosum_similarity_percentage` | float | Average percentage of alignment positions where the two residues have a non-negative BLOSUM62 score. |
| `indels_score_percentage` | float | Average percentage of alignment positions that are indel positions per pairwise comparison, as a fraction of the full alignment length. |
| `substitutions_score_percentage` | float | Average percentage of alignment positions that are substitution positions per pairwise comparison, as a fraction of the full alignment length. |
| `accuracy` | float | Model accuracy on masked-token prediction for this protein (fraction of correct predictions across all masked positions of repeat tokens). |
| `avg_prediction_prob` | float | Average probability assigned to the true label across all masked positions of repeat tokens. |
| `accuracy_identical` | float | Accuracy over *aligned-identical* positions only (positions where the aligned repeat units match exactly). Approximate repeats. |
| `avg_prob_aligned_identical` | float | Average prediction probability on aligned-identical positions. Approximate repeats. |
| `count_aligned_identical` | int | Number of aligned-identical positions. Approximate repeats. |
| `correct_aligned_identical` | int | Number of correct predictions among aligned-identical positions. Approximate repeats. |
| `high_confidence_positions` | list of int | List of aligned-identical positions the model predicted correctly with high confidence (in the paper we used 0.0, so any correct prediction qualifies). Approximate repeats. |
