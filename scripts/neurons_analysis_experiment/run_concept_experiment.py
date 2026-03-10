import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add script dir and repo root to path for local and package imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd
from plms_repeats_circuits.utils.model_utils import get_device, tokenize_input
from utils import create_neurons_analysis_dataset_pandas, NeuronsAnalysisDataset
from concepts.concept_utils import compare_seq_positions_to_token_values
from utils import get_repeats_info_about_sequence
from concepts.neuron_concept_maker import (
    ConceptGlobalIndicies, ConceptsMaker,
    TagSequenceWithRepeatsConceptsInput
)
from concepts.concept import Concept, ConceptCategory, SequenceMetadata
from activation_data import (
    ActivationData, ActivationDataBuilder
)
from auroc_analyzer_optimized import AUROCAnalyzerOptimized
from save_activation_data import save_activation_data
from utils import postprocess_auroc_results, get_best_concept_per_neuron_filtered, prepare_dataset_df
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type


def _filter_neurons_by_ratio(neurons_df: pd.DataFrame, repeat_type: str, ratio_threshold: float) -> pd.DataFrame:
    """Filter neurons by {repeat_type}_ratio_in_graph >= ratio_threshold."""
    ratio_col = f"{repeat_type}_ratio_in_graph"
    if ratio_col not in neurons_df.columns:
        return neurons_df
    return neurons_df[neurons_df[ratio_col] >= ratio_threshold].copy()


def load_activation_data(
    load_path: str,
    device: torch.device
) -> tuple:
    """
    Load pre-computed activation data from disk.
    
    Returns:
        (activation_data, concepts_global_indicies_dict)
    """
    print(f"\n[LOAD] Loading pre-computed activations from {load_path}...", flush=True)
    
    data = torch.load(load_path, map_location=device)
    
    activation_data = ActivationData(
        activations_with_repeats=data['activations_with_repeats'].to(device),
        tokens_with_repeats=data['tokens_with_repeats'].to(device),
        neuron_info_list=data['neuron_info_list'],
        seq_to_range_with_repeats=data['seq_to_range_with_repeats']
    )

    concepts_global_indicies_dict = data['concepts_global_indicies_dict']
    
    print(f"  ✓ Loaded {data['n_sequences']} sequences, {data['n_tokens']} tokens, {data['n_neurons']} neurons", flush=True)
    
    return activation_data, concepts_global_indicies_dict


def save_concepts_metadata(output_dir: Path, concepts_array_proteins_with_repeats):
    """Save concepts to CSV."""
    print("\n" + "="*60, flush=True)
    print("SAVING CONCEPTS DATA", flush=True)
    print("="*60, flush=True)
    concepts_rows = [{'concept_name': c.concept_name, 'concept_category': c.concept_category.name} for c in concepts_array_proteins_with_repeats]
    concepts_df = pd.DataFrame(concepts_rows)
    concepts_path = output_dir / "concepts.csv"
    concepts_df.to_csv(concepts_path, index=False)
    print(f"  Saved {len(concepts_df)} concepts to {concepts_path}", flush=True)


def compute_activations_from_scratch(
    args,
    device: torch.device,
    tokenizer,
    concepts_array_proteins_with_repeats,
    concept_maker: ConceptsMaker
) -> tuple:
    """
    Compute activations from scratch by processing sequences through the model.
    
    Returns:
        (activation_data, concepts_global_indicies_dict)
    """
    print(f"\n[1/5] Loading CSV from {args.csv_path}", flush=True)
    df = pd.read_csv(args.csv_path)
    radar_path = args.radar_csv
    df = prepare_dataset_df(df, radar_path=radar_path)
    print(f"  After proteins_single_repeat filter: {len(df)} rows", flush=True)

    print(f"[2/5] Loading Hooked model...", flush=True)
    model = load_model(model_type=args.model_type, device=device, use_transformer_lens_model=True, cache_attention_activations=False, cache_mlp_activations=True, output_type="sequence", cache_attn_pattern=False, split_qkv_input=False)
    model.eval()
    print(f"  Model loaded on {device}", flush=True)
    
    print("[3/5] Creating Dataset...", flush=True)
    
    # Filter suspected repeats if requested
    if args.filter_suspected_repeats:
        if 'additional_suspected_repeats' not in df.columns:
            raise ValueError(
                "Cannot filter suspected repeats: 'additional_suspected_repeats' column not found in CSV file. "
                "Please run the experiment without --filter_suspected_repeats flag, or use a CSV file "
                "that includes the 'additional_suspected_repeats' column."
            )
        original_size = len(df)
        df = df[df['additional_suspected_repeats'].apply(lambda x: len(ast.literal_eval(x) if isinstance(x, str) else x) == 0)].copy()
        print(f"  Filtered suspected repeats: {original_size} → {len(df)} sequences", flush=True)
    
    dataset_df = create_neurons_analysis_dataset_pandas(
        df,
        total_n_samples=args.n_samples,
        random_state=args.seed,
        tokenizer=tokenizer,
    )
    dataset = NeuronsAnalysisDataset(dataset_df)
    dataloader = dataset.to_dataloader(batch_size=1)
    print(f"  Dataset size: {len(dataset)}", flush=True)
    
    # Load neurons CSV
    print(f"[4/5] Loading neurons from {args.neurons_csv}...", flush=True)
    neurons_df = pd.read_csv(args.neurons_csv)
    neurons_df = _filter_neurons_by_ratio(
        neurons_df,
        repeat_type=getattr(args, 'repeat_type', 'identical'),
        ratio_threshold=getattr(args, 'ratio_threshold', 0.8)
    )
    print(f"  Loaded {len(neurons_df)} neurons (after ratio_threshold filter)", flush=True)
    
    # Initialize global indices dict
    concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies] = {}
    
    # Build activation data incrementally
    print("[5/5] Processing sequences and building activation data...", flush=True)
    activation_builder = ActivationDataBuilder(
        device=device, 
        activations_d_type=torch.float32,
        neurons_df=neurons_df,
        tokenizer=tokenizer
    )
    tokenization_shift = 1  # BOS token
    
    for i, batch in enumerate(dataloader):
        success = process_single_sequence(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
            tokenization_shift=tokenization_shift,
            activation_builder=activation_builder,
            concepts_array_proteins_with_repeats=concepts_array_proteins_with_repeats,
            concepts_global_indicies_dict=concepts_global_indicies_dict,
            concept_maker=concept_maker,
            model=model
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} sequences...", flush=True)
    
    # Finalize activation data
    print("\nFinalizing activation data...", flush=True)
    activation_data = activation_builder.build()
    print(f"  Activation tensor with repeats: {activation_data.activations_with_repeats.shape} [tokens x neurons]", flush=True)
    print(f"  Number of neurons: {len(activation_data.neuron_info_list)}", flush=True)
    
    return activation_data, concepts_global_indicies_dict


def create_sequence_metadata(
    seq: str,
    clean_masked_seq_tokenized: torch.Tensor,
    masked_pos: int,
    tokenization_shift: int,
    repeats_info,
    tokenizer,
    device,
    cluster_id: str = None,
    rep_id: str = None
) -> SequenceMetadata:
    """Create SequenceMetadata from sequence data."""
    bos_position_after_tokenization = compare_seq_positions_to_token_values([tokenizer.bos_token_id], clean_masked_seq_tokenized)
    eos_position_after_tokenization = compare_seq_positions_to_token_values([tokenizer.eos_token_id], clean_masked_seq_tokenized)
    unk_positions = compare_seq_positions_to_token_values([tokenizer.unk_token_id], clean_masked_seq_tokenized) if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None else torch.empty(0, dtype=torch.long, device=device)
    pad_positions = compare_seq_positions_to_token_values([tokenizer.pad_token_id], clean_masked_seq_tokenized) if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else torch.empty(0, dtype=torch.long, device=device)

    return SequenceMetadata(
        name=f"{cluster_id}_{rep_id}",
        is_seq_contain_repeat=repeats_info is not None,
        tokenized_seq=clean_masked_seq_tokenized,
        masked_position_after_tokenization=masked_pos,
        tokenization_shift=tokenization_shift,
        repeats_info_about_sequence=repeats_info,
        bos_position_after_tokenization=bos_position_after_tokenization[0] if len(bos_position_after_tokenization) > 0 else torch.tensor(-1, device=device),
        eos_position_after_tokenization=eos_position_after_tokenization[0] if len(eos_position_after_tokenization) > 0 else torch.tensor(-1, device=device),
        unk_positions=unk_positions,
        pad_positions=pad_positions,
        original_sequence=seq
    )


def process_single_sequence(
    batch,
    tokenizer,
    device,
    tokenization_shift: int,
    activation_builder: ActivationDataBuilder,
    concepts_array_proteins_with_repeats,
    concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies],
    concept_maker: ConceptsMaker,
    model,
) -> bool:
    """
    Process a single sequence: tokenize, build activation data, get repeats info, and tag with concepts.
    Returns True if successful, False otherwise.
    """

    clean_masked = batch[0][0]
    masked_positions = batch[1]
    repeat_locations = batch[2]
    repeat_alignments = batch[3]
    sequences = batch[4]
    names = batch[5]
    suspected_repeats = batch[6][0]
    cluster_id = batch[7][0]
    rep_id = batch[8][0]
    
    seq = sequences[0]
    masked_pos = masked_positions[0]
    rep_locs = repeat_locations[0]
    rep_aligns = repeat_alignments[0]
    name = names[0]
    
    # 1. Tokenize sequences (keep batch dimension and attention masks for model)
    clean_tokens, clean_attention_mask = tokenize_input(clean_masked, tokenizer)
    clean_tokens = clean_tokens.to(device)  # [batch, seq_len] - keep batch dimension
    clean_attention_mask = clean_attention_mask.to(device)  # [batch, seq_len]

    
    # 2. Capture activations from model
    from extract_activations import extract_activations_for_sequence
    clean_activations = extract_activations_for_sequence(
        model, clean_tokens, activation_builder.neuron_info_list, device, clean_attention_mask
    )  # [seq_len, n_neurons]
    
    # 3. Add clean sequence to activation data and get start index
    # Squeeze batch dimension only when adding to builder (builder expects 1D)
    clean_tokens_1d = clean_tokens[0]  # [seq_len] - squeeze batch for builder
    activation_builder.add_sequence_with_repeats(clean_tokens_1d, name, activations=clean_activations)
    seq_range_with_repeats = activation_builder.seq_to_range_with_repeats.get(name)
    start_idx_with_repeats = seq_range_with_repeats[0] if seq_range_with_repeats else 0
    
    # 4. Get repeats info (if sequence has repeats)
    repeats_info = None
    if len(rep_locs) > 0:
        try:
            repeats_info = get_repeats_info_about_sequence(
                seq=seq,
                tokenized_sequence=clean_tokens_1d,  # Use 1D version for repeats info
                repeat_locations=rep_locs,
                masked_pos_after_tokenization=masked_pos,
                alignments=rep_aligns,
                near_repeat_threshold=2,  # Exclude tokens within 3 positions of repeat boundaries
                tokenizer=tokenizer,
                suspected_repeat_locations=suspected_repeats,
                tokenization_shift=tokenization_shift,
                device=device
            )
        except Exception as e:
            print(f"  WARNING: Failed to get repeats info for {name}: {e}", flush=True)
            return False
    
    # 5. Create sequence metadata for clean sequence
    sequence_metadata_clean = create_sequence_metadata(
        seq=seq,
        clean_masked_seq_tokenized=clean_tokens_1d,
        masked_pos=masked_pos,
        tokenization_shift=tokenization_shift,
        repeats_info=repeats_info,
        tokenizer=tokenizer,
        device=device,
        cluster_id=cluster_id,
        rep_id=rep_id
    )
    
    # 6. Tag all concepts for proteins with repeats
    tag_input_with_repeats = TagSequenceWithRepeatsConceptsInput(
        sequence_metadata=sequence_metadata_clean,
        tokenizer=tokenizer,
        concepts_array_proteins_with_repeats=concepts_array_proteins_with_repeats,
        concepts_global_indicies_dict=concepts_global_indicies_dict,
        tokenized_seq_start_point_index_in_activations_tensor_for_proteins_with_repeats=start_idx_with_repeats,
        sequence_name=name
    )
    concept_maker.tag_sequence_with_repeats(tag_input_with_repeats)
    
    return True


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def _shuffle_concept_positions_for_dataset(
    all_positions: torch.Tensor,
    invalid_positions: torch.Tensor,
    valid_positions: torch.Tensor,
    device: torch.device
):
    """Shuffle valid positions for a single dataset (baseline)."""
    valid_size = valid_positions.numel()
    invalid_clean = invalid_positions if invalid_positions is not None and invalid_positions.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
    candidates = all_positions[~torch.isin(all_positions, invalid_clean)] if invalid_clean.numel() > 0 else all_positions
    if valid_size > 0 and candidates.numel() >= valid_size:
        shuffled_idx = torch.randperm(candidates.numel(), device=device)[:valid_size]
        return candidates[shuffled_idx]
    return torch.empty(0, dtype=torch.long, device=device)


def shuffle_concept_positions(
    concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies],
    activation_data: ActivationData,
    device: torch.device
):
    """Shuffle concept positions while preserving sizes as a baseline."""
    n_tokens = activation_data.activations_with_repeats.shape[0]
    all_positions = torch.arange(n_tokens, device=device)
    for cg in concepts_global_indicies_dict.values():
        cg.global_valid_positions_for_concept_on_proteins_with_repeats = _shuffle_concept_positions_for_dataset(
            all_positions,
            cg.global_invalid_positions_for_concept_on_proteins_with_repeats,
            cg.global_valid_positions_for_concept_on_proteins_with_repeats,
            device
        )


def main():
    # Set stdout to be unbuffered for immediate output (useful for SLURM/logging)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        # Fallback for older Python versions - use -u flag when running script
        pass
    
    parser = argparse.ArgumentParser(description="Run neuron concept experiment.")
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file (not needed when --load_activations is used)")
    parser.add_argument("--radar_csv", type=str, default=None, help="Path to RADAR CSV for suspected repeats (optional)")
    parser.add_argument("--neurons_csv", type=str, required=True, help="Path to neurons CSV file")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    
    # Analyzer arguments
    parser.add_argument("--min_samples", type=int, default=3000, help="Minimum samples (min_concept_size) per group (default: 3000)")
    parser.add_argument("--use_absolute_activations", action="store_true", help="Use abs(activations) instead of raw activations for analysis")
    parser.add_argument("--concepts_filter", type=str, nargs='+', help="Run analysis only on specified concepts (space-separated list). If not provided, runs on all concepts.")
    
    # Baseline modes
    parser.add_argument("--baseline_mode", type=str, choices=["none", "shuffled_concept_positions"], default="none", help="Baseline mode: none or shuffled_concept_positions")
    
    # Data filtering
    parser.add_argument("--filter_suspected_repeats", action="store_true", help="Filter out all sequences with suspected repeats before processing")
    
    # Save/Load activation data
    parser.add_argument("--save_activations", action="store_true", help="Save activation data and concept indices to disk for later analysis")
    parser.add_argument("--save_neuron_stats", action="store_true", help="Save neuron statistics to CSV")
    parser.add_argument("--save_all_results", action="store_true", help="Save full neuron_analysis_results_all_concepts.csv (all concepts per neuron); if False, only best per neuron")
    parser.add_argument("--load_activations", type=str, help="Load pre-computed activations from file (skips model inference)")
    parser.add_argument("--model_type", type=str, choices=["esm3", "esm-c"], required=True, help="Model type: esm3 or esmc")
    parser.add_argument("--repeat_type", type=str, default="identical", help="Repeat type for neurons ratio column (identical, approximate, synthetic)")
    parser.add_argument("--ratio_threshold", type=float, default=0.8, help="Min {repeat_type}_ratio_in_graph for neurons (default: 0.8)")
    args = parser.parse_args()
    
    set_random_seeds(args.seed)
    
    # Print all arguments
    print("="*60, flush=True)
    print("CONFIGURATION", flush=True)
    print("="*60, flush=True)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}", flush=True)
    print("="*60, flush=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    print("="*60, flush=True)
    print("NEURON CONCEPT EXPERIMENT", flush=True)
    print("="*60, flush=True)
    
    # Load tokenizer and device (needed for both paths)
    tokenizer = load_tokenizer_by_model_type(args.model_type)
    device = get_device()
    print(f"Device: {device}", flush=True)
    
    # Create concepts (needed for both paths)
    print("\n[SETUP] Creating Concepts...", flush=True)
    concept_maker = ConceptsMaker()
    concepts_array_proteins_with_repeats = concept_maker.make_concepts_array(tokenizer)
    print(f"  Concepts for proteins with repeats: {len(concepts_array_proteins_with_repeats)}", flush=True)
    
    # Choose path: load or compute
    if args.load_activations:
        print(f"\n[PATH] Loading pre-computed activations (FAST)...", flush=True)
        activation_data, concepts_global_indicies_dict = load_activation_data(args.load_activations, device)
    else:
        print(f"\n[PATH] Computing activations from scratch (SLOW)...", flush=True)
        activation_data, concepts_global_indicies_dict = compute_activations_from_scratch(
            args, device, tokenizer,
            concepts_array_proteins_with_repeats,
            concept_maker
        )
    if args.save_activations and not args.load_activations:
        activations_output_path = output_dir / "activations_with_concepts.pt"
        save_activation_data(
            activation_data=activation_data,
            concepts_global_indicies_dict=concepts_global_indicies_dict,
            output_path=activations_output_path
        )
    if not args.load_activations and getattr(args, 'save_neuron_stats', False):
        print("\n" + "="*60, flush=True)
        print("SAVING NEURON STATISTICS", flush=True)
        print("="*60, flush=True)
        from save_neuron_stats import save_neuron_stats_to_csv
        save_neuron_stats_to_csv(activation_data, str(output_dir / "neuron_stats"))
        

    if args.baseline_mode == "shuffled_concept_positions":
        print("\n" + "="*60, flush=True)
        print("SHUFFLING CONCEPT POSITIONS (BASELINE MODE)", flush=True)
        print("="*60, flush=True)
        
        # Show example before shuffling
        first_concept_name = list(concepts_global_indicies_dict.keys())[0]
        first_concept = concepts_global_indicies_dict[first_concept_name]
        original_positions_sample = first_concept.global_valid_positions_for_concept_on_proteins_with_repeats[:5].cpu().tolist()
        print(f"  Before shuffle - '{first_concept_name}' first 5 positions: {original_positions_sample}", flush=True)
        
        shuffle_concept_positions(
            concepts_global_indicies_dict,
            activation_data,
            device
        )
        
        # Verify shuffling happened
        shuffled_positions_sample = first_concept.global_valid_positions_for_concept_on_proteins_with_repeats[:5].cpu().tolist()
        print(f"  After shuffle  - '{first_concept_name}' first 5 positions: {shuffled_positions_sample}", flush=True)
        print(f"  ✓ Shuffled concept positions for {len(concepts_global_indicies_dict)} concepts", flush=True)
        
        if original_positions_sample == shuffled_positions_sample:
            print("  ⚠ WARNING: Positions appear unchanged! Shuffle may not have worked.", flush=True)
    

    # Filter concepts if specified (for efficiency, filter before analysis)
    concepts_array_proteins_with_repeats_filtered = concepts_array_proteins_with_repeats
    if args.concepts_filter:
        concepts_filter_set = set(args.concepts_filter)
        concepts_array_proteins_with_repeats_filtered = [
            c for c in concepts_array_proteins_with_repeats
            if c.concept_name in concepts_filter_set
        ]
        print(f"  Filtered to {len(concepts_array_proteins_with_repeats_filtered)} concepts", flush=True)
    
    analyzer_name = "AUROC"
    print("\n" + "="*60, flush=True)
    print(f"RUNNING {analyzer_name} ANALYZER", flush=True)
    print("="*60, flush=True)
    
    analyzer = AUROCAnalyzerOptimized(
        activation_data=activation_data,
        concepts_global_indicies_dict=concepts_global_indicies_dict,
        concepts_array_proteins_with_repeats=concepts_array_proteins_with_repeats_filtered,
        use_absolute_activations=args.use_absolute_activations,
        min_samples=args.min_samples,
        device=device,
        seed=args.seed
    )
    results_df = analyzer.analyze()
    
    # Post-process: add final_tag, final_auroc, is_positive_direction, other_concept_names
    results_df = postprocess_auroc_results(results_df, concepts_array_proteins_with_repeats_filtered)
    
    # Best concept per neuron
    best_df = get_best_concept_per_neuron_filtered(results_df, min_auroc=0.99)
    best_cols = ['layer', 'neuron_idx', 'component_id', 'concept_name', 'concept_category', 'final_tag', 'final_auroc', 'is_positive_direction', 'other_concept_names']
    best_out = best_df[best_cols].copy()
    best_path = output_dir / f"{args.model_type}_neurons_to_best_concepts.csv"
    best_out.to_csv(best_path, index=False)
    print(f"  Saved best concept per neuron to {best_path}", flush=True)
    
    results_path = output_dir / "neuron_analysis_results_all_concepts.csv"
    if args.save_all_results:
        results_df.to_csv(results_path, index=False)
        print(f"  Saved {analyzer_name} results to {results_path}", flush=True)
        print(f"  Concept-neuron pairs: {len(results_df)}", flush=True)
    
    # Save concepts metadata
    save_concepts_metadata(output_dir, concepts_array_proteins_with_repeats)
    
    print("\n" + "="*60, flush=True)
    print("EXPERIMENT COMPLETE", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

