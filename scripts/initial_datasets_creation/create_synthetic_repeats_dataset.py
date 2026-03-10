import argparse
import pandas as pd
import random
import logging
import time
import csv
import traceback
from pathlib import Path
from esm.tokenization import get_esm3_model_tokenizers, get_invalid_tokenizer_ids
import math

def get_min_max_spacing(tuples):
    min_space = math.inf
    max_space = -math.inf
    for i in range(1, len(tuples)):
        space = tuples[i][0] - tuples[i - 1][1]
        if space < min_space:
            min_space = space
        if space > max_space:
            max_space = space
    return min_space, max_space


def create_random_protein_sequence(repeat_segment_length, space_length, repeat_times, sequence_length):
    tokenizers = get_esm3_model_tokenizers()
    tokenizer = tokenizers.sequence
    valid_ids = (
        set(tokenizer.all_token_ids)
        - set(tokenizer.special_token_ids)
        - set(get_invalid_tokenizer_ids(tokenizer))
    )
    non_common_sequence_tokens_ids = {
        tokenizer.vocab['B'],
        tokenizer.vocab['X'],
        tokenizer.vocab['U'],
        tokenizer.vocab['Z'],
        tokenizer.vocab['O'],
        tokenizer.vocab['.'],
        tokenizer.vocab['-'],
    }
    valid_ids = valid_ids - non_common_sequence_tokens_ids
    valid_ids = list(valid_ids)

    repeat_key = [random.choice(valid_ids) for _ in range(repeat_segment_length)]
    repeated_array = []
    repeat_locations = []
    
    # Add repeats up to the specified number or until we run out of space
    for i in range(repeat_times):
        # Check if we have room for another repeat
        if len(repeated_array) + repeat_segment_length > sequence_length:
            break
            
        start_pos = len(repeated_array)
        end_pos = start_pos + repeat_segment_length - 1
        repeat_locations.append((start_pos, end_pos))
        repeated_array += repeat_key
        
        # Add space if this isn't the last repeat and we have room
        if i < repeat_times - 1 and len(repeated_array) + space_length <= sequence_length:
            repeated_array += [random.choice(valid_ids) for _ in range(space_length)]
    
    # Fill any remaining characters to match exact sequence length
    remaining_length = sequence_length - len(repeated_array)
    if remaining_length > 0:
        repeated_array += [random.choice(valid_ids) for _ in range(remaining_length)]

    actual_repeat_times = len(repeat_locations)
    repeated_str = tokenizer.decode(repeated_array).replace(" ", "")

    return repeated_str, repeat_locations, tokenizer.decode(repeat_key).replace(" ", ""), actual_repeat_times


def create_random_repeats_dataset(
    num_samples: int,
    repeat_segment_length: int,
    space_length: int,
    repeat_times: int,
    sequence_length: int,
    output_file_name: str,
    output_folder: str,
):
    output_folder = Path(output_folder)
    output_csv = output_folder / f"{output_file_name}.csv"

    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "cluster_id", "rep_id", "tax", "seq", "seq_len", "repeat_key",
                "repeat_length", "repeat_times", "repeat_locations", "min_space", "max_space"
            ])
        try:
            for i in range(num_samples):
                cluster_id = f"{repeat_segment_length}_{space_length}_{i}"
                rep_id = f"{repeat_segment_length}_{space_length}_{i}"
                tax = "UNKNOWN"
                seq, repeat_locations, repeat_key, actual_repeat_times = create_random_protein_sequence(
                    repeat_segment_length, space_length, repeat_times, sequence_length
                )
                sequence_length_actual = len(seq)
                repeat_length = len(repeat_key)
                assert repeat_segment_length == repeat_length
                sorted_repeat_locations = sorted(repeat_locations, key=lambda x: (x[0], x[1]))
                min_space, max_space = get_min_max_spacing(sorted_repeat_locations)
                writer.writerow([
                    cluster_id,
                    rep_id,
                    tax,
                    seq,
                    sequence_length_actual,
                    repeat_key,
                    repeat_length,
                    actual_repeat_times,
                    sorted_repeat_locations,
                    min_space,
                    max_space
                ])

                if (i + 1) % 1000 == 0:
                    logging.info(f"Processed {i + 1} records for repeat length {repeat_segment_length}, space length {space_length}.")
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of synthetic protein sequences with repeated segments.")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--repeat_segment_length_range", type=int, nargs=2, metavar=('MIN_LEN', 'MAX_LEN'),
                        required=True, help="Range (inclusive) of repeat segment lengths to generate.")
    parser.add_argument("--space_length_range", type=int, nargs=3, metavar=('MIN_LEN', 'MAX_LEN', 'STEP'),
                        required=True, help="Range (inclusive) of space lengths with step size.")
    parser.add_argument("--repeat_times", type=int, required=True, help="Number of times to repeat the segment.")
    parser.add_argument("--sequence_length", type=int, required=True, help="Total length of the generated sequence.")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the output CSV file (without extension).")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder where the output CSV and log will be saved.")
    args = parser.parse_args()

    # Ensure output folder exists
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_folder / f"{args.output_file_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

    # Compute segment lengths and space lengths
    segment_lengths = list(range(args.repeat_segment_length_range[0], args.repeat_segment_length_range[1] + 1))
    space_lengths = list(range(args.space_length_range[0], args.space_length_range[1] + 1, args.space_length_range[2]))
    
    # Calculate how many samples per combination
    total_combinations = len(segment_lengths) * len(space_lengths)
    samples_per_combination = args.num_samples // total_combinations

    logging.info(f"Generating {samples_per_combination} samples for each combination of:")
    logging.info(f"- Repeat lengths: {segment_lengths}")
    logging.info(f"- Space lengths: {space_lengths}")
    logging.info(f"- Target repeat times: {args.repeat_times}")
    logging.info(f"- Total sequence length: {args.sequence_length}")

    for segment_length in segment_lengths:
        for space_length in space_lengths:
            create_random_repeats_dataset(
                num_samples=samples_per_combination,
                repeat_segment_length=segment_length,
                space_length=space_length,
                repeat_times=args.repeat_times,
                sequence_length=args.sequence_length,
                output_file_name=args.output_file_name,
                output_folder=str(output_folder),
            )

    logging.info("Dataset generation completed.")





