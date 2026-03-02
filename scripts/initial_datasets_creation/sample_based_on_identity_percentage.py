import pandas as pd
import argparse
import numpy as np
import pandas as pd
import numpy as np

def sample_proteins(input_csv, output_csv, random_state):
    df = pd.read_csv(input_csv)
    
    required_columns = ["identity_percentage"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    bin_edges = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, np.inf]
    bin_labels = [
        "[50,55)", "[55,60)", "[60,65)", "[65,70)", "[70,75)",
        "[75,80)", "[80,85)", "[85,90)", "[90,95)", "[95,100)", "[100,inf)"
    ]

    df["identity_bin"] = pd.cut(
        df["identity_percentage"], 
        bins=bin_edges, 
        labels=bin_labels,
        right=False
    )

    bins_high = ["[75,80)", "[80,85)", "[85,90)", "[90,95)"]
    bins_low  = ["[50,55)", "[55,60)", "[60,65)", "[65,70)", "[70,75)"]
    
    df_high = df[df["identity_bin"].isin(bins_high)]
    N_high = len(df_high)
    
    n_per_low_bin = N_high // len(bins_low)
    
    print(f"Sampling {n_per_low_bin} from each low bin to match {N_high} high-identity examples.")
    
    df_low_list = []
    for bin_label in bins_low:
        df_bin = df[df["identity_bin"] == bin_label]
        if len(df_bin) >= n_per_low_bin:
            df_bin_sampled = df_bin.sample(n=n_per_low_bin, random_state=random_state)
            df_low_list.append(df_bin_sampled)
        else:
            print(f"Warning: bin {bin_label} has only {len(df_bin)} samples, taking all.")
            df_low_list.append(df_bin)
    
    df_low_sampled = pd.concat(df_low_list)

    df_final = pd.concat([df_high, df_low_sampled]).reset_index(drop=True)

    df_final = df_final.drop(columns=["identity_bin"])
    
    df_final.to_csv(output_csv, index=False)
    print(f"Saved sampled dataset with {len(df_final)} rows to {output_csv}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sample proteins from a dataset based on accuracy and repeat length")
    
    parser.add_argument("--input_csv", help="Path to the input CSV file")
    parser.add_argument("--output_csv", help="Path to save the sampled output CSV file")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility (default: 42)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    sample_proteins(args.input_csv, args.output_csv, args.random_state)

