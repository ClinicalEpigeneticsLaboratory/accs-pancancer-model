#!/usr/local/bin/python3.10
import sys
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def local_norm(
    mynorm: pd.DataFrame, manifest: pd.DataFrame, window: int = 1000
) -> pd.DataFrame:
    # Find common CpGs
    common_cpgs = manifest.index.intersection(mynorm.index)

    # Filter both DataFrames to keep only common CpGs
    manifest = manifest.loc[common_cpgs]
    mynorm = mynorm.loc[common_cpgs]

    # Sort manifest by chromosome and position
    manifest = manifest.sort_values(["CHR", "MAPINFO"])

    # Initialize standardized DataFrame
    normalized = mynorm.copy()

    # Process each chromosome independently
    for chr_name, chr_df in manifest.groupby("CHR"):
        # Extract positions and indices for the current chromosome
        positions = chr_df["MAPINFO"].values
        cpg_indices = chr_df.index

        # Using cKDTree for efficient local window lookup
        tree = cKDTree(positions.reshape(-1, 1))

        for idx, pos in zip(cpg_indices, positions):
            # Query for indices within the window range
            indices_within_window = tree.query_ball_point([pos], r=window)

            if len(indices_within_window) == 1:
                continue

            window_indices = cpg_indices[indices_within_window]
            local_mean = mynorm.loc[window_indices].mean(axis=0)
            normalized.loc[idx] = np.log2(mynorm.loc[idx] / local_mean)

    normalized.index = [f"{cpg}l" for cpg in normalized.index]
    return normalized


def global_norm(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.div(data.mean())
    normalized = normalized.applymap(lambda x: np.log2(x))
    normalized.index = [f"{cpg}g" for cpg in normalized.index]
    return normalized


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <mynorm.csv> <manifest.csv>")
        sys.exit(1)

    mynorm_path = sys.argv[1]
    manifest_path = sys.argv[2]

    # Read input files
    try:
        mynorm = pd.read_parquet(mynorm_path).set_index("CpG")
        manifest = pd.read_parquet(manifest_path)

    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)

    try:
        l_norm = local_norm(mynorm, manifest)
        g_norm = global_norm(mynorm)

        # Save output
        output_path = "normalized.parquet"
        pd.concat((mynorm, l_norm, g_norm)).to_parquet(output_path)
        print(f"Normalization complete. Output saved to {output_path}")

    except Exception as e:
        print(f"Error during normalization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
