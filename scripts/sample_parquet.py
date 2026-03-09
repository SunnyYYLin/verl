import argparse
import pandas as pd

def sample_parquet(input_path, output_path, sample_size, seed=42):
    df = pd.read_parquet(input_path)

    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    df.to_parquet(output_path, index=False)

    print(f"Sampled {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    sample_parquet(
        args.input,
        args.output,
        args.n,
        args.seed
    )