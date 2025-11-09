import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir

def main(args):
    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    # stratified split
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    ensure_dir(args.output_train)
    train_df.to_csv(args.output_train, index=False)
    test_df.to_csv(args.output_test, index=False)
    print(f"Saved {len(train_df)} train and {len(test_df)} test rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/bugs_sample.csv")
    parser.add_argument("--output_train", default="data/train.csv")
    parser.add_argument("--output_test", default="data/test.csv")
    args = parser.parse_args()
    main(args)
