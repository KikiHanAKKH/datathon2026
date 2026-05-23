import os
import json
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# --- Config -----------------------------------------------------------------
# Which bias version to preprocess. Change this line to switch versions:
#   outputs/credit_full.csv        -- all features
#   outputs/credit_drop_A2.csv     -- age dropped  (the primary model)
#   outputs/credit_drop_A1_A2.csv  -- A1 and age dropped
INPUT_PATH = os.path.join("outputs", "credit_drop_A2.csv")

LABEL_COL = "Class"

# Column roles. These list ALL possible A-columns; the script only acts on
# the ones actually present in the chosen input file, so it adapts to every
# bias version automatically.
CATEGORICAL_COLS = ["A4", "A5", "A6", "A12"]          # -> one-hot encoded
CONTINUOUS_COLS = ["A2", "A3", "A7", "A10", "A13", "A14"]  # -> scaled
BINARY_COLS = ["A1", "A8", "A9", "A11"]               # -> left as-is


def get_output_dir(input_path):
    """Derive the per-version output folder from the input filename.

    e.g. outputs/credit_drop_A2.csv  ->  outputs/credit_drop_A2/
    """
    version_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("outputs", version_name)


def load_data(path):
    """Load the chosen bias-version CSV, with a clear error if it's missing."""
    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found.")
        print("Run the column-dropping script first to create the bias")
        print("versions in outputs/, or check INPUT_PATH at the top of this file.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded '{path}': {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def main():
    output_dir = get_output_dir(INPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder for this version: {output_dir}/")

    # --- Load ----------------------------------------------------------------
    df = load_data(INPUT_PATH)

    # Safety net: drop CustomerID if it somehow survived into this file.
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("Dropped CustomerID (safety net).")

    if LABEL_COL not in df.columns:
        print(f"ERROR: expected label column '{LABEL_COL}' not found.")
        sys.exit(1)

    # Step 1: split off the label
    # Done before any feature transforms so row order stays aligned.
    labels = df[LABEL_COL]
    features = df.drop(columns=[LABEL_COL])
    print(f"Split off '{LABEL_COL}'. Feature columns: {list(features.columns)}")

    # Step 2: one-hot encode the categorical code columns
    cats_present = [c for c in CATEGORICAL_COLS if c in features.columns]
    features = pd.get_dummies(features, columns=cats_present, dtype=int)
    print(f"One-hot encoded: {cats_present}")

    # Step 3: scale the continuous columns
    conts_present = [c for c in CONTINUOUS_COLS if c in features.columns]
    scaler = StandardScaler()
    features[conts_present] = scaler.fit_transform(features[conts_present])
    print(f"Scaled (StandardScaler): {conts_present}")

    bins_present = [c for c in BINARY_COLS if c in features.columns]
    print(f"Left as-is (already 0/1): {bins_present}")

    # Sanity checks
    if features.isnull().any().any():
        print("WARNING: processed features contain missing values.")
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"WARNING: non-numeric columns remain: {non_numeric}")

    # Save outputs into this version's subfolder
    features_path = os.path.join(output_dir, "processed_features.csv")
    features.to_csv(features_path, index=False)
    print(f"\nSaved: {features_path}  "
          f"({features.shape[0]} rows, {features.shape[1]} feature columns)")

    labels_path = os.path.join(output_dir, "labels.csv")
    labels.to_csv(labels_path, index=False)
    print(f"Saved: {labels_path}")

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump({"scaler": scaler, "continuous_cols": conts_present},
                scaler_path)
    print(f"Saved: {scaler_path}")

    cols_path = os.path.join(output_dir, "feature_columns.json")
    with open(cols_path, "w") as f:
        json.dump(list(features.columns), f, indent=2)
    print(f"Saved: {cols_path}")


if __name__ == "__main__":
    main()