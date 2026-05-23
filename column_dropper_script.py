import os
import pandas as pd

INPUT_FILE = "Credit_Card_Applications.csv"
OUTPUT_DIR = "outputs"

FULL_OUTPUT = "credit_full.csv"
DROP_A2_OUTPUT = "credit_drop_A2.csv"
DROP_A1_A2_OUTPUT = "credit_drop_A1_A2.csv"


def load_data(path=INPUT_FILE):
    return pd.read_csv(path)


def drop_columns_if_present(df, columns):
    existing_columns = [col for col in columns if col in df.columns]
    return df.drop(columns=existing_columns)


def save_feature_versions(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Version 1: full dataset, but remove CustomerID
    full_df = drop_columns_if_present(df, ["CustomerID"])
    full_df.to_csv(os.path.join(OUTPUT_DIR, FULL_OUTPUT), index=False)

    # Version 2: remove CustomerID and A2
    drop_a2_df = drop_columns_if_present(df, ["CustomerID", "A2"])
    drop_a2_df.to_csv(os.path.join(OUTPUT_DIR, DROP_A2_OUTPUT), index=False)

    # Version 3: remove CustomerID, A1, and A2
    drop_a1_a2_df = drop_columns_if_present(df, ["CustomerID", "A1", "A2"])
    drop_a1_a2_df.to_csv(os.path.join(OUTPUT_DIR, DROP_A1_A2_OUTPUT), index=False)

    print("Saved CSV files:")
    print(f"- {os.path.join(OUTPUT_DIR, FULL_OUTPUT)}")
    print(f"- {os.path.join(OUTPUT_DIR, DROP_A2_OUTPUT)}")
    print(f"- {os.path.join(OUTPUT_DIR, DROP_A1_A2_OUTPUT)}")


def main():
    df = load_data()
    save_feature_versions(df)


if __name__ == "__main__":
    main()