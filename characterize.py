import os
import sys
import pandas as pd


# --- Config -----------------------------------------------------------------
ORIGINAL_CSV = "Credit_Card_Applications.csv"
VERSION_DIR = os.path.join("outputs", "credit_drop_A2")
LABELS_PATH = os.path.join(VERSION_DIR, "labels.csv")
CLUSTERS_PATH = os.path.join(VERSION_DIR, "cluster_labels.csv")


def load_or_exit(path, what):
    """Load a CSV, or exit with a clear message if it's missing."""
    if not os.path.exists(path):
        print(f"ERROR: {what} not found at '{path}'.")
        print("Make sure the earlier pipeline steps have been run.")
        sys.exit(1)
    return pd.read_csv(path)


def main():
    # --- Load the three inputs ----------------------------------------------
    original = load_or_exit(ORIGINAL_CSV, "original data")
    labels = load_or_exit(LABELS_PATH, "labels file")
    clusters = load_or_exit(CLUSTERS_PATH, "cluster assignments")

    # The three files must have the same number of rows, or they don't line
    # up and every per-cluster number would be wrong.
    n = len(clusters)
    if not (len(original) == len(labels) == n):
        print("ERROR: row counts differ between files --")
        print(f"  original={len(original)}, labels={len(labels)}, "
              f"clusters={n}")
        print("They must match for the clusters to line up. A likely cause")
        print("is outlier removal dropping rows from one file but not others.")
        sys.exit(1)

    # --- Assemble one table: original features + Class + cluster ------------
    df = original.copy()
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    # `Class` may already be in the original CSV; use the labels file as the
    # source of truth and align it.
    df["Class"] = labels.iloc[:, 0].values
    df["cluster"] = clusters["cluster"].values

    overall_approval = df["Class"].mean()
    print(f"Overall approval rate (baseline): {overall_approval:.1%}")
    print(f"Total applicants: {len(df)}\n")

    # --- Describe each cluster ----------------------------------------------
    feature_cols = [c for c in df.columns if c not in ("Class", "cluster")]
    rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        group = df[df["cluster"] == cluster_id]
        size = len(group)
        approval = group["Class"].mean()

        print(f"--- Cluster {cluster_id} ---")
        print(f"  size: {size} applicants ({size/len(df):.1%} of total)")
        print(f"  historical approval rate: {approval:.1%}  "
              f"(baseline {overall_approval:.1%})")
        print(f"  average profile:")
        for col in feature_cols:
            print(f"    {col:<5} {group[col].mean():.2f}")
        print()

        # Build one summary row for the output CSV.
        row = {"cluster": cluster_id,
               "size": size,
               "approval_rate": round(approval, 4)}
        for col in feature_cols:
            row[f"avg_{col}"] = round(group[col].mean(), 3)
        rows.append(row)

    # --- Save the profiles --------------------------------------------------
    profiles = pd.DataFrame(rows)
    out_path = os.path.join(VERSION_DIR, "cluster_profiles.csv")
    profiles.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()