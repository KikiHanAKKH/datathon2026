"""
app.py -- "Find Your Cluster" Streamlit tool (Pillar 2, USML)

Lets a credit applicant discover which cluster of past applicants they most
resemble, and see how that cluster compares to the others. It does NOT
predict approval or make a decision -- it helps the user navigate, showing
where they sit among applicant groups.

MODEL-AGNOSTIC: this app does not assume how many clusters the model has.
It reads however many clusters exist in cluster_profiles.csv and displays
them all. To use a different pipeline version, change MODEL_VERSION below --
nothing else needs to change, whether that model has 2, 3, or 5 clusters.

The model/scaler/columns/profiles are produced offline by the pipeline
(preprocess.py -> cluster.py -> characterize.py). This app only loads them.

Run with:  streamlit run app.py
"""

import json

import streamlit as st
import pandas as pd
import joblib


# ---------- MODEL SELECTION ----------
# Change this ONE value to swap models. It must match a folder in outputs/
# created by the pipeline, e.g. "credit_drop_A2", "credit_full",
# "credit_drop_A1_A2". The app adapts to whatever k that model used.
MODEL_VERSION = "final"

APPROVAL_RATES = {
    0: 0.66,
    1: 0.13,
    2: 0.60
}

VERSION_DIR = f"outputs/{MODEL_VERSION}"


# ---------- FEATURE CONFIG ----------
# Labels are intentionally NEUTRAL. The dataset is anonymized -- we do not
# know what A1-A14 mean, so the UI does not assert guesses (especially not
# guessed protected attributes).
FEATURE_CONFIG = {
    "A1": {"label": "A1 — binary category (meaning not disclosed)", "type": "categorical",
           "mapping": {"Group 0": 0, "Group 1": 1}},
    "A2": {"label": "A2 — possibly age (unconfirmed)", "type": "continuous"},
    "A3": {"label": "A3 — possibly debt or balance (unconfirmed)", "type": "continuous"},
    "A4": {"label": "A4 — 3-way category (meaning not disclosed)", "type": "categorical",
           "mapping": {"Category 1": 1, "Category 2": 2, "Category 3": 3}},
    "A5": {"label": "A5 — 14-way category (meaning not disclosed)", "type": "categorical",
           "mapping": {f"Category {i}": i for i in range(1, 15)}},
    "A6": {"label": "A6 — possibly employment or education type (unconfirmed)", "type": "categorical",
           "mapping": {f"Category {i}": i for i in range(1, 10)}},
    "A7": {"label": "A7 — possibly years employed (unconfirmed)", "type": "continuous"},
    "A8": {"label": "A8 — possibly prior-default flag (unconfirmed)", "type": "categorical",
           "mapping": {"Flag 0": 0, "Flag 1": 1}},
    "A9": {"label": "A9 — possibly currently-employed flag (unconfirmed)", "type": "categorical",
           "mapping": {"Flag 0": 0, "Flag 1": 1}},
    "A10": {"label": "A10 — possibly a credit metric (unconfirmed)", "type": "continuous"},
    "A11": {"label": "A11 — binary flag (meaning not disclosed)", "type": "categorical",
            "mapping": {"Flag 0": 0, "Flag 1": 1}},
    "A12": {"label": "A12 — 3-way category (meaning not disclosed)", "type": "categorical",
            "mapping": {"Category 1": 1, "Category 2": 2, "Category 3": 3}},
    "A13": {"label": "A13 — continuous metric (meaning not disclosed)", "type": "continuous"},
    "A14": {"label": "A14 — possibly income (unconfirmed)", "type": "continuous"},
}

# Must match preprocess.py.
CATEGORICAL_COLS = ["A4", "A5", "A6", "A12"]


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Find Your Cluster", layout="wide")


# ---------- LOAD MODEL FILES (cached -- loads once per session) ----------
@st.cache_resource
def load_model_files(version_dir):
    """Load the trained model and its helper files for the chosen version."""
    model = joblib.load(f"{version_dir}/kmeans.pkl")
    scaler_bundle = joblib.load(f"{version_dir}/scaler.pkl")
    with open(f"{version_dir}/feature_columns.json") as f:
        feature_columns = json.load(f)
    profiles = pd.read_csv(f"{version_dir}/cluster_profiles.csv")
    return model, scaler_bundle, feature_columns, profiles


# ---------- HELPER: styled table ----------
def centered_table(df):
    html = """
    <style>
    .centered-table { width:100%; border-collapse:collapse; margin:1rem 0;
        background-color:#111827; color:white; border-radius:8px;
        overflow:hidden; }
    .centered-table th { text-align:center!important; padding:0.6rem;
        background-color:#1e293b; border:1px solid #334155; color:white;
        font-weight:700; }
    .centered-table td { text-align:center!important; padding:0.6rem;
        border:1px solid #334155; color:white; }
    </style>
    <table class="centered-table"><thead><tr>
    """
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ---------- TRANSFORM + PREDICT ----------
def predict_cluster(user_input, pipeline):
    user_df = pd.DataFrame([user_input])
    return int(pipeline.predict(user_df)[0])


def describe_differences(profiles, user_cluster):
    """Find which features differ most between the user's cluster and the
    others. Returns (feature, user_value, others_avg) tuples, biggest gaps
    first. Works for any number of clusters."""
    avg_cols = [c for c in profiles.columns if c.startswith("avg_")]
    user_row = profiles[profiles["cluster"] == user_cluster].iloc[0]
    others = profiles[profiles["cluster"] != user_cluster]
    if others.empty:
        return []

    diffs = []
    for col in avg_cols:
        user_val = user_row[col]
        others_val = others[col].mean()
        scale = max(abs(user_val), abs(others_val), 1e-9)
        diffs.append((col.replace("avg_", ""), user_val, others_val,
                      abs(user_val - others_val) / scale))
    diffs.sort(key=lambda x: x[3], reverse=True)
    return diffs[:5]


# ---------- TITLE ----------
st.title("Find Your Applicant Cluster")
st.write(
    "Explore which applicant profile group you most resemble, and how it "
    "compares to the other groups, using anonymized credit application data."
)
st.info(
    "This dataset is anonymized -- the true meaning of each field "
    "(A1-A14) is not disclosed. Where a label says 'possibly X', that "
    "is our unconfirmed guess, not a fact. This tool is for exploratory "
    "self-discovery and does NOT make approval decisions."
)

# Load the model up front; a missing file is caught here.
try:
    profiles = pd.read_csv(f"{VERSION_DIR}/cluster_profiles.csv")
    pipeline = joblib.load(f"{VERSION_DIR}/clustering_pipeline.pkl")
except FileNotFoundError as e:
    st.error(
        f"Could not load a required model file: {e}\n\n"
        f"Run the pipeline for '{MODEL_VERSION}' first: preprocess.py -> "
        "cluster.py -> characterize.py."
    )
    st.stop()

n_clusters = len(profiles)
st.caption(f"Loaded model: '{MODEL_VERSION}' -- this model groups applicants "
           f"into {n_clusters} clusters.")


# ---------- FORM ----------
st.header("Applicant Information")

with st.form("applicant_form"):
    st.subheader("Background")
    a1 = st.pills(FEATURE_CONFIG["A1"]["label"],
                  options=list(FEATURE_CONFIG["A1"]["mapping"]),
                  default="Group 0")
    a2 = st.slider(FEATURE_CONFIG["A2"]["label"], 14, 80, 30)
    a4 = st.pills(FEATURE_CONFIG["A4"]["label"],
                  options=list(FEATURE_CONFIG["A4"]["mapping"]),
                  default="Category 1")
    a5 = st.pills(FEATURE_CONFIG["A5"]["label"],
                  options=list(FEATURE_CONFIG["A5"]["mapping"]),
                  default="Category 1")
    a6 = st.pills(FEATURE_CONFIG["A6"]["label"],
                  options=list(FEATURE_CONFIG["A6"]["mapping"]),
                  default="Category 1")
    a12 = st.pills(FEATURE_CONFIG["A12"]["label"],
                   options=list(FEATURE_CONFIG["A12"]["mapping"]),
                   default="Category 1")

    st.subheader("Financial / Credit Profile")
    a3 = st.slider(FEATURE_CONFIG["A3"]["label"], 0.0, 100.0, 10.0)
    a7 = st.slider(FEATURE_CONFIG["A7"]["label"], 0.0, 40.0, 5.0)
    a8 = st.pills(FEATURE_CONFIG["A8"]["label"],
                  options=list(FEATURE_CONFIG["A8"]["mapping"]),
                  default="Flag 0")
    a9 = st.pills(FEATURE_CONFIG["A9"]["label"],
                  options=list(FEATURE_CONFIG["A9"]["mapping"]),
                  default="Flag 0")
    a10 = st.slider(FEATURE_CONFIG["A10"]["label"], 0, 25, 5)
    a11 = st.pills(FEATURE_CONFIG["A11"]["label"],
                   options=list(FEATURE_CONFIG["A11"]["mapping"]),
                   default="Flag 0")
    a13 = st.slider(FEATURE_CONFIG["A13"]["label"], 0, 280, 20)
    a14 = st.slider(FEATURE_CONFIG["A14"]["label"], 1, 100000, 5000)

    submitted = st.form_submit_button("Find My Cluster")


# ---------- OUTPUT ----------
if submitted:
    user_input = {
        "A1": FEATURE_CONFIG["A1"]["mapping"][a1],
        "A2": a2,
        "A3": a3,
        "A4": FEATURE_CONFIG["A4"]["mapping"][a4],
        "A5": FEATURE_CONFIG["A5"]["mapping"][a5],
        "A6": FEATURE_CONFIG["A6"]["mapping"][a6],
        "A7": a7,
        "A8": FEATURE_CONFIG["A8"]["mapping"][a8],
        "A9": FEATURE_CONFIG["A9"]["mapping"][a9],
        "A10": a10,
        "A11": FEATURE_CONFIG["A11"]["mapping"][a11],
        "A12": FEATURE_CONFIG["A12"]["mapping"][a12],
        "A13": a13,
        "A14": a14,
    }

    st.header("Your Cluster Result")

    try:
        cluster_id = predict_cluster(user_input, pipeline)
    except Exception as e:
        st.error(f"Could not assign a cluster: {e}")
        st.stop()

    match = profiles[profiles["cluster"] == cluster_id]
    if match.empty:
        st.warning(f"Assigned to cluster {cluster_id}, but no profile found.")
        st.stop()
    user_row = match.iloc[0]

    # --- the user's cluster ---
    st.subheader(f"You most resemble Cluster {cluster_id}")
    approval_rate = APPROVAL_RATES.get(cluster_id)

    if approval_rate is not None:
        st.metric(
            "Historical approval rate for this cluster",
            f"{approval_rate * 100:.0f}%"
        )

        st.write(
            f"Based on past applicants in Cluster {cluster_id}, "
            f"about {approval_rate * 100:.0f}% were approved."
        )
    
    st.info(
        "This is the HISTORICAL approval rate of past applicants in this "
        "cluster. It describes the past -- it is not a prediction or a "
        "decision about you. The model groups applicants by similarity; "
        "it does not approve or deny anyone."
    )

    # --- comparison: every cluster, the user's highlighted ---
    st.subheader(f"How your cluster compares ({n_clusters} clusters total)")
    compare = profiles.copy()
    compare["This is you"] = compare["cluster"].apply(
        lambda c: "<-- YOU" if c == cluster_id else "")
    show_cols = [c for c in ["cluster", "size", "approval_rate",
                             "This is you"] if c in compare.columns]
    display = compare[show_cols].copy()
    if "approval_rate" in display.columns:
        display["approval_rate"] = \
            (display["approval_rate"] * 100).round(1).astype(str) + "%"
    centered_table(display)

    # --- what distinguishes the user's cluster from the others ---
    if n_clusters > 1:
        st.subheader("What distinguishes your cluster")
        diffs = describe_differences(profiles, cluster_id)
        st.write(
            "Compared with applicants in the other cluster(s), your cluster "
            "differs most on these features (anonymized):"
        )
        diff_table = pd.DataFrame(
            [{"Feature": f,
              "Your cluster's avg": round(uv, 2),
              "Other clusters' avg": round(ov, 2)}
             for f, uv, ov, _ in diffs]
        )
        centered_table(diff_table)
        st.caption(
            "This describes how the groups differ. It is not advice or a "
            "set of steps to take -- the meanings of A1-A14 are not known."
        )

    # --- the user's own inputs ---
    st.subheader("Your entered information")
    centered_table(pd.DataFrame([user_input]))

    st.caption(
        f"Model in use: '{MODEL_VERSION}'. Where this model was trained "
        "without the age-like field (A2), age is collected for your "
        "reference but is not used to assign your cluster."
    )