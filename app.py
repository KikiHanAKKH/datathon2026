import streamlit as st
import pandas as pd

# ---------- FEATURE CONFIG ----------

FEATURE_CONFIG = {
    "A1": {
        "label": "A1 – Likely Gender",
        "type": "categorical",
        "mapping": {
            "Category 0": 0,
            "Category 1": 1
        }
    },
    "A2": {
        "label": "A2 – Likely Age",
        "type": "continuous"
    },
    "A3": {
        "label": "A3 – Likely Debt / Balance",
        "type": "continuous"
    },
    "A4": {
        "label": "A4 – Likely Marital Status",
        "type": "categorical",
        "mapping": {
            "Marital Category 1": 1,
            "Marital Category 2": 2,
            "Marital Category 3": 3
        }
    },
    "A5": {
        "label": "A5 – Likely Applicant Category",
        "type": "categorical",
        "mapping": {
            "Applicant Type 1": 1,
            "Applicant Type 2": 2,
            "Applicant Type 3": 3,
            "Applicant Type 4": 4,
            "Applicant Type 5": 5,
            "Applicant Type 6": 6,
            "Applicant Type 7": 7,
            "Applicant Type 8": 8,
            "Applicant Type 9": 9,
            "Applicant Type 10": 10,
            "Applicant Type 11": 11,
            "Applicant Type 12": 12,
            "Applicant Type 13": 13,
            "Applicant Type 14": 14
        }
    },
    "A6": {
        "label": "A6 – Likely Employment Type",
        "type": "categorical",
        "mapping": {
            "Employment Type 1": 1,
            "Employment Type 2": 2,
            "Employment Type 3": 3,
            "Employment Type 4": 4,
            "Employment Type 5": 5,
            "Employment Type 6": 6,
            "Employment Type 7": 7,
            "Employment Type 8": 8
        }
    },
    "A7": {
        "label": "A7 – Likely Years Employed",
        "type": "continuous"
    },
    "A8": {
        "label": "A8 – Likely Prior Default",
        "type": "categorical",
        "mapping": {
            "No Prior Default": 0,
            "Prior Default": 1
        }
    },
    "A9": {
        "label": "A9 – Likely Currently Employed",
        "type": "categorical",
        "mapping": {
            "Not Currently Employed": 0,
            "Currently Employed": 1
        }
    },
    "A10": {
        "label": "A10 – Likely Credit Metric",
        "type": "continuous"
    },
    "A11": {
        "label": "A11 – Likely Risk Flag",
        "type": "categorical",
        "mapping": {
            "Flag 0": 0,
            "Flag 1": 1
        }
    },
    "A12": {
        "label": "A12 – Likely Region / Citizenship",
        "type": "categorical",
        "mapping": {
            "Region Group 1": 1,
            "Region Group 2": 2,
            "Region Group 3": 3
        }
    },
    "A13": {
        "label": "A13 – Unknown Financial Metric",
        "type": "continuous"
    },
    "A14": {
        "label": "A14 – Likely Income",
        "type": "continuous"
    }
}

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Find Your Cluster",
    layout="wide"
)

# ---------- HELPER ----------

def centered_table(df):
    table_html = """
    <style>
    .centered-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        margin-bottom: 1rem;
        background-color: #111827;
        color: white;
        border-radius: 8px;
        overflow: hidden;
    }

    .centered-table th {
        text-align: center !important;
        padding: 0.75rem;
        background-color: #1e293b;
        border: 1px solid #334155;
        color: white;
        font-weight: 700;
    }

    .centered-table td {
        text-align: center !important;
        padding: 0.75rem;
        border: 1px solid #334155;
        color: white;
    }
    </style>

    <table class="centered-table">
        <thead>
            <tr>
    """

    for col in df.columns:
        table_html += f"<th>{col}</th>"

    table_html += """
            </tr>
        </thead>
        <tbody>
    """

    for _, row in df.iterrows():
        table_html += "<tr>"
        for value in row:
            table_html += f"<td>{value}</td>"
        table_html += "</tr>"

    table_html += """
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)


# ---------- TITLE ----------

st.title("Find Your Applicant Cluster")

st.write(
    "Explore which applicant profile group you most resemble using anonymized credit application data."
)

st.info(
    "Field meanings are inferred from anonymized dataset metadata. "
    "Some categories are intentionally labeled generically because the dataset does not reveal their exact meanings."
)

# ---------- FORM ----------

st.header("Applicant Information")

with st.form("applicant_form"):

    st.subheader("Personal / Background")

    a1 = st.pills(
        FEATURE_CONFIG["A1"]["label"],
        options=list(FEATURE_CONFIG["A1"]["mapping"].keys()),
        default="Category 0"
    )

    a2 = st.slider(
        FEATURE_CONFIG["A2"]["label"],
        min_value=14,
        max_value=80,
        value=30
    )

    a4 = st.pills(
        FEATURE_CONFIG["A4"]["label"],
        options=list(FEATURE_CONFIG["A4"]["mapping"].keys()),
        default="Marital Category 1"
    )

    a5 = st.pills(
        FEATURE_CONFIG["A5"]["label"],
        options=list(FEATURE_CONFIG["A5"]["mapping"].keys()),
        default="Applicant Type 1"
    )

    a6 = st.pills(
        FEATURE_CONFIG["A6"]["label"],
        options=list(FEATURE_CONFIG["A6"]["mapping"].keys()),
        default="Employment Type 1"
    )

    a12 = st.pills(
        FEATURE_CONFIG["A12"]["label"],
        options=list(FEATURE_CONFIG["A12"]["mapping"].keys()),
        default="Region Group 1"
    )

    st.subheader("Financial / Credit Profile")

    a3 = st.slider(
        FEATURE_CONFIG["A3"]["label"],
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    a7 = st.slider(
        FEATURE_CONFIG["A7"]["label"],
        min_value=0.0,
        max_value=40.0,
        value=5.0
    )

    a8 = st.pills(
        FEATURE_CONFIG["A8"]["label"],
        options=list(FEATURE_CONFIG["A8"]["mapping"].keys()),
        default="No Prior Default"
    )

    a9 = st.pills(
        FEATURE_CONFIG["A9"]["label"],
        options=list(FEATURE_CONFIG["A9"]["mapping"].keys()),
        default="Currently Employed"
    )

    a10 = st.slider(
        FEATURE_CONFIG["A10"]["label"],
        min_value=0,
        max_value=25,
        value=5
    )

    a11 = st.pills(
        FEATURE_CONFIG["A11"]["label"],
        options=list(FEATURE_CONFIG["A11"]["mapping"].keys()),
        default="Flag 0"
    )

    a13 = st.slider(
        FEATURE_CONFIG["A13"]["label"],
        min_value=0,
        max_value=280,
        value=20
    )

    a14 = st.slider(
        FEATURE_CONFIG["A14"]["label"],
        min_value=1,
        max_value=100000,
        value=5000
    )

    submitted = st.form_submit_button("Find My Cluster")

# ---------- OUTPUT ----------

if submitted:

    user_input = {
        "A1": FEATURE_CONFIG["A1"]["mapping"][a1],
        "A2": a2,
        "A3": f"{a3:.2f}",
        "A4": FEATURE_CONFIG["A4"]["mapping"][a4],
        "A5": FEATURE_CONFIG["A5"]["mapping"][a5],
        "A6": FEATURE_CONFIG["A6"]["mapping"][a6],
        "A7": f"{a7:.2f}",
        "A8": FEATURE_CONFIG["A8"]["mapping"][a8],
        "A9": FEATURE_CONFIG["A9"]["mapping"][a9],
        "A10": a10,
        "A11": FEATURE_CONFIG["A11"]["mapping"][a11],
        "A12": FEATURE_CONFIG["A12"]["mapping"][a12],
        "A13": a13,
        "A14": a14
    }

    user_df = pd.DataFrame([user_input])

    st.header("Cluster Result")

    st.success("Model connection coming next.")

    st.write("Encoded applicant data:")

    centered_table(user_df)

    st.write("Later, this section will display:")

    st.write("- Assigned applicant cluster")
    st.write("- Cluster characteristics")
    st.write("- Similar applicant profile patterns")
    st.write("- Ethical / fairness notes")

    st.info(
        "This tool is intended for exploratory self-discovery only and does not make approval decisions."
    )