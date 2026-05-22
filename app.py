import streamlit as st

st.set_page_config(
    page_title="Find Your Cluster",
    layout="wide"
)

def category_options(name, count):
    return [f"{name} {i}" for i in range(1, count + 1)]

st.title("Find Your Applicant Cluster")

st.write(
    "Explore which applicant profile group you most resemble using anonymized credit application data."
)

st.info(
    "Field meanings are inferred from anonymized dataset metadata. "
    "Some categories are intentionally labeled generically because the dataset does not reveal their exact meanings."
)

st.header("Applicant Information")

with st.form("applicant_form"):

    st.subheader("Personal / Background")

    a1 = st.pills(
        "A1 – Likely Gender",
        options=["Category 0", "Category 1"],
        default="Category 0"
    )

    a2 = st.slider(
        "A2 – Likely Age",
        min_value=14,
        max_value=80,
        value=30
    )

    a4 = st.pills(
        "A4 – Likely Marital Status",
        options=category_options("Marital Category", 3),
        default="Marital Category 1"
    )

    a5 = st.pills(
        "A5 – Likely Applicant Category",
        options=category_options("Applicant Type", 14),
        default="Applicant Type 1"
    )

    a6 = st.pills(
        "A6 – Likely Employment Type",
        options=category_options("Employment Type", 8),
        default="Employment Type 1"
    )

    a12 = st.pills(
        "A12 – Likely Region / Citizenship",
        options=category_options("Region Group", 3),
        default="Region Group 1"
    )

    st.subheader("Financial / Credit Profile")

    a3 = st.slider(
        "A3 – Likely Debt / Balance",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    a7 = st.slider(
        "A7 – Likely Years Employed",
        min_value=0.0,
        max_value=40.0,
        value=5.0
    )

    a8 = st.pills(
        "A8 – Likely Prior Default",
        options=["No Prior Default", "Prior Default"],
        default="No Prior Default"
    )

    a9 = st.pills(
        "A9 – Likely Currently Employed",
        options=["Not Currently Employed", "Currently Employed"],
        default="Currently Employed"
    )

    a10 = st.slider(
        "A10 – Likely Credit Metric",
        min_value=0,
        max_value=25,
        value=5
    )

    a11 = st.pills(
        "A11 – Likely Risk Flag",
        options=["Flag 0", "Flag 1"],
        default="Flag 0"
    )

    a13 = st.slider(
        "A13 – Unknown Financial Metric",
        min_value=0,
        max_value=280,
        value=20
    )

    a14 = st.slider(
        "A14 – Likely Income",
        min_value=1,
        max_value=100000,
        value=5000
    )

    submitted = st.form_submit_button("Find My Cluster")

if submitted:

    st.header("Cluster Result")

    st.success("Model connection coming next.")

    st.write("Later, this section will display:")

    st.write("- Assigned applicant cluster")
    st.write("- Cluster characteristics")
    st.write("- Similar applicant profile patterns")
    st.write("- Ethical / fairness notes")

    st.info(
        "This tool is intended for exploratory self-discovery only and does not make approval decisions."
    )