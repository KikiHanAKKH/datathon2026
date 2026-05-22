# Find Your Cluster

Find Your Cluster is a clustering tool built for Pillar 02 (USML) of Datathon 2026.

The app groups applicants with similar financial and demographic patterns using anonymized credit application data. Instead of predicting approvals, the goal is to help users explore which applicant profile they most resemble.

## Features

- Streamlit web app
- K-Means clustering
- Interactive applicant input UI
- Cluster exploration and interpretation

## Dataset

The dataset contains anonymized features (`A1–A14`) with approximate meanings inferred from feature distributions and dataset characteristics.

Examples:

| Feature | Likely Meaning |
|---|---|
| A2 | Age |
| A3 | Debt / Balance |
| A7 | Years Employed |
| A14 | Income |

## Ethics

This tool is exploratory only.

It does not:
- approve applications,
- reject applications,
- make financial decisions.

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python • Pandas • Scikit-learn • Streamlit