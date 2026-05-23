# Overview

This project uses unsupervised machine learning to group similar credit card applicants into clusters. The goal was not to predict approval or denial, but to help users explore what kind of applicant profile they most closely match based on anonymized application data.

The application was built with Streamlit and uses K-Means clustering on preprocessed applicant features.

# Preprocessing

The dataset contained anonymized columns labeled A1–A14 along with a binary Class column. Since this is an unsupervised learning project, the Class label was removed before clustering and only kept for later analysis.

Categorical columns were one-hot encoded so the clustering model would not incorrectly treat category values as ordered numbers. Continuous columns were scaled using StandardScaler so larger-valued features would not dominate clustering distance calculations.

Binary 0/1 columns were left unchanged.

# Model Choice

K-Means was chosen because it is simple, fast, and works well on structured tabular datasets. It also makes the clusters easier to explain and visualize.

PCA was used to reduce the feature space into two dimensions for visualization purposes.

Cluster quality was evaluated mainly using silhouette score and manual inspection of cluster separation.

# Feature Choices & Trade-offs

We removed A2 because it strongly resembled age-related data. Its range (~14–80) looked much more like a demographic feature than a financial one, so we excluded it to reduce possible age bias in clustering.

We also reviewed A1, A8, A9, and A11, but their meanings were too unclear to confidently label as protected attributes. Rather than removing features based mostly on assumptions, we chose a more conservative approach and only removed the strongest suspected demographic signal.

Some remaining features may still indirectly correlate with demographic traits through proxy relationships. However, removing too many variables risked weakening the financial structure of the dataset and reducing cluster quality.

# Limitations

Because the dataset is anonymized, feature meanings cannot be fully confirmed. Some fairness decisions were based on statistical patterns rather than certainty.

The dataset is also relatively small, and clustering does not produce objectively correct answers. The usefulness of the model depends on whether the clusters reveal meaningful applicant patterns.

# Responsible AI Statement

This project is an exploratory clustering tool, not a real credit approval system. It does not approve, deny, or recommend financial products. Users could still be harmed if they treat cluster results as guarantees or financial advice, so the interface clearly states that the tool is exploratory only.

We reviewed the dataset for possible demographic bias and removed A2 because it strongly resembled age-related information. Since the dataset is anonymized, some proxy relationships may still remain. Because of this, the system should not be used for real-world financial decision-making without deeper fairness testing and validation.
